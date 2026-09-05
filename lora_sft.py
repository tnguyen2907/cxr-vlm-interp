"""Train separate image-first and text-first decoder LoRA adapters."""

import argparse
import gc

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from experiment_utils import (
    DATA_CSV, MEDGEMMA_MODEL_ID, MODEL_DTYPE, PROMPT_ORDERS, RANDOM_STATE,
    TARGET_LABELS, add_token_types, answer_token_ids, configure_runtime,
    prepare_pixels, resolve_path, run_log, tokenize_prompts,
)

PER_DEVICE_TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 5e-5
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
DATALOADER_NUM_WORKERS = 4
DECODER_LINEAR_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class AnswerCollator:
    def __init__(self, processor, prompts, pixels):
        self.processor, self.prompts, self.pixels = processor, prompts, pixels
        self.yes_id, self.no_id = answer_token_ids(processor)

    def __call__(self, examples):
        rows = [self.prompts[(ex["prompt_order"], ex["finding"])]["input_ids"][0] for ex in examples]
        width = max(len(row) for row in rows) + 1
        ids = torch.full((len(rows), width), self.processor.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (row, example) in enumerate(zip(rows, examples)):
            answer = self.yes_id if example["answer"] == "yes" else self.no_id
            ids[i, :len(row)] = row
            ids[i, len(row)] = answer
            mask[i, :len(row) + 1] = 1
            labels[i, len(row)] = answer
        batch = add_token_types(self.processor, {"input_ids": ids, "attention_mask": mask, "labels": labels})
        batch["pixel_values"] = self.pixels[[int(ex["image_idx"]) for ex in examples]]
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=resolve_path, required=True, help="Fresh runs/<run_name> folder")
    parser.add_argument("--prompt-order", choices=["all"] + PROMPT_ORDERS, default="all")
    args = parser.parse_args()
    with run_log(args.output_root):
        run(args)


def run(args):
    from trl import SFTConfig, SFTTrainer
    configure_runtime()
    print("Manifest:", DATA_CSV)
    output_dir = args.output_root / "lora_sft"
    output_dir.mkdir()
    df = pd.read_csv(DATA_CSV)
    train_df = df[df.probe_split == "train"].reset_index(drop=True)
    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    prompts = tokenize_prompts(processor)
    pixels = prepare_pixels(train_df.image_path.tolist(), processor)
    collator = AnswerCollator(processor, prompts, pixels)
    orders = PROMPT_ORDERS if args.prompt_order == "all" else [args.prompt_order]
    for order in orders:
        dataset = Dataset.from_list([
            {"image_idx": i, "study_id": row.study_id, "prompt_order": order,
             "finding": label, "answer": "yes" if row[label] == 1 else "no"}
            for i, row in train_df.iterrows() for label in TARGET_LABELS
        ])
        sample = collator([dataset[0], dataset[1]])
        assert sample["labels"].ne(-100).sum(dim=1).tolist() == [1, 1]
        del sample
        model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_MODEL_ID, dtype=MODEL_DTYPE)
        model.config.use_cache = False
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and "language_model" in name
                   and "vision_tower" not in name and "multi_modal_projector" not in name
                   and name.split(".")[-1] in DECODER_LINEAR_NAMES]
        print(order, "examples", len(dataset), "LoRA target modules", len(targets))
        adapter_dir = output_dir / f"{order}_adapter"
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=str(adapter_dir), per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, num_train_epochs=NUM_TRAIN_EPOCHS,
                learning_rate=LEARNING_RATE, bf16=True, max_length=None, packing=False,
                warmup_ratio=0.03, lr_scheduler_type="cosine", max_grad_norm=1.0,
                logging_steps=20, save_strategy="epoch", save_total_limit=1, report_to="none",
                remove_unused_columns=False, dataloader_num_workers=DATALOADER_NUM_WORKERS,
                dataset_kwargs={"skip_prepare_dataset": True}, gradient_checkpointing=False, seed=RANDOM_STATE,
            ),
            train_dataset=dataset, data_collator=collator, processing_class=processor,
            peft_config=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                  bias="none", task_type="CAUSAL_LM", target_modules=targets),
        )
        trainer.model.print_trainable_parameters()
        trainer.train()
        trainer.save_model(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir))
        pd.DataFrame(trainer.state.log_history).to_csv(output_dir / f"{order}_train_log.csv", index=False)
        del trainer, model, dataset
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
