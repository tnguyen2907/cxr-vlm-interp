"""Train text-first report-only or concatenated report/classification LoRA."""

import argparse
import gc

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, set_seed

from experiment_utils import (
    DATA_CSV, MEDGEMMA_MODEL_ID, MODEL_DTYPE, RANDOM_STATE, TARGET_LABELS,
    add_token_types, answer_token_ids, configure_runtime, prepare_pixels,
    resolve_path, run_log, tokenize_prompts,
)
from report_generation import REPORT_CSV, read_cohort, report_prompt

EFFECTIVE_BATCH_SIZE = 64
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 1e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
DATALOADER_NUM_WORKERS = 4
DECODER_LINEAR_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
OBJECTIVES = ["report", "concatenated"]


def report_training_rows(cohort, processor, max_length):
    """Tokenize Findings targets and mask every image/prompt token."""
    def token_ids(text):
        text = text.replace(processor.boi_token, processor.full_image_sequence)
        return processor.tokenizer(text, add_special_tokens=False).input_ids

    prefix = token_ids(report_prompt(processor, "text_first"))
    end_token = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_token is None or end_token == processor.tokenizer.unk_token_id:
        raise ValueError("Gemma end-of-turn token is missing.")

    rows = []
    for image_idx, row in enumerate(cohort.itertuples(index=False)):
        full = token_ids(report_prompt(processor, "text_first", row.reference_text))
        if full[:len(prefix)] != prefix:
            raise ValueError("Assistant boundary retokenized; report-only masking needs review.")
        end = full.index(end_token, len(prefix)) + 1
        full = full[:end]
        if len(full) > max_length:
            raise ValueError(f"Study {row.study_id} has {len(full)} tokens, above context limit {max_length}.")
        rows.append({
            "image_idx": image_idx,
            "study_id": row.study_id,
            "task": "report",
            "finding": "",
            "input_ids": full,
            "labels": [-100] * len(prefix) + full[len(prefix):],
        })

    image_id = getattr(processor, "image_token_id", None)
    if image_id is None or any(row["input_ids"].count(image_id) != 256 for row in rows):
        raise ValueError("Every report-SFT row must contain exactly 256 image placeholders.")
    lengths = pd.Series([len(row["input_ids"]) for row in rows])
    print("text_first report lengths", lengths.describe(percentiles=[0.5, 0.9, 0.99]).to_dict())
    return rows


def add_classification_labels(cohort, manifest):
    labels = pd.read_csv(manifest, dtype={"study_id": str})[["study_id", *TARGET_LABELS]]
    if labels.study_id.duplicated().any():
        raise ValueError("The manifest must contain one label row per study.")
    cohort = cohort.merge(labels, on="study_id", how="left", validate="one_to_one", sort=False)
    if cohort[TARGET_LABELS].isna().all(axis=1).any():
        raise ValueError("Classification labels are missing for a report-training study.")
    for label in TARGET_LABELS:
        cohort[label] = pd.to_numeric(cohort[label], errors="coerce").eq(1).astype("int8")
    return cohort.reset_index(drop=True)


def classification_training_rows(cohort, processor, max_length):
    """Build the existing text-first one-token yes/no classification examples."""
    prompts = tokenize_prompts(processor)
    yes_id, no_id = answer_token_ids(processor)
    prefixes = {
        label: prompts[("text_first", label)]["input_ids"][0].tolist()
        for label in TARGET_LABELS
    }
    rows = []
    for image_idx, row in cohort.iterrows():
        for label in TARGET_LABELS:
            answer = yes_id if row[label] == 1 else no_id
            full = prefixes[label] + [answer]
            if len(full) > max_length:
                raise ValueError(f"Study {row.study_id} classification prompt exceeds context limit {max_length}.")
            rows.append({
                "image_idx": image_idx,
                "study_id": row.study_id,
                "task": "classification",
                "finding": label,
                "input_ids": full,
                "labels": [-100] * (len(full) - 1) + [answer],
            })

    image_id = getattr(processor, "image_token_id", None)
    if image_id is None or any(row["input_ids"].count(image_id) != 256 for row in rows):
        raise ValueError("Every classification-SFT row must contain exactly 256 image placeholders.")
    return rows


def sft_training_rows(cohort, processor, max_length, objective):
    rows = report_training_rows(cohort, processor, max_length)
    if objective == "concatenated":
        rows.extend(classification_training_rows(cohort, processor, max_length))
    counts = pd.Series([row["task"] for row in rows]).value_counts().to_dict()
    print(objective, "training rows", len(rows), counts)
    return rows


class ReportCollator:
    def __init__(self, processor, pixels):
        self.processor, self.pixels = processor, pixels
        self.supervised_tokens = 0
        self.examples = 0

    def __call__(self, examples):
        width = max(len(row["input_ids"]) for row in examples)
        ids = torch.full((len(examples), width), self.processor.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, row in enumerate(examples):
            length = len(row["input_ids"])
            ids[i, :length] = torch.tensor(row["input_ids"])
            labels[i, :length] = torch.tensor(row["labels"])
            mask[i, :length] = 1
        batch = add_token_types(self.processor, {"input_ids": ids, "attention_mask": mask, "labels": labels})
        batch["pixel_values"] = self.pixels[[int(row["image_idx"]) for row in examples]]
        self.supervised_tokens += int(labels.ne(-100).sum())
        self.examples += len(examples)
        return batch


def decoder_lora_targets(model):
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "language_model" in name
        and "vision_tower" not in name
        and "multi_modal_projector" not in name
        and name.split(".")[-1] in DECODER_LINEAR_NAMES
    ]
    if not targets:
        raise ValueError("No decoder LoRA target modules were found.")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=resolve_path, default=DATA_CSV)
    parser.add_argument("--reports", type=resolve_path, default=REPORT_CSV)
    parser.add_argument("--output-root", type=resolve_path, required=True, help="Fresh runs/<run_name> folder")
    parser.add_argument("--microbatch", type=int, choices=[4, 8, 16, 32], required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="report")
    args = parser.parse_args()
    with run_log(args.output_root):
        run(args)


def run(args):
    from trl import SFTConfig, SFTTrainer

    configure_runtime()
    set_seed(RANDOM_STATE)
    cohort = read_cohort(args.manifest, args.reports, split="train")
    if len(cohort) != 20_000 or cohort.study_id.nunique() != 20_000:
        raise ValueError(f"Expected exactly 20,000 unique report-training studies, found {len(cohort)}.")
    if args.objective == "concatenated":
        cohort = add_classification_labels(cohort, args.manifest)

    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model_context = AutoConfig.from_pretrained(MEDGEMMA_MODEL_ID).text_config.max_position_embeddings
    rows = sft_training_rows(cohort, processor, model_context, args.objective)
    pixels = prepare_pixels(cohort.image_path.tolist(), processor)
    collator = ReportCollator(processor, pixels)
    sample = collator(rows[:2])
    assert sample["labels"].ne(-100).sum().item() == sum(
        token != -100 for row in rows[:2] for token in row["labels"]
    )
    assert (sample["labels"][sample["attention_mask"].eq(0)] == -100).all()
    del sample

    model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_MODEL_ID, dtype=MODEL_DTYPE)
    model.config.use_cache = False
    targets = decoder_lora_targets(model)
    print("Training rows", len(rows), "microbatch", args.microbatch,
          "gradient accumulation", EFFECTIVE_BATCH_SIZE // args.microbatch,
          "LoRA target modules", len(targets))

    output_name = "report_sft" if args.objective == "report" else "concatenated_sft"
    output_dir = args.output_root / output_name / "text_first_adapter"
    output_dir.mkdir(parents=True)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.microbatch,
            gradient_accumulation_steps=EFFECTIVE_BATCH_SIZE // args.microbatch,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            optim="adamw_torch",
            weight_decay=0.0,
            bf16=True,
            max_length=None,
            packing=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            dataset_kwargs={"skip_prepare_dataset": True},
            gradient_checkpointing=False,
            seed=RANDOM_STATE,
        ),
        train_dataset=Dataset.from_list(rows),
        data_collator=collator,
        processing_class=processor,
        peft_config=LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(output_dir / "train_log.csv", index=False)

    del trainer, model, pixels, rows
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
