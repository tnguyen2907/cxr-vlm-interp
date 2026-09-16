"""Generate held-out Findings reports; benchmarks import the same inference code."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from experiment_utils import (
    DATA_CSV, IMPORTED_RUN_ROOT, MEDGEMMA_MODEL_ID, MODEL_DTYPE, PROMPT_ORDERS,
    RANDOM_STATE, configure_runtime, load_rgb, resolve_path, run_log,
)
from resource_monitor import Measure

REPORT_CSV = Path("/opt/gpudata/cxr/chexpertplus/report.csv")
REPORT_INSTRUCTION = (
    "Describe the radiographic findings in this chest X-ray. "
    "Write only the Findings section of a radiology report."
)
MODEL_NAMES = ["base_medgemma", "lora_image_first", "lora_text_first"]
MAX_NEW_TOKENS = 512
MAX_MODEL_LEN = 2048
HF_BATCH_SIZE = 8
SUBMISSION_SIZE = 256
RESULT_KEY = ["model_name", "prompt_order", "study_id"]
COHORT_COLUMNS = ["study_id", "subject_id", "image_path", "reference_text"]
RESULT_COLUMNS = COHORT_COLUMNS + ["model_name", "prompt_order", "generated_text", "finish_reason", "generated_tokens"]


def read_cohort(manifest=DATA_CSV, reports=REPORT_CSV, split="test", rows=None):
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if frame.study_id.duplicated().any() or frame["study_id"].eq("").any():
        raise ValueError("The manifest must contain one nonempty study_id per row.")
    if frame.groupby("subject_id").probe_split.nunique().gt(1).any():
        raise ValueError("Manifest patients overlap between splits.")
    frame = frame.loc[frame.probe_split.eq(split), ["study_id", "subject_id", "image_path"]]
    refs = pd.read_csv(reports, usecols=["study_id", "findings"], dtype=str, keep_default_na=False)
    refs = refs.drop_duplicates()
    if refs.study_id.duplicated().any():
        raise ValueError("Conflicting reference findings for the same study_id.")
    joined = frame.merge(refs, on="study_id", how="left", validate="one_to_one", indicator=True)
    missing = joined["_merge"].ne("both")
    empty = joined.findings.fillna("").str.strip().eq("")
    print("Report coverage", {"split": split, "studies": len(frame),
                              "missing_report": int(missing.sum()),
                              "empty_findings": int((~missing & empty).sum()),
                              "eligible": int((~empty).sum())})
    joined = joined.loc[~empty].rename(columns={"findings": "reference_text"})[COHORT_COLUMNS]
    if rows is not None:
        joined = joined.sample(n=min(rows, len(joined)), random_state=RANDOM_STATE)
    if joined.empty or joined.subject_id.eq("").any():
        raise ValueError("No eligible reports, or missing patient IDs.")
    return joined.reset_index(drop=True)


def report_messages(order, answer=None):
    image = {"type": "image"}
    text = {"type": "text", "text": REPORT_INSTRUCTION}
    messages = [{"role": "user", "content": [image, text] if order == "image_first" else [text, image]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def report_prompt(processor, order, answer=None):
    return processor.apply_chat_template(
        report_messages(order, answer), tokenize=False, add_generation_prompt=answer is None,
    )


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if isinstance(value, pd.DataFrame):
        value.to_csv(temporary, index=False)
    else:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def package_versions():
    result = {}
    for package in ("torch", "transformers", "peft", "vllm", "trl", "radgraph", "f1chexbert"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def selected_models(args):
    if args.model == "custom":
        if not args.model_name:
            raise ValueError("--model custom requires --model-name.")
        checkpoint = str(resolve_path(args.checkpoint)) if resolve_path(args.checkpoint).exists() else args.checkpoint
        return [{"name": args.model_name, "checkpoint": checkpoint,
                 "adapter": str(args.adapter_path) if args.adapter_path else None}]
    if args.adapter_path or args.model_name or args.checkpoint != MEDGEMMA_MODEL_ID:
        raise ValueError("Checkpoint/adapter/name overrides require --model custom.")
    names = MODEL_NAMES if args.model == "all" else [args.model]
    return [{"name": name, "checkpoint": MEDGEMMA_MODEL_ID,
             "adapter": None if name == "base_medgemma" else
             str(args.adapter_root / (name.removeprefix("lora_") + "_adapter"))} for name in names]


@contextmanager
def generator(spec, engine="transformers", prefix_caching=False):
    """One model at a time; keep the engine alive across submission groups."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
    processor = AutoProcessor.from_pretrained(spec["checkpoint"])
    processor.tokenizer.padding_side = "left"
    config = GenerationConfig.from_pretrained(spec["checkpoint"])
    stops = config.eos_token_id or processor.tokenizer.eos_token_id
    stops = stops if isinstance(stops, list) else [stops]
    print("Resolved model", spec, "stop_token_ids", stops)
    if spec["adapter"]:
        if not (Path(spec["adapter"]) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter: {spec['adapter']}")
    if engine == "vllm":
        # configure_runtime/telemetry can initialize CUDA before engine creation.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM
        rank = 16
        if spec["adapter"]:
            rank = json.loads((Path(spec["adapter"]) / "adapter_config.json").read_text())["r"]
        model = LLM(
            model=spec["checkpoint"], dtype="bfloat16", tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"do_pan_and_scan": False},
            enable_prefix_caching=prefix_caching, enable_chunked_prefill=True,
            enable_lora=bool(spec["adapter"]), max_lora_rank=max(8, 2 ** (rank - 1).bit_length()),
            seed=RANDOM_STATE, disable_log_stats=False, generation_config="vllm",
        )
        print("vLLM resolved model config", model.llm_engine.model_config)
    else:
        model = AutoModelForImageTextToText.from_pretrained(spec["checkpoint"], dtype=MODEL_DTYPE).cuda().eval()
        if spec["adapter"]:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, spec["adapter"]).eval()
    try:
        yield model, processor, stops
    finally:
        if engine == "vllm":
            model.llm_engine.engine_core.shutdown()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def generate_reports(model, processor, stops, frame, order, *, engine="transformers", adapter=None,
                     batch_size=HF_BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS, diagnostics=False):
    """Return a submission's reports in input order. Empty reports are valid results."""
    prompt = report_prompt(processor, order)
    results = []
    if frame.empty:
        return results
    # vLLM receives the entire submission; its scheduler chooses GPU batches.
    step = len(frame) if engine == "vllm" else batch_size
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(frame), step):
            images = []
            tick = perf_counter()
            try:
                for image in pool.map(load_rgb, frame.image_path.iloc[start:start + step]):
                    images.append(image)
                if engine == "vllm":
                    from vllm import SamplingParams
                    from vllm.lora.request import LoRARequest
                    requests = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in images]
                    outputs = model.generate(
                        requests, SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                                 stop_token_ids=stops, logprobs=10 if diagnostics else None),
                        lora_request=LoRARequest("report_adapter", 1, adapter) if adapter else None,
                        tokenization_kwargs={"add_special_tokens": False}, use_tqdm=True,
                    )
                    for output in outputs:
                        generated = output.outputs[0]
                        row = {"generated_text": generated.text, "generated_tokens": len(generated.token_ids),
                               "finish_reason": generated.finish_reason}
                        if diagnostics:
                            row["prompt_ids"] = list(output.prompt_token_ids)
                            row["first_logprobs"] = {int(k): float(v.logprob) for k, v in generated.logprobs[0].items()}
                            row["first_token"] = generated.token_ids[0]
                        results.append(row)
                    del requests, outputs
                else:
                    inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt",
                                       padding=True, add_special_tokens=False, do_pan_and_scan=False)
                    if inputs["input_ids"].shape[1] + max_new_tokens > MAX_MODEL_LEN:
                        raise ValueError("Prompt + generation exceeds the context budget; no truncation is allowed.")
                    inputs = {k: v.to(device=model.device, dtype=MODEL_DTYPE) if v.is_floating_point()
                              else v.to(model.device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        if diagnostics:
                            first = model(**inputs, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
                            values, indices = first.topk(10)
                        from transformers import GenerationConfig
                        tokens = model.generate(
                            **inputs, generation_config=GenerationConfig(
                                do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                                eos_token_id=stops, pad_token_id=processor.tokenizer.pad_token_id,
                            ), logits_to_keep=1,
                        )[:, inputs["input_ids"].shape[1]:].cpu().tolist()
                    for i, row_tokens in enumerate(tokens):
                        end = next((j + 1 for j, token in enumerate(row_tokens) if token in stops), len(row_tokens))
                        row_tokens = row_tokens[:end]
                        row = {"generated_text": processor.tokenizer.decode(row_tokens, skip_special_tokens=True),
                               "generated_tokens": end, "finish_reason": "stop" if row_tokens[-1] in stops else "length"}
                        if diagnostics:
                            row["prompt_ids"] = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].cpu().tolist()
                            row["first_logprobs"] = dict(zip(indices[i].cpu().tolist(), values[i].cpu().tolist()))
                            row["first_token"] = row_tokens[0]
                        results.append(row)
                    del inputs, tokens
                    if diagnostics:
                        del first, values, indices
                print("Generation batch", {"rows": len(images), "wall_sec": perf_counter() - tick}, flush=True)
            finally:
                for image in images:
                    image.close()
                images.clear()
    return results


def load_progress(directory, cohort, settings, resume):
    """Saved CSV rows, not separate checkpoints, are the completion record."""
    metadata = directory / "metadata.json"
    cohort_path = directory / "cohort.csv"
    result_path = directory / "generations.csv"
    if resume and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["settings"] != settings:
            raise ValueError("Resume settings differ; use the original arguments or a fresh run.")
        old = pd.read_csv(cohort_path, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(old, cohort.astype(str), check_dtype=False)
    elif result_path.exists():
        raise ValueError("Generations exist without run metadata; cannot safely resume.")
    else:
        atomic_save(cohort, cohort_path)
        atomic_save({"settings": settings, "versions": package_versions()}, metadata)
    if not result_path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pd.read_csv(result_path, dtype={c: str for c in RESULT_COLUMNS if c != "generated_tokens"}, keep_default_na=False)
    if results.duplicated(RESULT_KEY).any():
        raise ValueError("Duplicate generation keys in saved results.")
    counts = pd.to_numeric(results.generated_tokens)
    if not results.finish_reason.isin(["stop", "length"]).all() or not (counts.ge(0) & counts.le(MAX_NEW_TOKENS)).all():
        raise ValueError("Incomplete/invalid saved generation rows.")
    expected = {(s["name"], order, study) for s in settings["models"] for order in settings["orders"] for study in cohort.study_id}
    if not set(results[RESULT_KEY].itertuples(index=False, name=None)).issubset(expected):
        raise ValueError("Saved generations do not belong to this cohort/model selection.")
    for _, block in results.groupby(["model_name", "prompt_order"]):
        wanted = cohort.set_index("study_id").loc[block.study_id].reset_index()
        pd.testing.assert_frame_equal(block[COHORT_COLUMNS].reset_index(drop=True).astype(str), wanted.astype(str))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=resolve_path, default=DATA_CSV)
    parser.add_argument("--reports", type=resolve_path, default=REPORT_CSV)
    parser.add_argument("--output-root", type=resolve_path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coverage-only", action="store_true", help="Join and print coverage without loading models")
    parser.add_argument("--split", choices=["test", "train"], default="test", help="train is for saved benchmark pilots only")
    parser.add_argument("--model", choices=["all", "custom"] + MODEL_NAMES, default="all")
    parser.add_argument("--prompt-order", choices=["all"] + PROMPT_ORDERS, default="all")
    parser.add_argument("--adapter-root", type=resolve_path, default=IMPORTED_RUN_ROOT / "lora_sft")
    parser.add_argument("--checkpoint", default=MEDGEMMA_MODEL_ID, help="Checkpoint for --model custom")
    parser.add_argument("--adapter-path", type=resolve_path, help="Optional adapter for --model custom")
    parser.add_argument("--model-name", help="Result name for --model custom")
    parser.add_argument("--engine", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--submission-size", type=int, default=SUBMISSION_SIZE, help="0 queues all pending studies")
    parser.add_argument("--batch-size", type=int, default=HF_BATCH_SIZE, help="Transformers GPU batch size only")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--rows", type=int, help="Optional seeded subset for a pilot, not the full experiment")
    args = parser.parse_args()
    if args.submission_size < 0 or args.batch_size < 1 or (args.rows is not None and args.rows < 1):
        parser.error("Invalid submission, batch, or row count.")
    with run_log(args.output_root, resume=args.resume):
        run(args)


def run(args):
    cohort = read_cohort(args.manifest, args.reports, split=args.split, rows=args.rows)
    if args.coverage_only:
        print(cohort.drop(columns="reference_text").head().to_string(index=False))
        return
    models = selected_models(args)
    orders = PROMPT_ORDERS if args.prompt_order == "all" else [args.prompt_order]
    directory = args.output_root / "report_generation"
    directory.mkdir(exist_ok=True)
    settings = {"models": models, "orders": orders, "engine": args.engine, "split": args.split,
                "instruction": REPORT_INSTRUCTION, "max_new_tokens": MAX_NEW_TOKENS,
                "max_model_len": MAX_MODEL_LEN, "dtype": "bfloat16", "seed": RANDOM_STATE,
                "submission_size": args.submission_size, "batch_size": args.batch_size,
                "prefix_caching": args.prefix_caching, "pan_and_scan": False}
    results = load_progress(directory, cohort, settings, args.resume)
    expected = len(cohort) * len(models) * len(orders)
    print("Completed", len(results), "pending", expected - len(results))
    if len(results) == expected:
        return
    configure_runtime()
    print("Versions", package_versions())
    for spec in models:
        pending_orders = {order: cohort.loc[~cohort.study_id.isin(results.loc[
            results.model_name.eq(spec["name"]) & results.prompt_order.eq(order), "study_id"])] for order in orders}
        if not any(len(block) for block in pending_orders.values()):
            continue
        load_start = perf_counter()
        with generator(spec, args.engine, args.prefix_caching) as (model, processor, stops):
            print("Model load/compile sec", perf_counter() - load_start)
            for order, pending in pending_orders.items():
                size = args.submission_size or max(len(pending), 1)
                for start in range(0, len(pending), size):
                    block = pending.iloc[start:start + size]
                    with Measure(spec["name"] + "_" + order, args.output_root):
                        outputs = generate_reports(model, processor, stops, block, order, engine=args.engine,
                                                   adapter=spec["adapter"], batch_size=args.batch_size)
                    fresh = block.reset_index(drop=True).assign(model_name=spec["name"], prompt_order=order)
                    fresh = pd.concat([fresh, pd.DataFrame(outputs)], axis=1)[RESULT_COLUMNS]
                    results = pd.concat([results, fresh], ignore_index=True)
                    atomic_save(results, directory / "generations.csv")
                    print("Saved", len(results), "of", expected, flush=True)
        # Drop the caller's reference before loading the next model.
        del model, processor, stops
        gc.collect()
        torch.cuda.empty_cache()
    print("Complete. Score saved reports separately with evaluate_reports.py.")


if __name__ == "__main__":
    main()
