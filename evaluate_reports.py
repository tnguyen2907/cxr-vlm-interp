"""Score saved Findings reports without rerunning generation."""

import argparse
import json
from pathlib import Path
import shutil
from types import MethodType

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from experiment_utils import RANDOM_STATE, TARGET_LABELS, configure_runtime, resolve_path, run_log
from report_generation import COHORT_COLUMNS, RESULT_KEY, atomic_save, package_versions
from resource_monitor import Measure

CHEXBERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
EVAL_BATCH_SIZE = 16


def bert_inputs_with_special_tokens(tokenizer, token_ids_0, token_ids_1=None):
    output = [tokenizer.cls_token_id, *token_ids_0, tokenizer.sep_token_id]
    if token_ids_1 is not None:
        output.extend([*token_ids_1, tokenizer.sep_token_id])
    return output


def enable_legacy_scorer_compat():
    """Bridge the two legacy scorer packages to Transformers 5.8-5.9."""
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
        def encode_plus(tokenizer, *args, **kwargs):
            return tokenizer._encode_plus(*args, **kwargs)
        PreTrainedTokenizerBase.encode_plus = encode_plus

    # RadGraph's vendored AllenNLP may forward this bool to AutoTokenizer,
    # where it collides with the tokenizer method in newer Transformers.
    from radgraph.allennlp.common import cached_transformers
    if not getattr(cached_transformers.get_tokenizer, "_cxr_compat", False):
        original = cached_transformers.get_tokenizer

        def get_tokenizer(model_name, **kwargs):
            kwargs.pop("add_special_tokens", None)
            tokenizer = original(model_name, **kwargs)
            if not hasattr(tokenizer, "build_inputs_with_special_tokens"):
                tokenizer.build_inputs_with_special_tokens = MethodType(
                    bert_inputs_with_special_tokens, tokenizer,
                )
            return tokenizer

        get_tokenizer._cxr_compat = True
        cached_transformers.get_tokenizer = get_tokenizer


class ReportScorer:
    def __init__(self, device="cuda"):
        enable_legacy_scorer_compat()
        from radgraph import F1RadGraph
        from f1chexbert import F1CheXbert
        from f1chexbert.f1chexbert import CACHE_DIR
        from huggingface_hub import hf_hub_download
        # f1chexbert 0.0.2 expects a flat file, but its old force_filename download
        # no longer creates one with modern huggingface_hub.
        checkpoint = Path(CACHE_DIR) / "chexbert.pth"
        if not checkpoint.exists():
            source = hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, checkpoint)
        self.chexbert = F1CheXbert(device=device)
        if self.chexbert.target_names != CHEXBERT_LABELS:
            raise ValueError("Unexpected CheXbert label order.")
        self.radgraph = F1RadGraph(model_type="radgraph-xl", reward_level="partial",
                                  cuda=0 if device == "cuda" else -1)

    @torch.inference_mode()
    def labels(self, reports):
        texts = pd.Series(reports, dtype=str).str.strip().str.replace("\n", " ", regex=False)
        texts = texts.str.replace(r"\s+", " ", regex=True).str.strip().tolist()
        encoded = self.chexbert.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        ids = encoded["input_ids"].to(self.chexbert.device)
        mask = encoded["attention_mask"].to(self.chexbert.device)
        outputs = self.chexbert.model(ids, mask.float())
        # Native class IDs: 1 positive, 2 negative, 3 uncertain, 0 unmentioned.
        return torch.stack([head.argmax(-1).eq(1) for head in outputs], dim=1).cpu().numpy().astype(np.int8)

    @torch.inference_mode()
    def annotations(self, reports):
        output = self.radgraph.radgraph(reports)
        return [output[str(i)] for i in range(len(reports))]


def score_reports(frame, scorer, reference_cache=None, batch_size=EVAL_BATCH_SIZE):
    """Use library inference/rewards; cache reference annotations across conditions."""
    from radgraph.rewards import compute_reward
    reference_cache = {} if reference_cache is None else reference_cache
    references = list(dict.fromkeys(frame.reference_text))
    missing = [text for text in references if text not in reference_cache]
    for start in tqdm(range(0, len(missing), batch_size), desc="Reference scorers"):
        texts = missing[start:start + batch_size]
        labels, annotations = scorer.labels(texts), scorer.annotations(texts)
        reference_cache.update({text: {"labels": label.tolist(), "annotation": annotation}
                                for text, label, annotation in zip(texts, labels, annotations)})
    rows = []
    for start in tqdm(range(0, len(frame), batch_size), desc="Generated-report scorers"):
        block = frame.iloc[start:start + batch_size]
        texts = block.generated_text.tolist()
        # Empty generations predict no positive findings and receive RadGraph=0.
        nonempty = [i for i, text in enumerate(texts) if text.strip()]
        predicted = np.zeros((len(texts), len(CHEXBERT_LABELS)), dtype=np.int8)
        annotations = {}
        if nonempty:
            nonempty_texts = [texts[i] for i in nonempty]
            predicted[nonempty] = scorer.labels(nonempty_texts)
            annotations = dict(zip(nonempty, scorer.annotations(nonempty_texts)))
        for i, row in enumerate(block.to_dict("records")):
            reference = reference_cache[row["reference_text"]]
            result = {k: row[k] for k in RESULT_KEY + ["subject_id", "generated_tokens", "finish_reason"]}
            result["empty_generation"] = not bool(row["generated_text"].strip())
            result["radgraph_f1"] = float(compute_reward(annotations[i], reference["annotation"], "partial")) if i in annotations else 0.0
            for j, label in enumerate(CHEXBERT_LABELS):
                result["reference_" + label] = reference["labels"][j]
                result["predicted_" + label] = int(predicted[i, j])
            rows.append(result)
    return pd.DataFrame(rows), reference_cache


def aggregate_metrics(frame, weights=None):
    weights = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype=float)
    true = frame[["reference_" + label for label in CHEXBERT_LABELS]].to_numpy(dtype=bool)
    pred = frame[["predicted_" + label for label in CHEXBERT_LABELS]].to_numpy(dtype=bool)
    tp = ((true & pred) * weights[:, None]).sum(axis=0)
    fp = ((~true & pred) * weights[:, None]).sum(axis=0)
    fn = ((true & ~pred) * weights[:, None]).sum(axis=0)
    denominator = 2 * tp + fp + fn
    per_label = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    result = {"radgraph_f1": np.average(frame.radgraph_f1, weights=weights)}
    for name, indices in {
        "all": list(range(14)), "trained5": [CHEXBERT_LABELS.index(label) for label in TARGET_LABELS],
        "remaining9": [i for i, label in enumerate(CHEXBERT_LABELS) if label not in TARGET_LABELS],
    }.items():
        total = denominator[indices].sum()
        result[f"chexbert_{name}_micro_f1"] = float(2 * tp[indices].sum() / total) if total else 0.0
        result[f"chexbert_{name}_macro_f1"] = float(per_label[indices].mean())
    result.update({"chexbert_" + label + "_f1": float(value) for label, value in zip(CHEXBERT_LABELS, per_label)})
    return result


def summarize(frame, resamples=1000):
    """Paired patient bootstrap: identical sampled patients for every comparison."""
    summaries, differences = [], []
    for order, ordered in frame.groupby("prompt_order", sort=False):
        groups = {name: part.sort_values("study_id").reset_index(drop=True)
                  for name, part in ordered.groupby("model_name", sort=False)}
        first = next(iter(groups.values()))
        for part in groups.values():
            pd.testing.assert_frame_equal(part[["study_id", "subject_id"]], first[["study_id", "subject_id"]])
        point = {name: aggregate_metrics(part) for name, part in groups.items()}
        draws = {name: [] for name in groups}
        patients, patient_index = np.unique(first.subject_id, return_inverse=True)
        rng = np.random.default_rng(RANDOM_STATE)
        for _ in range(resamples):
            counts = np.bincount(rng.integers(len(patients), size=len(patients)), minlength=len(patients))
            weights = counts[patient_index]
            for name, part in groups.items():
                draws[name].append(aggregate_metrics(part, weights))
        for name, metrics in point.items():
            for metric, value in metrics.items():
                values = [draw[metric] for draw in draws[name]]
                low, high = np.quantile(values, [0.025, 0.975]) if values else (np.nan, np.nan)
                summaries.append({"model_name": name, "prompt_order": order, "metric": metric,
                                  "value": value, "ci_low": low, "ci_high": high, "num_studies": len(first)})
                if "base_medgemma" in groups and name != "base_medgemma":
                    deltas = [a[metric] - b[metric] for a, b in zip(draws[name], draws["base_medgemma"])]
                    low, high = np.quantile(deltas, [0.025, 0.975]) if deltas else (np.nan, np.nan)
                    differences.append({"model_name": name, "prompt_order": order, "metric": metric,
                                        "difference_vs_base": value - point["base_medgemma"][metric],
                                        "ci_low": low, "ci_high": high})
    return pd.DataFrame(summaries), pd.DataFrame(differences, columns=[
        "model_name", "prompt_order", "metric", "difference_vs_base", "ci_low", "ci_high"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=resolve_path, required=True, help="Existing generation run")
    parser.add_argument("--output-root", type=resolve_path, required=True, help="Fresh evaluation run")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 0:
        parser.error("Invalid batch size or bootstrap count.")
    with run_log(args.output_root):
        run(args)


def run(args):
    source = args.input_root / "report_generation"
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    cohort = pd.read_csv(source / "cohort.csv", dtype=str, keep_default_na=False)
    frame = pd.read_csv(source / "generations.csv", dtype={c: str for c in COHORT_COLUMNS}, keep_default_na=False)
    expected = {(spec["name"], order, study) for spec in metadata["settings"]["models"]
                for order in metadata["settings"]["orders"] for study in cohort.study_id}
    if frame.duplicated(RESULT_KEY).any() or set(frame[RESULT_KEY].itertuples(index=False, name=None)) != expected:
        raise ValueError("Generation run is incomplete or duplicated. Resume generation before evaluation.")
    for _, part in frame.groupby(["model_name", "prompt_order"]):
        wanted = cohort.set_index("study_id").loc[part.study_id].reset_index()
        pd.testing.assert_frame_equal(part[COHORT_COLUMNS].reset_index(drop=True).astype(str), wanted.astype(str))
    configure_runtime(require_gpu=args.device == "cuda")
    output = args.output_root / "report_generation"
    output.mkdir()
    atomic_save({"generation_run": str(args.input_root), "versions": package_versions(),
                 "radgraph_model": "radgraph-xl", "reward_level": "partial",
                 "chexbert_positive_class": 1, "empty_generation": "RadGraph=0; all CheXbert positives=0",
                 "bootstrap_samples": args.bootstrap_samples, "bootstrap_unit": "subject_id", "seed": RANDOM_STATE},
                output / "evaluation_metadata.json")
    scorer = ReportScorer(args.device)
    reference_cache, parts = {}, []
    for (name, order), part in frame.groupby(["model_name", "prompt_order"], sort=False):
        with Measure("evaluate_" + name + "_" + order, args.output_root):
            scores, reference_cache = score_reports(part, scorer, reference_cache, args.batch_size)
        parts.append(scores)
        atomic_save(pd.concat(parts, ignore_index=True), output / "per_study_metrics.csv")
        atomic_save(reference_cache, output / "reference_annotations.json")
    scores = pd.concat(parts, ignore_index=True)
    summary, differences = summarize(scores, args.bootstrap_samples)
    atomic_save(summary, output / "metrics.csv")
    atomic_save(differences, output / "paired_differences.csv")
    quality = frame.assign(empty_generation=frame.generated_text.str.strip().eq(""),
                           length_capped=frame.finish_reason.eq("length")).groupby(["model_name", "prompt_order"]).agg(
        num_studies=("study_id", "size"), empty_rate=("empty_generation", "mean"),
        length_capped_rate=("length_capped", "mean"), mean_generated_tokens=("generated_tokens", "mean"))
    atomic_save(quality.reset_index(), output / "generation_quality.csv")
    # A deterministic, matched sample for manual review, not an automatic clinical judgment.
    study_sample = cohort.sample(n=min(20, len(cohort)), random_state=RANDOM_STATE).study_id
    atomic_save(frame[frame.study_id.isin(study_sample)].sort_values(["study_id", "model_name", "prompt_order"]),
                output / "qualitative_review.csv")
    print(summary.to_string(index=False))
    print("Paired differences", differences.to_string(index=False), sep="\n")
    print(quality.to_string())


if __name__ == "__main__":
    main()
