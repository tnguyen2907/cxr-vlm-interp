"""Shared linear/MHA extraction and training on original MedGemma representations."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, ExitStack
import gc
from pathlib import Path
import pickle
import re
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import h5py
from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from peft import PeftModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, AutoProcessor
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead

from resource_monitor import Measure, resource_snapshot
from experiment_utils import (
    ACTIVATION_ROOT, DATA_CSV, MEDGEMMA_MODEL_ID, MEDSIGLIP_MODEL_ID, MODEL_DTYPE,
    IMAGE_LOAD_NUM_WORKERS, IMAGE_PROCESS_BATCH_SIZE,
    IMPORTED_RUN_ROOT, PROMPT_ORDERS, RANDOM_STATE, TARGET_LABELS, answer_token_ids,
    cached_image_forward, configure_runtime, image_batches, image_token_id,
    resolve_path, run_log, tokenize_prompts,
)

MEDGEMMA_BATCH_SIZE = 24
MEDSIGLIP_BATCH_SIZE = 512
LINEAR_PARALLEL_JOBS = 4
LINEAR_INNER_NUM_THREADS = 4
MAX_ITER = 2000
MHA_BATCH_SIZE = 512
MHA_EPOCHS = 30
MHA_LEARNING_RATE = 5e-5
MHA_WEIGHT_DECAY = 1e-4
MHA_NUM_HEADS = 20
MHA_MLP_DIM = 10240
LAYER_BLOCK_SIZE = 5
MODEL_NAMES = ["base_medgemma", "lora_image_first", "lora_text_first"]
POOLED_FEATURES = ["medgemma_layer_mean_image_token", "medgemma_layer_last_image_token",
                   "medgemma_layer_final_prompt_token"]
CONTROL_FEATURES = ["medgemma_pre_projector_mean_image_token", "medgemma_pre_decoder_mean_image_token"]
MHA_FEATURE = "mha_pooled_image_token"
YES_NO_FEATURE = "medgemma_yes_no_logprob"
RESULT_KEY = ["model_name", "prompt_order", "condition", "label", "feature", "layer"]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def load_medgemma(adapter_path=None, device="cuda:0"):
    model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_MODEL_ID, dtype=MODEL_DTYPE)
    if adapter_path is not None:
        print("Adapter", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        # The cache is shared only with adapters that alter decoder projections.
        changed = [name for name in model.state_dict() if "lora_" in name or "modules_to_save" in name]
        if not changed or any("language_model.layers." not in name for name in changed):
            raise ValueError("This adapter may change the cached vision/projector features.")
    return model.to(device).eval().requires_grad_(False)


def select_layers(config, selection):
    config = config.text_config
    count = config.num_hidden_layers
    if selection == "all":
        return list(range(count))
    if selection == "global":
        types = getattr(config, "layer_types", None)
        if types is None:
            raise ValueError("No layer_types in loaded config; provide explicit --mha-layers indices.")
        result = [i for i, kind in enumerate(types) if kind == "full_attention"]
    else:
        result = sorted(set(int(i) for i in selection.split(",")))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"Invalid layers: {result}")
    return result


def extract_image_features(model, pixels):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    captured = {}
    norm = base.model.multi_modal_projector.mm_soft_emb_norm
    hook = norm.register_forward_pre_hook(lambda module, args: captured.update(x=args[0]))
    try:
        with torch.inference_mode():
            outputs = base.get_image_features(pixel_values=pixels)
            projected = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            preprojector = captured["x"].float().mean(dim=1)
        return projected, preprojector
    finally:
        hook.remove()


def processed_image_batches(paths, processor, sig_processor=None, workers=IMAGE_LOAD_NUM_WORKERS,
                            prefetch_batches=0, timings=None):
    """One CPU processor; at most prefetch_batches pending batches plus the consumer's batch."""
    count = (len(paths) + IMAGE_PROCESS_BATCH_SIZE - 1) // IMAGE_PROCESS_BATCH_SIZE
    with closing(image_batches(paths, batch_size=IMAGE_PROCESS_BATCH_SIZE, workers=workers, timings=timings)) as source:
        def prepare_batch():
            start, images = next(source)
            try:
                pixels = None
                if processor is not None:
                    tick = perf_counter()
                    pixels = processor.image_processor(
                        images=images, return_tensors="pt", do_pan_and_scan=False,
                    )["pixel_values"]
                    if timings is not None:
                        timings["medgemma_processor_sec"] = timings.get("medgemma_processor_sec", 0) + perf_counter() - tick
                sig_pixels = None
                if sig_processor is not None:
                    tick = perf_counter()
                    sig_pixels = sig_processor(images=images, return_tensors="pt")["pixel_values"]
                    if timings is not None:
                        timings["medsiglip_processor_sec"] = timings.get("medsiglip_processor_sec", 0) + perf_counter() - tick
                return start, pixels, sig_pixels
            finally:
                for image in images:
                    image.close()
                images.clear()

        if not prefetch_batches:
            for _ in range(count):
                yield prepare_batch()
        else:
            # Only this worker advances the image iterator or calls the CPU processors.
            with ThreadPoolExecutor(max_workers=1) as pool:
                submitted = min(prefetch_batches, count)
                pending = deque(pool.submit(prepare_batch) for _ in range(submitted))
                while pending:
                    batch = pending.popleft().result()
                    if submitted < count:
                        pending.append(pool.submit(prepare_batch))
                        submitted += 1
                    yield batch
                    del batch
        next(source, None)  # Finish the image iterator and its progress bar after the last batch.


def prepare_visual_features(paths, medgemma, processor, include_medsiglip=True,
                            workers=IMAGE_LOAD_NUM_WORKERS, prefetch_batches=0, timings=None):
    """Bounded raw images; retain projected tokens, never a full raw/pixel cache."""
    device = next(medgemma.parameters()).device if medgemma is not None else torch.device("cuda:0")
    n = len(paths)
    projected = preprojector = None
    if medgemma is not None:
        config = medgemma.config
        projected = torch.empty((n, config.mm_tokens_per_image, config.text_config.hidden_size), dtype=MODEL_DTYPE)
        preprojector = np.empty((n, config.vision_config.hidden_size), dtype=np.float32)
    sig_features = None
    sig_processor = None
    if include_medsiglip:
        sig_processor = AutoProcessor.from_pretrained(MEDSIGLIP_MODEL_ID)
        sig_model = AutoModel.from_pretrained(MEDSIGLIP_MODEL_ID, dtype=MODEL_DTYPE).to(device).eval()
        sig_features = np.empty((n, sig_model.config.vision_config.hidden_size), dtype=np.float32)
        sig_buffer = torch.empty((MEDSIGLIP_BATCH_SIZE, 3, 448, 448), dtype=MODEL_DTYPE)
        sig_count = sig_start = 0

    with closing(processed_image_batches(paths, processor, sig_processor, workers, prefetch_batches, timings)) as batches:
        tick = perf_counter()
        for start, pixels, sig_pixels in batches:
            if timings is not None:
                timings["input_wait_sec"] = timings.get("input_wait_sec", 0) + perf_counter() - tick
            batch_size = len(pixels) if pixels is not None else len(sig_pixels)
            end = start + batch_size
            if medgemma is not None:
                tick = perf_counter()
                features, pre = extract_image_features(medgemma, pixels.to(device, dtype=MODEL_DTYPE))
                projected[start:end].copy_(features)
                preprojector[start:end] = pre.cpu().numpy()
                if timings is not None:
                    timings["medgemma_gpu_pipeline_sec"] = timings.get("medgemma_gpu_pipeline_sec", 0) + perf_counter() - tick
                del features, pre
            del pixels
            if include_medsiglip:
                offset = 0
                while offset < batch_size:
                    count = min(MEDSIGLIP_BATCH_SIZE - sig_count, batch_size - offset)
                    tick = perf_counter()
                    sig_buffer[sig_count:sig_count + count].copy_(sig_pixels[offset:offset + count])
                    if timings is not None:
                        timings["medsiglip_cpu_staging_sec"] = timings.get("medsiglip_cpu_staging_sec", 0) + perf_counter() - tick
                    sig_count += count
                    offset += count
                    if sig_count == MEDSIGLIP_BATCH_SIZE or (end == n and offset == batch_size):
                        tick = perf_counter()
                        with torch.inference_mode():
                            out = sig_model.get_image_features(pixel_values=sig_buffer[:sig_count].to(device))
                            embedding = out.pooler_output if hasattr(out, "pooler_output") else out
                            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
                        sig_features[sig_start:sig_start + sig_count] = embedding.float().cpu().numpy()
                        if timings is not None:
                            timings["medsiglip_gpu_pipeline_sec"] = timings.get("medsiglip_gpu_pipeline_sec", 0) + perf_counter() - tick
                        sig_start += sig_count
                        sig_count = 0
            del sig_pixels
            tick = perf_counter()
    return projected, preprojector, sig_features


def forward_prompt(model, prompt, image_features):
    device = next(model.parameters()).device
    inputs = {key: value.repeat(len(image_features), 1).to(device) for key, value in prompt.items()}
    inputs["pixel_values"] = image_features.to(device, dtype=MODEL_DTYPE)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1, return_dict=True)
    return inputs, outputs


def extract_activations(model, prompt, projected, layers, yes_no_ids, disk_dir=None, collect_linear=True, timings=None,
                        collect_scores=None):
    # Preserve existing benchmark callers; production can request scores without pooled arrays.
    if collect_scores is None:
        collect_scores = collect_linear
    n, tokens, dim = projected.shape
    count = model.config.text_config.num_hidden_layers
    cached = {} if disk_dir is not None else {i: torch.empty((n, tokens, dim), dtype=MODEL_DTYPE) for i in layers}
    pooled = {name: np.empty((n, count, dim), dtype=np.float32) for name in POOLED_FEATURES} if collect_linear else {}
    logprob = np.empty((n, 2), dtype=np.float32) if collect_scores else None
    if disk_dir is not None:
        disk_dir.mkdir(parents=True, exist_ok=True)
        needed = n * tokens * dim * 4 * len(layers)
        if shutil.disk_usage(disk_dir).free < needed + 10 * 1024**3:
            raise RuntimeError(f"Insufficient activation disk space; payload requires {needed / 1024**3:.1f} GiB")
    with ExitStack() as stack:
        datasets = {}
        if disk_dir is not None:
            for layer in layers:
                file = stack.enter_context(h5py.File(disk_dir / f"layer_{layer:02d}.h5", "x"))
                datasets[layer] = file.create_dataset("image_tokens", (n, tokens, dim), dtype="float32")
        stack.enter_context(cached_image_forward(model))
        for start in tqdm(range(0, n, MEDGEMMA_BATCH_SIZE), desc="Decoder extraction"):
            end = min(start + MEDGEMMA_BATCH_SIZE, n)
            if timings is not None:
                if projected.is_cuda or next(model.parameters()).is_cuda:
                    torch.cuda.synchronize()
                tick = perf_counter()
            inputs, outputs = forward_prompt(model, prompt, projected[start:end])
            if timings is not None:
                if outputs.logits.is_cuda:
                    torch.cuda.synchronize()
                timings["forward_and_input_transfer_sec"] = timings.get("forward_and_input_transfer_sec", 0) + perf_counter() - tick
                tick = perf_counter()
            mask = inputs["input_ids"].eq(image_token_id(model.config))
            assert torch.all(mask.sum(dim=1) == tokens), "Image-token expansion changed"
            last_positions = inputs["attention_mask"].sum(dim=1) - 1
            if collect_scores:
                probs = outputs.logits[:, -1].float().log_softmax(-1)
                logprob[start:end] = probs[:, list(yes_no_ids)].cpu().numpy()
            for layer, hidden in enumerate(outputs.hidden_states[1:]):
                if not collect_linear and layer not in layers:
                    continue
                image_hidden = hidden[mask].reshape(end - start, tokens, dim)
                if collect_linear:
                    image_float = image_hidden.float()
                    pooled[POOLED_FEATURES[0]][start:end, layer] = image_float.mean(1).cpu().numpy()
                    pooled[POOLED_FEATURES[1]][start:end, layer] = image_float[:, -1].cpu().numpy()
                    pooled[POOLED_FEATURES[2]][start:end, layer] = hidden[
                        torch.arange(end - start, device=hidden.device), last_positions,
                    ].float().cpu().numpy()
                if layer in layers:
                    if disk_dir is None:
                        cached[layer][start:end].copy_(image_hidden)
                    else:
                        array = image_hidden.float().cpu().numpy()
                        write_start = perf_counter()
                        datasets[layer][start:end] = array
                        if timings is not None:
                            timings["hdf5_write_sec"] = timings.get("hdf5_write_sec", 0) + perf_counter() - write_start
            if timings is not None:
                timings["feature_copy_and_write_sec"] = timings.get("feature_copy_and_write_sec", 0) + perf_counter() - tick
    return cached, pooled, logprob


def load_layer(path):
    with h5py.File(path, "r") as file:
        dataset = file["image_tokens"]
        result = torch.empty(dataset.shape, dtype=MODEL_DTYPE)
        for start in range(0, len(dataset), 16):
            batch = dataset[start:start + 16]
            if not np.isfinite(batch).all():
                raise ValueError(f"Non-finite activation in {path}")
            result[start:start + len(batch)].copy_(torch.from_numpy(batch))
    return result


def metrics(y, scores):
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite probe scores")
    return {"positive_prevalence": float(np.mean(y)), "auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores))}


def fit_linear(x, y, train_idx, test_idx):
    if not np.isfinite(x).all():
        raise ValueError("Non-finite linear feature matrix")
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=MAX_ITER, random_state=RANDOM_STATE,
    ))
    pipeline.fit(x[train_idx].astype(np.float32, copy=False), y[train_idx])
    result = metrics(y[test_idx], pipeline.predict_proba(x[test_idx])[:, 1])
    result["n_iter"] = int(pipeline.named_steps["logisticregression"].n_iter_[0])
    return pipeline, result


def train_linear_features(features, y, train_idx, test_idx, backend="loky", only=None, on_complete=None):
    jobs = [(name, layer, x if x.ndim == 2 else x[:, layer])
            for name, x in features.items() for layer in ([None] if x.ndim == 2 else range(x.shape[1]))
            if only is None or (name, layer) in only]
    if not jobs:
        return []
    fitted = []
    options = {"inner_max_num_threads": LINEAR_INNER_NUM_THREADS} if backend == "loky" else {}
    # Thread limits are process-wide: one outer context, never per threaded fit.
    with threadpool_limits(limits=LINEAR_INNER_NUM_THREADS), parallel_config(
        backend=backend, n_jobs=LINEAR_PARALLEL_JOBS, **options,
    ):
        with Parallel(max_nbytes=None, return_as="generator") as pool:
            with closing(pool(delayed(fit_linear)(x, y, train_idx, test_idx) for _, _, x in jobs)) as outputs:
                for (name, layer, _), (pipeline, result) in zip(jobs, outputs):
                    if on_complete is not None:
                        on_complete(name, layer, pipeline, result)
                    fitted.append((name, layer, pipeline, result))
    print("Linear fits hitting max_iter", sum(result["n_iter"] >= MAX_ITER for _, _, _, result in fitted))
    return fitted


class MHAPoolingProbe(torch.nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=MHA_MLP_DIM, num_attention_heads=MHA_NUM_HEADS, need_weights=True):
        super().__init__()
        self.pooler = SiglipMultiheadAttentionPoolingHead(SiglipVisionConfig(
            hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads,
            layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh",
        ))
        self.classifier = torch.nn.Linear(hidden_size, 1)
        self.need_weights = need_weights

    def forward(self, tokens):
        if self.need_weights:
            pooled = self.pooler(tokens)
        else:
            query = self.pooler.probe.repeat(tokens.shape[0], 1, 1)
            hidden = self.pooler.attention(query, tokens, tokens, need_weights=False)[0]
            pooled = (hidden + self.pooler.mlp(self.pooler.layernorm(hidden)))[:, 0]
        return self.classifier(pooled).squeeze(-1)


def train_mha(tokens, y, train_idx, test_idx, epochs=MHA_EPOCHS, cache_on_gpu=True,
              need_weights=True, device="cuda:0", probe=None, batch_size=MHA_BATCH_SIZE):
    device = torch.device(device)
    if probe is None:
        torch.manual_seed(RANDOM_STATE)
        probe = MHAPoolingProbe(tokens.shape[-1], need_weights=need_weights)
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=MHA_LEARNING_RATE, weight_decay=MHA_WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    start = perf_counter()
    cache = tokens.to(device) if cache_on_gpu else tokens
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    transfer_sec = perf_counter() - start
    train_indices = torch.as_tensor(train_idx)
    start = perf_counter()
    for epoch in range(epochs):
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        loss_sum = torch.zeros((), device=device)
        probe.train()
        for offset in range(0, len(permutation), batch_size):
            index = permutation[offset:offset + batch_size]
            device_index = index.to(device)
            batch = cache[device_index if cache.is_cuda else index].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=MODEL_DTYPE, enabled=device.type == "cuda"):
                logits = probe(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets[device_index])
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(index)
        final_loss = float(loss_sum / len(train_idx))
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    train_sec = perf_counter() - start
    start = perf_counter()
    probe.eval()
    scores = []
    with torch.inference_mode():
        for offset in range(0, len(test_idx), batch_size):
            index = torch.as_tensor(test_idx[offset:offset + batch_size], device=cache.device)
            with torch.autocast(device_type=device.type, dtype=MODEL_DTYPE, enabled=device.type == "cuda"):
                output = probe(cache[index].to(device))
            scores.append(output.float().cpu())
    scores = torch.cat(scores).numpy()
    result = {**metrics(y[test_idx], scores), "n_iter": epochs}
    timing = {"transfer_sec": transfer_sec, "train_sec": train_sec,
              "eval_sec": perf_counter() - start, "final_train_loss": final_loss}
    return probe.cpu(), result, timing


def probe_path(root, model_name, order, label, feature, layer):
    folder = root / "probes" / slug(model_name) / order / slug(label) / ("layer_none" if layer is None else f"layer_{layer:02d}")
    return folder / (feature + (".pt" if feature == MHA_FEATURE else ".pkl"))


@contextmanager
def atomic_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_probe(root, model_name, order, label, feature, layer, probe, result):
    path = probe_path(root, model_name, order, label, feature, layer)
    with atomic_output(path) as temporary:
        if isinstance(probe, torch.nn.Module):
            torch.save({"state_dict": probe.state_dict(), "hidden_size": probe.classifier.in_features,
                        "intermediate_size": MHA_MLP_DIM, "num_attention_heads": MHA_NUM_HEADS}, temporary)
        else:
            with temporary.open("wb") as file:
                pickle.dump(probe, file)
    return {"model_name": model_name, "prompt_order": order, "condition": label, "feature": feature,
            "layer": layer, "label": label, "feature_dim": (probe.classifier.in_features if isinstance(probe, torch.nn.Module)
                                                             else probe.n_features_in_),
            **result, "model_path": str(path)}


def update_table(rows, path, replace_by=RESULT_KEY):
    """Replace matching keys, preserving other results; score blocks use combination keys."""
    frame = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, float_precision="round_trip")
        replaced = pd.MultiIndex.from_frame(old[replace_by]).isin(pd.MultiIndex.from_frame(frame[replace_by]))
        frame = pd.concat([old.loc[~replaced], frame], ignore_index=True)
    with atomic_output(path) as temporary:
        frame.to_csv(temporary, index=False)


def completed_results(root, df=None):
    """Lightweight checks of final files only; temporary files never count as progress."""
    path = root / "results/experiment_metrics.csv"
    if not path.exists():
        return set()
    table = pd.read_csv(path, float_precision="round_trip")
    valid = np.isfinite(table[["auroc", "auprc", "positive_prevalence"]].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    valid &= table[RESULT_KEY[:-1]].notna().all(axis=1)
    complete_scores = set()
    score_path = root / "results/experiment_yes_no_scores.csv"
    if df is not None and score_path.exists():
        scores = pd.read_csv(score_path, float_precision="round_trip")
        for identity, block in scores.groupby(["model_name", "prompt_order", "label"]):
            if (len(block) != len(df) or not block.study_id.is_unique
                    or set(block.study_id) != set(df.study_id)):
                continue
            numeric = block[["logprob_yes", "logprob_no", "score_yes_minus_no"]].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric).all().all() or block[["y_true", "pred_yes", "probe_split"]].isna().any().any():
                continue
            aligned = block.set_index("study_id").loc[df.study_id]
            if identity[2] not in df or not np.array_equal(aligned.y_true, df[identity[2]].eq(1)):
                continue
            if np.array_equal(aligned.probe_split, df.probe_split):
                complete_scores.add(identity)
    completed = set()
    for row in table.loc[valid].itertuples(index=False):
        layer = None if pd.isna(row.layer) else float(row.layer)
        if layer is not None:
            if not np.isfinite(layer) or not layer.is_integer():
                continue
            layer = int(layer)
        key = (row.model_name, row.prompt_order, row.condition, row.label, row.feature, layer)
        if row.feature == YES_NO_FEATURE:
            if (row.model_name, row.prompt_order, row.label) in complete_scores:
                completed.add(key)
        else:
            saved = probe_path(root, row.model_name, row.prompt_order, row.label, row.feature, layer)
            if (saved.resolve().is_relative_to((root / "probes").resolve())
                    and saved.is_file() and saved.stat().st_size > 0):
                completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=resolve_path, required=True, help="Fresh runs/<run_name> folder, or existing folder with --resume")
    parser.add_argument("--resume", action="store_true", help="Preserve completed probes and continue an existing run with unchanged inputs/settings")
    parser.add_argument("--adapter-root", type=resolve_path, default=IMPORTED_RUN_ROOT / "lora_sft")
    parser.add_argument("--model", choices=["all"] + MODEL_NAMES, default="all")
    parser.add_argument("--prompt-order", choices=["all"] + PROMPT_ORDERS, default="all")
    parser.add_argument("--finding", choices=["all"] + TARGET_LABELS, default="all")
    parser.add_argument("--mha-layers", default="global", help="global, all, or comma-separated zero-indexed layers")
    parser.add_argument("--activation-cache", choices=["ram", "disk"], default="ram")
    args = parser.parse_args()
    with run_log(args.output_root, resume=args.resume):
        run(args)


def run(args):
    print("Manifest:", DATA_CSV)
    print("Adapter root:", args.adapter_root)
    models = MODEL_NAMES if args.model == "all" else [args.model]
    linear_root = args.output_root / "probing/linear_probe"
    mha_root = args.output_root / "probing/multi_head_attention_probe"
    df = pd.read_csv(DATA_CSV)
    train_idx = np.flatnonzero(df.probe_split.eq("train"))
    test_idx = np.flatnonzero(df.probe_split.eq("test"))
    labels = df[TARGET_LABELS].eq(1).to_numpy(dtype=np.int8)
    orders = PROMPT_ORDERS if args.prompt_order == "all" else [args.prompt_order]
    findings = TARGET_LABELS if args.finding == "all" else [args.finding]
    config = AutoConfig.from_pretrained(MEDGEMMA_MODEL_ID)
    selected = select_layers(config, args.mha_layers)
    linear_keys = [(name, layer) for name in POOLED_FEATURES for layer in range(config.text_config.num_hidden_layers)]
    linear_keys += [(name, None) for name in CONTROL_FEATURES]
    done_linear = completed_results(linear_root, df) if args.resume else set()
    done_mha = completed_results(mha_root) if args.resume else set()
    baseline_missing = [label for label in TARGET_LABELS if
                        ("medsiglip_standalone", "baseline", label, label, "medsiglip_standalone", None) not in done_linear]
    work, summary = [], []
    for model_name in models:
        for order in orders:
            for label in findings:
                prefix = (model_name, order, label, label)
                missing_linear = [key for key in linear_keys if prefix + key not in done_linear]
                missing_mha = [layer for layer in selected if prefix + (MHA_FEATURE, layer) not in done_mha]
                need_scores = prefix + (YES_NO_FEATURE, None) not in done_linear
                summary.append({"model": model_name, "order": order, "finding": label,
                                "linear_complete": len(linear_keys) - len(missing_linear), "linear_pending": len(missing_linear),
                                "mha_complete": len(selected) - len(missing_mha), "mha_pending": len(missing_mha),
                                "scores_pending": need_scores})
                if missing_linear or missing_mha or need_scores:
                    work.append((model_name, order, label, missing_linear, missing_mha, need_scores))
    print("MedSigLIP complete", len(TARGET_LABELS) - len(baseline_missing), "pending", len(baseline_missing))
    print(pd.DataFrame(summary).to_string(index=False))
    if not work and not baseline_missing:
        print("All requested results are complete; nothing to run.")
        return
    for name in {item[0] for item in work}:
        if name != "base_medgemma":
            adapter = args.adapter_root / (name.removeprefix("lora_") + "_adapter")
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Adapter config missing: {adapter}")
    configure_runtime()
    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID) if work else None
    model = load_medgemma() if work else None
    with Measure("shared_visual_features", ACTIVATION_ROOT):
        projected, preprojector, standalone = prepare_visual_features(
            df.image_path.tolist(), model, processor, include_medsiglip=bool(baseline_missing),
        )
    # Adapter verification below uses the same batch shape as image preparation.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if baseline_missing:
        with Measure("medsiglip_linear", ACTIVATION_ROOT):
            for label in baseline_missing:
                i = TARGET_LABELS.index(label)
                with threadpool_limits(limits=LINEAR_INNER_NUM_THREADS):
                    probe, result = fit_linear(standalone, labels[:, i], train_idx, test_idx)
                row = save_probe(linear_root, "medsiglip_standalone", "baseline", label, "medsiglip_standalone", None, probe, result)
                update_table([row], linear_root / "results/experiment_metrics.csv")
                del probe
    del standalone
    if not work:
        print("After run cleanup", resource_snapshot(ACTIVATION_ROOT))
        return
    prompts, answer_ids = tokenize_prompts(processor), answer_token_ids(processor)
    predecoder = None
    if any((CONTROL_FEATURES[1], None) in item[3] for item in work):
        predecoder = np.empty((len(df), projected.shape[-1]), dtype=np.float32)
        for start in range(0, len(df), MEDGEMMA_BATCH_SIZE):
            predecoder[start:start + MEDGEMMA_BATCH_SIZE] = projected[start:start + MEDGEMMA_BATCH_SIZE].float().mean(1).numpy()
    for model_name in models:
        model_work = [item for item in work if item[0] == model_name]
        if not model_work:
            continue
        adapter = None if model_name == "base_medgemma" else args.adapter_root / (model_name.removeprefix("lora_") + "_adapter")
        model = load_medgemma(adapter)
        for _, images in image_batches(df.image_path.iloc[:MEDGEMMA_BATCH_SIZE]):
            pixels = processor.image_processor(images=images, return_tensors="pt", do_pan_and_scan=False)["pixel_values"]
            actual, actual_pre = extract_image_features(model, pixels.to("cuda:0", dtype=MODEL_DTYPE))
            torch.testing.assert_close(actual.cpu(), projected[:len(images)], rtol=0, atol=0)
            torch.testing.assert_close(actual_pre.cpu(), torch.from_numpy(preprojector[:len(images)]), rtol=0, atol=0)
            del actual, actual_pre, pixels
        print(model_name, "MHA layers", selected, flush=True)
        for _, order, label, missing_linear, missing_mha, need_scores in model_work:
            y = labels[:, TARGET_LABELS.index(label)]
            identity = f"{model_name}/{order}/{slug(label)}"

            def save_linear_result(name, layer, probe, result):
                row = save_probe(linear_root, model_name, order, label, name, layer, probe, result)
                update_table([row], linear_root / "results/experiment_metrics.csv")

            controls = {name: value for name, value in zip(CONTROL_FEATURES, (preprojector, predecoder))
                        if (name, None) in missing_linear}
            need_pooled = any(name in POOLED_FEATURES for name, _ in missing_linear)
            if controls and not need_pooled:
                with Measure(identity + "/linear_controls", ACTIVATION_ROOT):
                    fitted = train_linear_features(controls, y, train_idx, test_idx, on_complete=save_linear_result)
                del fitted
            with ExitStack() as unit:
                disk_dir = None
                if args.activation_cache == "disk" and missing_mha:
                    ACTIVATION_ROOT.mkdir(parents=True, exist_ok=True)
                    disk_dir = Path(unit.enter_context(TemporaryDirectory(prefix=args.output_root.name + "_", dir=ACTIVATION_ROOT)))
                blocks = [missing_mha] if disk_dir else [missing_mha[i:i + LAYER_BLOCK_SIZE] for i in range(0, len(missing_mha), LAYER_BLOCK_SIZE)]
                if not blocks and (need_pooled or need_scores):
                    blocks = [[]]
                for block_i, block in enumerate(blocks):
                    with Measure(identity + f"/extract_{block_i}", ACTIVATION_ROOT):
                        cached, pooled, logprob = extract_activations(
                            model, prompts[(order, label)], projected, block, answer_ids, disk_dir,
                            collect_linear=block_i == 0 and need_pooled,
                            collect_scores=block_i == 0 and need_scores,
                        )
                    if block_i == 0 and need_scores:
                        score = logprob[:, 0] - logprob[:, 1]
                        scores = df[["study_id", "subject_id", "dicom_id", "probe_split"]].assign(
                            model_name=model_name, prompt_order=order, label=label, y_true=y,
                            logprob_yes=logprob[:, 0], logprob_no=logprob[:, 1], score_yes_minus_no=score, pred_yes=score > 0,
                        )
                        update_table(scores, linear_root / "results/experiment_yes_no_scores.csv",
                                     replace_by=["model_name", "prompt_order", "label"])
                        row = {"model_name": model_name, "prompt_order": order, "condition": label,
                               "feature": YES_NO_FEATURE, "layer": None, "label": label,
                               "feature_dim": 1, **metrics(y[test_idx], score[test_idx]), "n_iter": None, "model_path": ""}
                        update_table([row], linear_root / "results/experiment_metrics.csv")
                        del scores, score
                    if block_i == 0 and need_pooled:
                        pooled.update(controls)
                        with Measure(identity + "/linear", ACTIVATION_ROOT):
                            fitted = train_linear_features(pooled, y, train_idx, test_idx, only=missing_linear,
                                                           on_complete=save_linear_result)
                        del fitted
                    del pooled, logprob
                    for layer in block:
                        with Measure(identity + f"/mha_L{layer}", ACTIVATION_ROOT):
                            tokens = load_layer(disk_dir / f"layer_{layer:02d}.h5") if disk_dir else cached.pop(layer)
                            probe, result, timing = train_mha(tokens, y, train_idx, test_idx)
                            row = save_probe(mha_root, model_name, order, label, MHA_FEATURE, layer, probe, result)
                            update_table([row], mha_root / "results/experiment_metrics.csv")
                            print({"layer": layer, **result, **timing})
                            del tokens, probe
                        if disk_dir:
                            (disk_dir / f"layer_{layer:02d}.h5").unlink()
                        gc.collect()
                        torch.cuda.empty_cache()
                    del cached
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("After block cleanup", resource_snapshot(ACTIVATION_ROOT))
            del controls
            print("After combination cleanup", resource_snapshot(ACTIVATION_ROOT))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    del projected, preprojector, predecoder
    gc.collect()
    torch.cuda.empty_cache()
    print("After run cleanup", resource_snapshot(ACTIVATION_ROOT))

if __name__ == "__main__":
    main()
