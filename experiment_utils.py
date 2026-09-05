"""Image preparation and prompt handling shared by the executable experiments."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import os
import sys
import traceback
from types import MethodType

import numpy as np
from PIL import Image
import psutil
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
IMPORTED_RUN_ROOT = PROJECT_ROOT / "runs/2026-09-05_imported_artifacts"
DATA_CSV = IMPORTED_RUN_ROOT / "processed_data/chexpertplus_frontal_5labels.csv"
ACTIVATION_ROOT = Path("/opt/gpudata/trung/temp/mha_activation_cache")
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDSIGLIP_MODEL_ID = "google/medsiglip-448"
MODEL_DTYPE = torch.bfloat16
RANDOM_STATE = 42
TARGET_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
PROMPT_ORDERS = ["image_first", "text_first"]
IMAGE_LOAD_NUM_WORKERS = 4
IMAGE_PROCESS_BATCH_SIZE = 24


def configure_runtime(require_gpu=True):
    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        process.cpu_affinity(process.cpu_affinity()[:32])
        print("CPU affinity", process.cpu_affinity())
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    torch.set_num_threads(4)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if require_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one allocated GPU with CUDA_VISIBLE_DEVICES before running.")
        torch.cuda.manual_seed_all(RANDOM_STATE)
        print("GPU", torch.cuda.get_device_properties(0))
    print("torch", torch.__version__)


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def run_log(output_root):
    """Keep a fresh run's Python console output without overwriting earlier runs."""
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "console.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print("Run folder:", output_root)
            print("Command:", sys.argv)
            try:
                yield
            except BaseException:
                # The interpreter prints the traceback to the terminal after restoration.
                traceback.print_exc(file=log)
                raise


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def image_batches(paths, batch_size=IMAGE_PROCESS_BATCH_SIZE, workers=IMAGE_LOAD_NUM_WORKERS):
    # Submit only one batch, not all paths: completed futures can also retain images.
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in tqdm(range(0, len(paths), batch_size), desc="Prepare images"):
            images = list(pool.map(load_rgb, paths[start:start + batch_size]))
            try:
                yield start, images
            finally:
                for image in images:
                    image.close()
                images.clear()


def prepare_pixels(paths, processor, workers=IMAGE_LOAD_NUM_WORKERS):
    pixels = None
    for start, images in image_batches(paths, workers=workers):
        batch = processor.image_processor(
            images=images, return_tensors="pt", do_pan_and_scan=False,
        )["pixel_values"]
        if pixels is None:
            pixels = torch.empty((len(paths), *batch.shape[1:]), dtype=MODEL_DTYPE)
        pixels[start:start + len(images)].copy_(batch)
        del batch
    print("cached_pixel_values", tuple(pixels.shape), pixels.dtype)
    return pixels


def prompt_text(processor, prompt_order, label):
    question = f"Question: Is there {label.lower()} in this image? Answer yes or no."
    image_item = {"type": "image"}
    if prompt_order == "image_first":
        content = [image_item, {"type": "text", "text": f"\n{question}\nAnswer: "}]
    else:
        content = [{"type": "text", "text": f"{question}\n"}, image_item, {"type": "text", "text": "\nAnswer: "}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=False, tokenize=False,
    )
    if isinstance(text, list):
        text = text[0]
    return text.replace(processor.boi_token, processor.full_image_sequence)


def add_token_types(processor, batch):
    if hasattr(processor, "create_mm_token_type_ids"):
        batch["token_type_ids"] = torch.as_tensor(
            processor.create_mm_token_type_ids(batch["input_ids"]), dtype=torch.long,
        )
    return batch


def tokenize_prompts(processor):
    return {
        (order, label): add_token_types(processor, processor.tokenizer(
            [prompt_text(processor, order, label)], padding=True, return_tensors="pt",
        ))
        for order in PROMPT_ORDERS for label in TARGET_LABELS
    }


def answer_token_ids(processor):
    ids = [processor.tokenizer(word, add_special_tokens=False).input_ids for word in ("yes", "no")]
    if any(len(row) != 1 for row in ids):
        raise ValueError("The existing experiment requires single-token yes/no answers.")
    return ids[0][0], ids[1][0]


def image_token_id(config):
    return config.image_token_id if hasattr(config, "image_token_id") else config.image_token_index


@contextmanager
def cached_image_forward(model):
    """Reuse HF's entire multimodal forward; replace only its image-feature call.

    Inside this context, pass projected image tokens in pixel_values. HF still
    performs placeholder replacement, embedding scaling, and multimodal masks.
    The correctness benchmark must pass on the server's Transformers version.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    multimodal = base.model
    previous = multimodal.__dict__.get("get_image_features")

    def get_cached_features(self, pixel_values, **kwargs):
        return BaseModelOutputWithPooling(pooler_output=pixel_values)

    multimodal.get_image_features = MethodType(get_cached_features, multimodal)
    try:
        yield
    finally:
        if previous is None:
            del multimodal.get_image_features
        else:
            multimodal.get_image_features = previous
