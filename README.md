# [Paper PDF](paper/paper.pdf)

# Probing Chest X-ray Representations in MedGemma

This project studies where chest X-ray finding information is accessible inside MedGemma 4B. The main experiment compares logistic-regression probes trained on standalone MedSigLIP embeddings, pre-decoder MedGemma image tokens, decoder-layer residual states, final prompt-token states, and MedGemma yes/no next-token scores. The current development branch also adds a benchmark for a SigLIP-style multi-head attention pooling probe over MedGemma image-token sequences.

The current report focuses on CheXpert+ with five findings: atelectasis, cardiomegaly, consolidation, edema, and pleural effusion. The main result is that the standalone MedSigLIP embedding performs best by mean AUROC, while MedGemma decoder representations and zero-shot generative classification do not improve linear separability in this setup.

## Executable Workflow

The paper source is `paper/paper.tex`; its required preamble and bibliography
remain in `paper/preamble/` and `paper/bibs/references.bib`. Figures and tables
come from `runs/2026-09-05_imported_artifacts/`, so that ignored snapshot is
needed to rebuild the paper. The existing PDF is retained unchanged.

The current `dev` workstream uses scripts for execution and one notebook
for visualization. Historical execution and benchmark notebooks, including
saved outputs, are preserved unchanged in `temp/archive/notebooks/`. The current
script-based benchmark lives under `temp/benchmark/` and is copied to the server
manually, not tracked in Git.

```text
process_data.py                  prepare the existing study-level split
lora_sft.py                      train the two adapters only when requested
probing.py                       shared linear/MHA probing and yes/no evaluation
experiment_utils.py             bounded images and exact prompt handling
visualization.ipynb
resource_monitor.py             shared production/benchmark resource monitoring
temp/benchmark/benchmark_probing.py staged comparisons, console.log only
paper/                          existing paper sources and PDF
temp/archive/notebooks/         historical execution and benchmark notebooks
temp/tests/                     optional local development checks
```

`temp/` is ignored by Git, so benchmarks, archived notebooks, and optional checks
stay local unless copied separately. Do not delete it indiscriminately: `temp/artifacts-*`
may contain archived results and trained adapters.

Scripts locate the project from `__file__`, so data/output paths are independent
of the working directory. The server repository is now
`/home/tdnguyen/workspace/cxr-vlm-interp`. Dataset paths remain under
`/opt/gpudata/cxr`. The home-directory Hugging Face cache is unchanged.

## Existing Snapshot

Existing data, adapters, and results are inputs under
`runs/2026-09-05_imported_artifacts/`. This is the import date, not a training
date. Preserve its internal layout, including legacy `results/` and `probes/`
folders when present. The local move is complete, and the matching server move
was confirmed by the user on 2026-09-05.

The default manifest is `processed_data/chexpertplus_frontal_5labels.csv` inside
that snapshot; adapters default to its `lora_sft/`. The local snapshot contains
legacy baseline results but no adapter weights; the user reported that the
server snapshot contains the adapters. Moving a folder does not require
regenerating the split or retraining models.

`/runs/` is gitignored. New benchmark and production commands require a fresh
`--output-root runs/<name>` and keep a `console.log` there. Do not pre-create
that run folder. Existing run folders are never reused, even after a failure.

## Environment

Install dependencies with either:

```bash
conda env create -f environment.yml
```

or:

```bash
pip install -r requirements.txt
```

The refactor adds `psutil`. Keep the working server environment; do not upgrade
Transformers merely to run these scripts. The correctness benchmark prints the
installed versions because cached image-feature handling depends on HF's model
implementation. Local tests are not a substitute for this server check.

## Run Benchmarks First

Use your assigned GPU, not an arbitrary device number. In your server shell:

```bash
cd "$HOME/workspace/cxr-vlm-interp"
conda activate cxr-vlm-interp
export CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU"
python temp/benchmark/benchmark_probing.py --stage correctness --output-root runs/benchmark_correctness_01
python temp/benchmark/benchmark_probing.py --stage mha --rows 512 --epochs 1 --output-root runs/benchmark_mha_small_01
```

`ALLOCATED_GPU` stands for the GPU ID or UUID assigned to you. When your scheduler
already sets `CUDA_VISIBLE_DEVICES`, retain it and omit the export. The scripts
require exactly one visible GPU and cap CPU affinity at 32 available logical
CPUs. They do not launch multi-GPU jobs or occupy all 288 CPUs.

Copy `temp/benchmark/` to the same location in the server checkout first; Git
does not transfer it. The runner also works from inside that folder:

```bash
cd "$HOME/workspace/cxr-vlm-interp/temp/benchmark"
python benchmark_probing.py --stage correctness --output-root runs/benchmark_correctness_02
```

It resolves production imports and data paths from its location, not the shell's
working directory. The shared `resource_monitor.py` remains with the tracked
production scripts, so normal probing does not depend on `temp/`.

After these pass, run the full-data MHA comparison and return stdout:

```bash
python "$HOME/workspace/cxr-vlm-interp/temp/benchmark/benchmark_probing.py" --stage mha --epochs 3 --output-root runs/benchmark_mha_3epochs_01
python "$HOME/workspace/cxr-vlm-interp/temp/benchmark/benchmark_probing.py" --stage mha --epochs 30 --output-root runs/benchmark_mha_30epochs_01
```

The correctness stage compares the original and cached-feature forwards across
all three models, both prompt orders, and five findings on eight images. It
checks hidden states/logits and unchanged vision features; it does not train
probes or measure classification accuracy.

Other comparisons are optional, not a required sweep. `--stage preprocess`
compares one versus four loading workers on 2,048 images. `--stage linear`
compares processes versus threads. Prioritize the cache stage before committing
to all-layer MHA: start with `--stage cache --rows 512 --epochs 1`.
Each full-data stage repeats image preparation and activation extraction, so
running all stages separately has substantial setup cost.

Run other stages individually from `temp/benchmark/`, using a fresh name each time:

```bash
python benchmark_probing.py --stage preprocess --output-root runs/benchmark_preprocess_01
python benchmark_probing.py --stage linear --output-root runs/benchmark_linear_01
python benchmark_probing.py --stage cache --rows 512 --epochs 1 --output-root runs/benchmark_cache_small_01
python benchmark_probing.py --stage cache --epochs 3 --output-root runs/benchmark_cache_full_01
python benchmark_probing.py --stage streams --epochs 3 --output-root runs/benchmark_streams_01
```

The MHA stage compares CPU batch transfers with GPU-resident activations on L17,
using matched initialization and sample order. `--no-attention-weights` is a
separate optional speed comparison after the attention equivalence check passes.
The cache stage compares two five-layer RAM blocks against ten FP32 HDF5 layer
files. A full-data cache run (`--stage cache --epochs 3`) writes about **610 GiB**;
do not start it until the small run passes and bulk storage is available.
`--stage streams --epochs 3` optionally compares two independent probes on the
same representative layer, sequentially and on two CUDA streams. It is not the
production default and is not guaranteed to be faster.

Historical baseline/linear metrics are read from the imported snapshot's
`probing/linear_probe/results/experiment_metrics.csv`, falling back to its
legacy `results/experiment_metrics.csv`. For another result set, pass
`--linear-metrics <path>`. Benchmarks print their tables and keep `console.log`;
they do not save metrics CSVs or probe models.

Resource tables report wall time, CPU core-equivalent use, sampled GPU
utilization, CUDA allocated/reserved memory, process-tree RSS/USS/PSS, available
system RAM, and disk free space. Summed RSS can double-count shared mappings;
process CPU sampling can miss workers that exit between samples. HDF5 timings
include filesystem caching, not guaranteed durable-media throughput. Filesystem
cache is not cleared, and occupied system RAM alone is not a process leak.

Optional synthetic checks are kept in `temp/tests/`, outside the normal run
workflow. When that local folder is available, run them from the repository
with `python -m unittest discover -s temp/tests -v`.

## Run Probing

Run these commands from the repository root. The existing manifest and both
adapters are reused from the imported snapshot; no SFT rerun is needed.
New outputs require a fresh run folder:

```bash
python probing.py --output-root runs/probing_global_01 --mha-layers global
```

This runs three models, both prompt orders, and five findings. It refreshes all
34-layer linear probes and trains MHA on the full-attention layers specified by
the loaded model configuration (expected `5,11,17,23,29`). Optional selectors are
`--model`, `--prompt-order`, and `--finding`; use `--help` for accepted names.
Adapters default to `runs/2026-09-05_imported_artifacts/lora_sft`; use `--adapter-root` to select another
pair. Their learning rate is not inferred from current training constants.

`--mha-layers all` selects all 34 layers; comma-separated indices select a subset.
The default `--activation-cache ram` retains at most five bf16 layers, repeating
decoder-only extraction for subsequent blocks. `--activation-cache disk` writes
each selected layer once as FP32 and loads/deletes it in turn. Do not select an
all-layer full run before reviewing the new benchmark's total-cost estimate.

- One bf16 image-token layer: **30.5 GiB**, transferred to the GPU once per probe.
- Five bf16 layers: **152.6 GiB**; total job target roughly **200-250 GiB**.
- One FP32 HDF5 layer: **61 GiB**; 34 layers: **2.03 TiB** per combination.
- Five global layers across all combinations: **150 MHA probes**; all layers: **1,020**.
- MHA checkpoints retain FP32 parameters: approximately **300 MiB per probe**, not 157 MB.

Raw images are held only in bounded batches. MedSigLIP and MedGemma use their own
processors; MedGemma's frozen projected tokens are shared after verifying the
adapters do not alter the vision/projector. A small verification batch is
reprocessed for each model. Prompt formatting, multimodal masks, layer indexing,
and yes/no token definitions remain those of the earlier experiment.

Linear probes use StandardScaler plus LBFGS (`max_iter=2000`, default tolerance),
four processes with four inner threads. MHA uses the SigLIP pooling-head class,
20 heads, `2560 -> 10240 -> 2560` MLP, batch 512, 30 epochs, AdamW at `5e-5`, and
weight decay `1e-4`. Parameters stay FP32 with bf16 autocast; test AUROC is evaluated
after training, not used to select an epoch. Earlier test-set tuning was
exploratory; new hyperparameter selection requires a training-derived validation
set before making an unbiased paper claim.

## Outputs and Cleanup

```text
runs/
  2026-09-05_imported_artifacts/   existing contents, unchanged
  <benchmark_run>/
    console.log
  <probing_run>/
    console.log
    probing/
      linear_probe/{results,probes}/
      multi_head_attention_probe/{results,probes}/
      visualization/{plots,tables}/
  <sft_run>/
    console.log
    lora_sft/
  <data_run>/
    console.log
    processed_data/chexpertplus_frontal_5labels.csv
```

The CSV names and representation names remain compatible with previous results.
Each fitted linear pipeline is a separate pickle; each MHA probe is a separate
`.pt` state dictionary with dimensions. Full metric tables and yes/no scores are
written after completed combinations; MHA metrics/checkpoints are saved per
layer. Timings are printed and captured in `console.log`, not included in metric
CSVs. The shared log helper mirrors Python stdout/stderr and records uncaught
tracebacks; native-library and child-process output can bypass that redirection.
There is no automatic
resume mechanism; an interruption preserves already saved outputs, and a rerun
uses a new output root.

Activation files live only in a run-owned temporary directory under
`/opt/gpudata/trung/temp/mha_activation_cache/`. They are removed after use and on
handled exceptions/interrupts. A forced kill or machine failure cannot execute
cleanup; inspect only that run's directory afterward. Never delete
`temp/artifacts-*` in the repository: those contain archived results/adapters.

In `visualization.ipynb`, `ARTIFACT_ROOT` defaults to
`Path("runs/2026-09-05_imported_artifacts")`; change it to `Path("runs/<run_name>")`
for a new run. Run its cells to
regenerate the same six figures and report tables, with MHA added when present.
The snapshot's legacy `results/` linear layout is also accepted. No paper files are
modified.

## Data Preparation and Optional SFT

Do not rerun these to start probing. To intentionally create a new manifest or
train a new pair of adapters:

```bash
python process_data.py --output-root runs/data_new_01
python lora_sft.py --output-root runs/sft_new_01
```

Data preparation now takes `--output-root` instead of `--output`, saving the
manifest in that run's `processed_data/`. It does not switch the default input
manifest away from the imported snapshot. SFT uses the selected
settings: rank 16, alpha 32, dropout 0.05, decoder projection targets,
batch 32, accumulation 2, one epoch, learning rate `1e-4`, and cosine schedule.
It caches processed CPU pixels once for both orders and supervises only the
single yes/no answer token using full-vocabulary cross-entropy. Adapter files
and training logs are saved in the new run's `lora_sft/` folder. Existing adapters
are not overwritten or retrained as part of this migration.

The next research phase, after MHA, is report-generation transfer using the base
model and existing adapters. It is not implemented by this refactor.
