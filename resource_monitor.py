"""Small stdout-only resource sampler used by scripts and benchmarks."""

from pathlib import Path
from time import perf_counter
import shutil
import subprocess
import threading

import numpy as np
import psutil
import torch


def resource_snapshot(disk_path):
    processes = [psutil.Process()] + psutil.Process().children(recursive=True)
    rss = uss = pss = 0
    for process in processes:
        try:
            info = process.memory_full_info()
            rss += info.rss
            uss += getattr(info, "uss", 0)
            pss += getattr(info, "pss", 0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    path = Path(disk_path)
    while not path.exists():
        path = path.parent
    return {
        "tree_rss_sum_gib": rss / 1024**3,  # Can double-count shared mappings.
        "tree_uss_gib": uss / 1024**3,
        "tree_pss_gib": pss / 1024**3 if pss else np.nan,
        "system_available_gib": psutil.virtual_memory().available / 1024**3,
        "disk_free_gib": shutil.disk_usage(path).free / 1024**3,
    }


class Measure:
    def __init__(self, stage, disk_path=Path.cwd(), interval=3):
        self.stage, self.disk_path, self.interval = stage, disk_path, interval
        self.samples, self.cpu_seconds = [], {}
        self.stop = threading.Event()
        self.row = {}

    def sample(self):
        row = resource_snapshot(self.disk_path)
        for process in [psutil.Process()] + psutil.Process().children(recursive=True):
            try:
                key = (process.pid, process.create_time())
                times = process.cpu_times()
                self.cpu_seconds[key] = times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        now = perf_counter()
        cpu_total = sum(self.cpu_seconds.values())
        row["cpu_cores"] = ((cpu_total - self.last_cpu_total) / (now - self.last_sample_time)
                            if hasattr(self, "last_sample_time") else 0)
        self.last_cpu_total, self.last_sample_time = cpu_total, now
        row.update(gpu_util_pct=np.nan, device_vram_gib=np.nan)
        if self.gpu_uuid:
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "-i", self.gpu_uuid,
                    "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                ], text=True, timeout=2)
                util, memory = map(float, output.strip().split(","))
                row.update(gpu_util_pct=util, device_vram_gib=memory / 1024)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        self.samples.append(row)

    def sample_loop(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.gpu_uuid = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.gpu_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", "")) or None
        self.sample()
        self.cpu_before = sum(self.cpu_seconds.values())
        self.start = perf_counter()
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = perf_counter() - self.start
        self.stop.set()
        self.thread.join()
        self.sample()
        cores = (sum(self.cpu_seconds.values()) - self.cpu_before) / elapsed
        process = psutil.Process()
        budget = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else psutil.cpu_count()
        self.row = {"stage": self.stage, "wall_sec": elapsed, "sampled_avg_cpu_cores": cores,
                    "allocated_cpu_pct": 100 * cores / budget, **self.samples[-1]}
        for key in ("cpu_cores", "tree_rss_sum_gib", "tree_uss_gib", "tree_pss_gib", "device_vram_gib", "gpu_util_pct"):
            values = [s[key] for s in self.samples if np.isfinite(s[key])]
            self.row[f"peak_{key}"] = max(values) if values else np.nan
            if key == "gpu_util_pct":
                self.row["avg_gpu_util_pct"] = float(np.mean(values)) if values else np.nan
        if torch.cuda.is_available():
            self.row["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            self.row["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        print(self.row, flush=True)
