# recon_runtime.py
# Runtime analysis: Brisbane night (ref) vs Brisbane morning (qry)
# 4 reconstructions × 4 time resolutions × 4 feature extractors (labels only; VPR not timed)
# No I/O. Reconstruction timing = full-sequence time / number of frames (query only).
# Writes CSV with per-frame and total reconstruction times per combo.

import argparse
import csv
import importlib
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from parser_config import get_parser, apply_defaults
from datasets.brisbane_events import BrisbaneEventDataset, Brisbane_RGB_Dataset


# ---------------------- helpers ----------------------
def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def base_args(dataset_type: str) -> Any:
    """Build a fresh args namespace with defaults applied."""
    p = get_parser()
    a = p.parse_args(["--dataset_type", dataset_type])
    a = apply_defaults(a)
    return a


def build_recon_args(
    dataset_type: str,
    cli: argparse.Namespace,
    recon_name: str,
    time_res: float,
    ref_idx: int,
    qry_idx: int,
) -> Any:
    """
    Build a fresh namespace for reconstruction timing (time-based binning only).
    """
    a = base_args(dataset_type)
    a.dataset_type = "Brisbane"
    a.reconstruct_method_name = recon_name

    # binning / recon params (force time-based)
    a.adaptive_bin = int(cli.adaptive_bin)
    a.max_odoms = list(cli.max_odoms)
    a.odom_weights = list(cli.odom_weights)
    a.max_bins = int(cli.max_bins)
    a.adaptive_bin_tag = cli.bin_tag
    a.use_exponential = int(cli.use_exponential)
    a.time_res = float(time_res)
    a.count_bin = 0
    a.events_per_bin = int(cli.events_per_bin)

    # sequences and ids (Brisbane indexes: 0 sunset1, 1 night)
    a.sequences = ["sunset1", "night"]
    a.ref_seq_idx = int(ref_idx)
    a.qry_seq_idx = int(qry_idx)

    # explicitly disable video saving in downstream code if present
    if not hasattr(a, "save_frames_video"):
        a.save_frames_video = 0

    return a


def init_dataset_and_reconstructor(args: Any) -> Tuple[Any, Any]:
    """
    Create dataset and reconstructor objects for Brisbane.
    """
    dataset = BrisbaneEventDataset(args.dataset_path)

    if args.reconstruct_method_name != "RGB_camera":
        module_path = f"reconstruction.{args.reconstruct_method_name}"
        reconstruction_module = importlib.import_module(module_path)
        reconstructor_class = getattr(reconstruction_module, "EventReconstructor")
        reconstructor = reconstructor_class()
    else:
        # Not expected in this sweep; kept for completeness
        reconstructor = None
        dataset = Brisbane_RGB_Dataset(args.dataset_path)

    return dataset, reconstructor


def measure_query_recon_time(dataset, reconstructor, args: Any) -> Dict[str, float]:
    """
    Reconstruct the entire query sequence once, measure total time,
    then compute per-frame time = total / num_frames. No I/O here.
    Returns dict with total, per-frame, and frame count.
    """
    t0 = time.perf_counter()
    frames, gt_positions = dataset.process_sequence(args, "qry", reconstructor)
    total_s = time.perf_counter() - t0
    n = len(frames)
    per_frame = float("inf") if n == 0 else total_s / n

    # free RAM early
    del frames, gt_positions
    return {"recon_total_s": total_s, "recon_per_frame_s": per_frame, "qry_frames": n}


# cache metrics per (recon, dt, ref_idx, qry_idx)
_RECON_METRICS_CACHE: Dict[Tuple[str, float, int, int], Dict[str, float]] = {}


def time_recon_only(
    dataset_type: str,
    cli: argparse.Namespace,
    recon_name: str,
    time_res: float,
    ref_idx: int,
    qry_idx: int,
) -> Dict[str, float]:
    """Measure reconstruction once per (recon, dt, pair)."""
    key = (recon_name, float(time_res), int(ref_idx), int(qry_idx))
    if key in _RECON_METRICS_CACHE:
        return _RECON_METRICS_CACHE[key]

    args = build_recon_args(dataset_type, cli, recon_name, time_res, ref_idx, qry_idx)
    dataset, reconstructor = init_dataset_and_reconstructor(args)

    m = measure_query_recon_time(dataset, reconstructor, args)
    out = {
        "t_recon_total": m["recon_total_s"],
        "t_recon_per_frame": m["recon_per_frame_s"],
        "n_q_frames": m["qry_frames"],
    }
    _RECON_METRICS_CACHE[key] = out
    return out


def run_grid(
    dataset_type: str,
    recon_methods: List[str],
    time_resolutions: List[float],
    feature_extractors: List[str],
    pairs: List[Tuple[int, int]],
    repeats: int,
    cli: argparse.Namespace,
    csv_path: str = "single_query_grid.csv",
) -> None:
    """
    For each (reconstruction, dt), measure query reconstruction time once,
    average over repeats, then duplicate the metric across feature extractors.
    """
    rows: List[Dict[str, Any]] = []
    ref_idx, qry_idx = pairs[0]  # e.g., [(0, 1)]

    for recon_name in recon_methods:
        for dt in time_resolutions:
            totals, per_frames, n_qs = [], [], []
            for _ in range(max(1, repeats)):
                m = time_recon_only(dataset_type, cli, recon_name, dt, ref_idx, qry_idx)
                totals.append(m["t_recon_total"])
                per_frames.append(m["t_recon_per_frame"])
                n_qs.append(m["n_q_frames"])

            mean_total = statistics.fmean(totals)
            mean_per_frame = statistics.fmean(per_frames)
            mean_nq = statistics.fmean(n_qs)
            std_total = statistics.pstdev(totals) if repeats > 1 else 0.0

            # duplicate across feature extractors (labels only)
            for feat in feature_extractors:
                rows.append({
                    "reconstruction": recon_name,
                    "time_res": dt,
                    "feature_extractor": feat,
                    "dataset": dataset_type,
                    "ref_seq": "night",
                    "qry_seq": "morning",
                    "repeats": repeats,
                    "mean_recon_total_s": mean_total,
                    "std_recon_total_s": std_total,
                    "mean_recon_per_frame_s": mean_per_frame,
                    "mean_qry_frames": mean_nq,
                })

    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows).")


# ---------------------- CLI ----------------------
def make_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--csv_path", type=str, default="single_query_grid.csv")
    ap.add_argument("--dataset_type", type=str, default="Brisbane")

    # binning knobs (time-based only)
    ap.add_argument("--bin_tag", type=str, default="single_q_runtime")
    ap.add_argument("--adaptive_bin", type=int, default=0)
    ap.add_argument("--max_bins", type=int, default=20)
    ap.add_argument("--odom_weights", type=str, default="0.5,0.05,0,0")
    ap.add_argument("--max_odoms", type=str, default="5,16,1,10")
    ap.add_argument("--use_exponential", type=int, default=0)
    ap.add_argument("--events_per_bin", type=int, default=100000)
    return ap.parse_args([])


# ---------------------- main ----------------------
if __name__ == "__main__":
    # 4×4×4 definition (feature_extractors are labels only; VPR not timed)
    RECONS = ["eventCount", "eventCount_noPolarity", "timeSurface", "e2vid"]
    TIME_RES = [1.0, 0.5, 0.25, 0.1]
    METHODS = ["mixvpr", "megaloc", "netvlad", "cosplace"]

    # Brisbane only: sunset1 (0) vs night (1); query is "morning" in naming, but
    # your dataset indices map to ["sunset1", "night"]. We keep indices as in your original script.
    EXP_PAIRS = [(0, 1)]

    cli = make_cli()
    cli.odom_weights = parse_float_list(cli.odom_weights)
    cli.max_odoms = parse_float_list(cli.max_odoms)

    run_grid(
        dataset_type=cli.dataset_type,
        recon_methods=RECONS,
        time_resolutions=TIME_RES,
        feature_extractors=METHODS,
        pairs=EXP_PAIRS,
        repeats=int(cli.repeats),
        cli=cli,
        csv_path=cli.csv_path,
    )
