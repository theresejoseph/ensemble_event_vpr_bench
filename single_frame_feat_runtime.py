# single_query_runtime.py
# Runtime analysis: Brisbane night (ref) vs Brisbane morning (qry)
# Measures three buckets only:
#   1) loading event frame data into the model pipeline (dataloader loop minus forward)
#   2) feature extraction forward pass time
#   3) similarity matrix generation time
# Writes a CSV with timings per combo.

# --- import shim so script runs from repo root OR one folder above ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent
_VMETH = _REPO / "vpr_methods_evaluation"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_VMETH) not in sys.path:
    sys.path.insert(0, str(_VMETH))

# Provide alias so modules that do `from vpr_models import ...` still work.
try:
    import importlib
    pkg = importlib.import_module("vpr_methods_evaluation.vpr_models")
    sys.modules.setdefault("vpr_models", pkg)
except Exception:
    pass

import argparse
import csv
import time
from glob import glob
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from vpr_methods_evaluation.parse import parse_arguments
from vpr_methods_evaluation.test_dataset import TestDataset
import vpr_methods_evaluation.vpr_models as vpr_models
from load_and_save import make_paths


# ---------------------- helpers ----------------------
def ensure_model_cfg(args, strict: bool = False):
    """Normalize and validate model configuration on args; fill defaults if missing."""
    def _setdefault(name, value):
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, value)

    _setdefault("method", "megaloc")
    if not hasattr(args, "backbone"):
        args.backbone = None
    if not hasattr(args, "descriptors_dimension"):
        args.descriptors_dimension = None
    if not hasattr(args, "no_labels"):
        args.no_labels = False
    args.use_labels = 0 if args.no_labels else 1

    try:
        if not hasattr(args, "device") or args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"
    except Exception:
        _setdefault("device", "cpu")

    _setdefault("num_workers", 4)
    _setdefault("batch_size", 4)
    _setdefault("pin_memory", 1)
    _setdefault("positive_dist_threshold", 25)

    img = getattr(args, "image_size", None)
    if isinstance(img, int):
        args.image_size = [img]
    elif isinstance(img, (list, tuple)):
        args.image_size = list(img)
    elif img is None:
        args.image_size = None
    else:
        raise ValueError(f"image_size must be int, [H,W], or None, got {img!r}")

    back_norm = {
        None: None,
        "vgg16": "VGG16", "VGG16": "VGG16",
        "resnet18": "ResNet18", "ResNet18": "ResNet18",
        "resnet50": "ResNet50", "ResNet50": "ResNet50",
        "resnet101": "ResNet101", "ResNet101": "ResNet101", "Resnet101": "ResNet101",
        "resnet152": "ResNet152", "ResNet152": "ResNet152",
        "dinov2": "Dinov2", "DINOv2": "Dinov2", "Dinov2": "Dinov2",
    }
    bb_in = getattr(args, "backbone", None)
    args.backbone = back_norm.get(bb_in, bb_in)

    m = args.method

    # Per-method rules
    if m == "netvlad":
        allowed = [None, "VGG16"]
        if args.backbone not in allowed:
            if strict:
                raise ValueError("NetVLAD backbone must be None or VGG16")
            args.backbone = "VGG16"
        if args.descriptors_dimension not in [None, 4096, 32768]:
            if strict:
                raise ValueError("NetVLAD dim must be one of [None, 4096, 32768]")
            args.descriptors_dimension = 4096
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 4096

    elif m == "sfrs":
        if args.backbone not in [None, "VGG16"]:
            raise ValueError("SFRS backbone must be None or VGG16")
        if args.descriptors_dimension not in [None, 4096]:
            raise ValueError("SFRS descriptors_dimension must be one of [None, 4096]")
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 4096

    elif m == "cosplace":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 2048
        if args.backbone == "VGG16" and args.descriptors_dimension not in [64, 128, 256, 512]:
            raise ValueError("CosPlace + VGG16 needs dim in [64,128,256,512]")
        if args.backbone == "ResNet18" and args.descriptors_dimension not in [32, 64, 128, 256, 512]:
            raise ValueError("CosPlace + ResNet18 needs dim in [32,64,128,256,512]")
        if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and \
           args.descriptors_dimension not in [32, 64, 128, 256, 512, 1024, 2048]:
            raise ValueError(f"CosPlace + {args.backbone} needs dim in [32..2048]")

    elif m == "convap":
        if args.backbone not in [None, "ResNet50"]:
            raise ValueError("Conv-AP backbone must be None or ResNet50")
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 8192
        if args.descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
            raise ValueError("Conv-AP dim must be one of [None,512,2048,4096,8192]")

    elif m == "mixvpr":
        if args.backbone not in [None, "ResNet50"]:
            raise ValueError("MixVPR backbone must be None or ResNet50")
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 4096
        if args.descriptors_dimension not in [None, 128, 512, 4096]:
            raise ValueError("MixVPR dim must be one of [None,128,512,4096]")

    elif m == "eigenplaces":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 2048
        if args.backbone == "VGG16" and args.descriptors_dimension not in [512]:
            raise ValueError("EigenPlaces + VGG16 needs dim=512")
        if args.backbone == "ResNet18" and args.descriptors_dimension not in [256, 512]:
            raise ValueError("EigenPlaces + ResNet18 needs dim in [256,512]")
        if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and \
           args.descriptors_dimension not in [128, 256, 512, 2048]:
            raise ValueError(f"EigenPlaces + {args.backbone} needs dim in [128,256,512,2048]")

    elif m == "eigenplaces-indoor":
        args.backbone = "ResNet50"
        args.descriptors_dimension = 2048

    elif m == "apgem":
        args.backbone = "ResNet101"
        args.descriptors_dimension = 2048

    elif m.startswith("anyloc"):
        args.backbone = "Dinov2"
        args.descriptors_dimension = 49152

    elif m == "salad":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 8448

    elif m == "clique-mining":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 8448

    elif m == "salad-indoor":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 8448

    elif m == "cricavpr":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 10752

    elif m == "megaloc":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 8448

    elif m == "boq":
        if args.backbone not in ["ResNet50", "Dinov2", None]:
            raise ValueError("BoQ backbone must be ResNet50 or Dinov2")
        if args.backbone in [None, "ResNet50"]:
            args.backbone = "ResNet50"
            args.descriptors_dimension = 16384
            args.image_size = [384, 384]
        else:
            args.descriptors_dimension = 12288
            args.image_size = [322, 322]

    elif m == "dinomix":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 4096
        args.image_size = [224, 224]

    elif m == "sad":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 512

    # Post-validation
    if args.image_size is not None:
        if not (isinstance(args.image_size, list) and all(isinstance(x, int) for x in args.image_size)):
            raise ValueError("image_size must be a list of ints (e.g., [H, W]) or None")
        if len(args.image_size) > 2:
            raise ValueError(f"--image_size takes up to 2 values, got {len(args.image_size)}")

    if args.descriptors_dimension is not None:
        if not isinstance(args.descriptors_dimension, int) or args.descriptors_dimension <= 0:
            raise ValueError("descriptors_dimension must be a positive int")

    return args


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def prepare_folders(args: Any, ref_seq: str, qry_seq: str) -> Dict[str, int]:
    make_paths(args, ref_seq)
    args.database_folder = str(args.save_images_dir)
    make_paths(args, qry_seq)
    args.queries_folder = str(args.save_images_dir)
    n_ref = len(glob(f"{args.database_folder}/**/*", recursive=True))
    n_qry = len(glob(f"{args.queries_folder}/**/*", recursive=True))
    return {"n_ref": n_ref, "n_qry": n_qry}


# ---------------------- timing core ----------------------
@torch.inference_mode()
def time_load_and_extract(args: Any) -> Dict[str, Any]:
    """Time event loading, feature extraction, and similarity-matrix build."""
    ensure_model_cfg(args)

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=getattr(args, "positive_dist_threshold", 25),
        image_size=getattr(args, "image_size", None),
        use_labels=getattr(args, "use_labels", 0),
    )

    all_desc = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")

    db_loader = DataLoader(
        dataset=Subset(test_ds, range(test_ds.num_database)),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=bool(getattr(args, "pin_memory", 0)),
    )
    qry_loader = DataLoader(
        dataset=Subset(test_ds, range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)),
        num_workers=args.num_workers,
        batch_size=1,
        pin_memory=bool(getattr(args, "pin_memory", 0)),
    )

    t_forward = 0.0
    t_loop = 0.0

    def run_loader(loader):
        nonlocal t_forward, t_loop
        for images, indices in loader:
            t0 = time.perf_counter()
            images = images.to(args.device, non_blocking=True)
            t1 = time.perf_counter()
            desc = model(images)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            t_forward += (t2 - t1)
            t_loop += (t2 - t0)
            all_desc[indices.numpy(), :] = desc.detach().cpu().numpy()

    run_loader(db_loader)
    run_loader(qry_loader)

    database_descriptors = all_desc[:test_ds.num_database]
    queries_descriptors = all_desc[test_ds.num_database:]

    t_feat_total_s = t_forward
    t_load_total_s = max(0.0, t_loop - t_forward)

    n_total_frames = test_ds.num_database + test_ds.num_queries
    t_load_per_frame_s = t_load_total_s / max(1, n_total_frames)
    t_feat_per_frame_s = t_feat_total_s / max(1, n_total_frames)

    # Similarity matrix timing (L2; S = -||q - r||^2)
    q = queries_descriptors
    r = database_descriptors
    t0 = time.perf_counter()
    q2 = (q ** 2).sum(axis=1, keepdims=True)
    r2 = (r ** 2).sum(axis=1, keepdims=True).T
    qr = q @ r.T
    _S = -(q2 + r2 - 2.0 * qr)
    t_sim_mat_s = time.perf_counter() - t0

    return {
        "t_load_total_s": t_load_total_s,
        "t_load_per_frame_s": t_load_per_frame_s,
        "t_feat_total_s": t_feat_total_s,
        "t_feat_per_frame_s": t_feat_per_frame_s,
        "t_sim_mat_s": t_sim_mat_s,
        "n_ref_frames": test_ds.num_database,
        "n_qry_frames": test_ds.num_queries,
    }


def do_one_combo(recon_name: str, dt: float, method: str,
                 ref_seq: str = "night", qry_seq: str = "morning") -> Dict[str, Any]:
    """Build fresh args via your repo parser, prep folders, and time the combo."""
    a = parse_arguments(method=method)
    a.reconstruct_method_name = recon_name
    a.time_res = float(dt)
    a.count_bin = 0
    if getattr(a, "adaptive_bin", None) is None:
        a.adaptive_bin = 0
    if getattr(a, "events_per_bin", None) is None:
        a.events_per_bin = 100_000

    counts = prepare_folders(a, ref_seq, qry_seq)
    stats = time_load_and_extract(a)

    return {
        "reconstruction": recon_name,
        "time_res": dt,
        "feature_extractor": method,
        "dataset": "Brisbane",
        "ref_seq": ref_seq,
        "qry_seq": qry_seq,
        "n_ref_frames": counts["n_ref"],
        "n_qry_frames": counts["n_qry"],
        "t_load_total_s": stats["t_load_total_s"],
        "t_load_per_frame_s": stats["t_load_per_frame_s"],
        "t_feat_total_s": stats["t_feat_total_s"],
        "t_feat_per_frame_s": stats["t_feat_per_frame_s"],
        "t_sim_mat_s": stats["t_sim_mat_s"],
    }


def run_grid_exact(recon_methods: List[str],
                   time_resolutions: List[float],
                   feature_extractors: List[str],
                   csv_path: str) -> None:
    rows: List[Dict[str, Any]] = []

    for method in feature_extractors:
        rows.append(do_one_combo('timeSurface', '1.0', method,
                                    ref_seq="night", qry_seq="morning"))

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} with {len(rows)} rows.")


# ---------------------- CLI ----------------------
def make_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="./results/single_query_load_feat_runtime.csv")

    # binning knobs (time-based only) – kept for parity if your parse() uses them
    ap.add_argument("--dataset_type", type=str, default="Brisbane")
    ap.add_argument("--bin_tag", type=str, default="single_q_runtime")
    ap.add_argument("--adaptive_bin", type=int, default=0)
    ap.add_argument("--max_bins", type=int, default=20)
    ap.add_argument("--odom_weights", type=str, default="0.5,0.05,0,0")
    ap.add_argument("--max_odoms", type=str, default="5,16,1,10")
    ap.add_argument("--use_exponential", type=int, default=0)
    ap.add_argument("--events_per_bin", type=int, default=100000)

    # perf knobs (optional; actual model/device resolved in ensure_model_cfg)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", type=int, default=1)
    return ap.parse_args([])


# ---------------------- main ----------------------
if __name__ == "__main__":
    RECONS = ["eventCount", "eventCount_noPolarity", "timeSurface", "e2vid"]
    TIME_RES = [1.0, 0.5, 0.25, 0.1]
    METHODS = ["mixvpr", "megaloc", "netvlad", "cosplace"]  # must be valid vpr_models keys

    cli = make_cli()
    # If you need these CLI binning knobs to affect parse_arguments, pass via env or modify parse() accordingly.
    # Here they are not directly used because we construct method-specific args inside do_one_combo.

    run_grid_exact(
        recon_methods=RECONS,
        time_resolutions=TIME_RES,
        feature_extractors=METHODS,
        csv_path=cli.csv_path,
    )
