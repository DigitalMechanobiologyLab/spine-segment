from __future__ import annotations

import argparse
import os
from pathlib import Path

from spine_segment.api import segment_files
from spine_segment.backend import SpineSegmentBackendError, load_backend
from spine_segment.compartments import CortTrabConfig
from spine_segment.io import expand_input_paths
from spine_segment.model_bundle import (
    ModelBundleDownloadError, resolve_beta_detailed_model,
)
from spine_segment.native_torch_backend import create_backend as create_native_backend
from spine_segment.sequence_solver import SpineSequenceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spine-segment",
        description="Standalone vertebral segmentation CLI.",
    )
    parser.add_argument("inputs", nargs="+", help="Input CT volumes (.nii or .nii.gz).")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory that will receive outputs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Runtime device preference.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("SPINE_SEGMENT_BACKEND", ""),
        help="Backend factory in the form module_path:factory. "
        "Defaults to the native PyTorch backend.",
    )
    parser.add_argument(
        "--model-bundle",
        default=os.environ.get("SPINE_SEGMENT_MODEL_BUNDLE", ""),
        help="Path to a spine-segment model bundle. Defaults to "
        "SPINE_SEGMENT_MODEL_BUNDLE, ./model-bundle, ./build/model-bundle-pytorch, "
        "./build/model-bundle, or an automatically downloaded cached bundle.",
    )
    parser.add_argument(
        "--no-model-download",
        action="store_true",
        default=os.environ.get("SPINE_SEGMENT_NO_DOWNLOAD", "").strip().lower()
        not in ("", "0", "false", "no", "off"),
        help="Disable automatic first-run model bundle download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--level-only",
        action="store_true",
        help="Write *_vertebral-level.nii.gz and *_centroids.json; skip process/body plus cort/trab stages.",
    )
    parser.add_argument(
        "--localization-only",
        action="store_true",
        help="Write only *_centroids.json with detected vertebral centroids in original scan voxel coordinates.",
    )
    parser.add_argument(
        "--centroids",
        help="Existing *_centroids.json artifact used by --level-only instead of localization.",
    )
    parser.add_argument(
        "--beta-detailed-labels",
        action="store_true",
        help="Experimental ten-class vertebra labels; uses the packaged model or a verified download.",
    )
    parser.add_argument(
        "--beta-detailed-model",
        help="Optional override for the experimental Dataset737 model folder.",
    )
    parser.add_argument(
        "--sequence-confidence-weight",
        type=float,
        default=SpineSequenceConfig().lambda_weight,
        help="Localization confidence versus average-spine geometry weight (0 < weight < 1). "
        "Higher values preserve confident detections in curved spines; historical value: 0.2.",
    )
    parser.add_argument(
        "--vertebra-batch-size",
        type=int,
        default=1,
        help="Number of MDAT vertebra segmentation patches to run per PyTorch batch.",
    )
    parser.add_argument(
        "--process-body-batch-size",
        type=int,
        default=4,
        help="Number of process/body relabel crops to run per PyTorch batch.",
    )
    parser.add_argument(
        "--cortical-threshold-hu",
        type=float,
        default=500.0,
        help="HU threshold used for connected cortical assignment within the outer shell.",
    )
    parser.add_argument(
        "--cort-trab-isotropic-spacing-mm",
        type=float,
        default=1.0,
        help="Isotropic spacing used internally for cort/trab shell generation.",
    )
    parser.add_argument(
        "--cortical-max-thickness-mm",
        type=float,
        default=6.0,
        help="Maximum distance from the vertebral surface considered for threshold-based cortical assignment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = expand_input_paths(list(args.inputs))
    if not inputs:
        parser.error("No input images were resolved.")

    missing = [path for path in inputs if not path.exists()]
    if missing:
        parser.error(f"Input does not exist: {missing[0]}")

    if args.localization_only and args.level_only:
        parser.error("--localization-only and --level-only are mutually exclusive.")
    if args.centroids and not args.level_only:
        parser.error("--centroids requires --level-only.")
    if args.centroids and args.localization_only:
        parser.error("--centroids cannot be combined with --localization-only.")
    if args.centroids and len(inputs) != 1:
        parser.error("--centroids requires exactly one input CT.")
    if args.centroids and not Path(args.centroids).is_file():
        parser.error(f"Centroid artifact does not exist: {args.centroids}")
    if args.beta_detailed_model and args.beta_detailed_labels:
        parser.error("Choose either --beta-detailed-labels or --beta-detailed-model")
    if (args.beta_detailed_model or args.beta_detailed_labels) and (args.level_only or args.localization_only):
        parser.error("Beta detailed labels require full segmentation")
    if args.beta_detailed_model:
        from spine_segment.beta_subregions import _validate_model_folder
        try:
            _validate_model_folder(Path(args.beta_detailed_model))
        except ValueError as exc:
            parser.error(str(exc))

    try:
        sequence_config = SpineSequenceConfig(lambda_weight=args.sequence_confidence_weight)
    except ValueError as exc:
        parser.error(str(exc))

    if args.backend:
        backend = load_backend(args.backend)
    else:
        backend = create_native_backend(
            args.model_bundle or "./build/model-bundle",
            allow_download=not bool(args.no_model_download),
        )
    if hasattr(backend, "sequence_config"):
        backend.sequence_config = sequence_config
    if hasattr(backend, "vertebra_segmentation_batch_size"):
        backend.vertebra_segmentation_batch_size = max(1, int(args.vertebra_batch_size))
    if hasattr(backend, "process_body_batch_size"):
        backend.process_body_batch_size = max(1, int(args.process_body_batch_size))
    detailed_model = args.beta_detailed_model
    if args.beta_detailed_labels:
        try:
            detailed_model = resolve_beta_detailed_model(
                bundle_root=getattr(backend, "bundle_root", None),
                allow_download=not bool(args.no_model_download),
            )
        except (ModelBundleDownloadError, ValueError) as exc:
            print(f"[spine-segment] beta model error: {exc}")
            return 2
    cort_trab_config = CortTrabConfig(
        cortical_threshold_hu=float(args.cortical_threshold_hu),
        isotropic_spacing_mm=float(args.cort_trab_isotropic_spacing_mm),
        cortical_max_thickness_mm=float(args.cortical_max_thickness_mm),
    )
    try:
        results = segment_files(
            input_paths=inputs,
            output_dir=Path(args.output),
            backend=backend,
            device=args.device,
            overwrite=bool(args.overwrite),
            cort_trab_config=cort_trab_config,
            level_only=bool(args.level_only),
            localization_only=bool(args.localization_only),
            centroids_path=args.centroids,
            beta_detailed_model=detailed_model,
        )
    except SpineSegmentBackendError as exc:
        print(f"[spine-segment] backend error: {exc}")
        return 2
    except FileExistsError as exc:
        print(f"[spine-segment] output exists: {exc}")
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[spine-segment] input error: {exc}")
        return 2

    for result in results:
        fields = [
            f"input={result.input_path}",
            f"device={result.device}",
        ]
        if args.localization_only:
            fields.append(f"centroids={result.output_paths.centroids}")
        else:
            fields.append(f"vertebral_level={result.output_paths.vertebral_level}")
            fields.append(f"centroids={result.output_paths.centroids}")
            if not args.level_only:
                fields.append(f"process_body={result.output_paths.process_body}")
                fields.append(f"cort_trab={result.output_paths.cort_trab}")
                if detailed_model:
                    fields.append(f"detailed_subregions={result.output_paths.detailed_subregions}")
        print("[spine-segment] " + " ".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
