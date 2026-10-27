"""Opt-in, mask-only Dataset737 inference; labels are experimental pseudo-labels."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk


CLASS_NAMES = {
    1: "vertebral_body", 2: "superior_endplate_surface",
    3: "inferior_endplate_surface", 4: "focal_outgrowth_candidate",
    5: "pedicles", 6: "lamina", 7: "spinous_process",
    8: "transverse_processes", 9: "superior_articular_processes",
    10: "inferior_articular_processes",
}
THORACOLUMBAR_IDS = frozenset(range(8, 25))


def prepare_vertebra_crops(levels: sitk.Image, *, margin_mm: float = 8.0) -> dict[int, sitk.Image]:
    """Produce binary T1–L5 crops on the input grid; CT intensity is unused."""
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and nonnegative")
    array = sitk.GetArrayFromImage(levels)
    ids = sorted(set(map(int, np.unique(array))) & THORACOLUMBAR_IDS)
    if not ids:
        raise ValueError("No T1–L5 vertebrae found in the level segmentation")
    margin = np.ceil(margin_mm / np.asarray(levels.GetSpacing())[::-1]).astype(int)
    crops = {}
    for label in ids:
        indices = np.argwhere(array == label)
        lo = np.maximum(0, indices.min(axis=0) - margin)
        hi = np.minimum(array.shape, indices.max(axis=0) + margin + 1)
        roi = sitk.RegionOfInterest(
            levels, [int(v) for v in (hi - lo)[::-1]], [int(v) for v in lo[::-1]]
        )
        binary = sitk.GetImageFromArray((sitk.GetArrayFromImage(roi) == label).astype(np.uint8))
        binary.CopyInformation(roi)
        crops[label] = binary
    return crops


def assemble_predictions(
    levels: sitk.Image, predictions: dict[int, sitk.Image]
) -> tuple[sitk.Image, dict[int, dict[str, int]]]:
    """Clip predictions to owning bones and fill gaps from body/process only."""
    from scipy import ndimage as ndi

    source = sitk.GetArrayFromImage(levels)
    expected = set(map(int, np.unique(source))) & THORACOLUMBAR_IDS
    if set(predictions) != expected:
        raise ValueError("Prediction labels must match all T1–L5 vertebrae")
    out = np.zeros(source.shape, np.uint8)
    report = {}
    for label in sorted(expected):
        prediction = predictions[label]
        raw = sitk.GetArrayFromImage(prediction)
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError(f"Vertebra {label} detailed labels must be finite integers")
        unknown = set(map(int, np.unique(raw))) - set(range(11))
        if unknown:
            raise ValueError(f"Vertebra {label} has unknown detailed labels: {sorted(unknown)}")
        if (prediction.GetDimension() != 3
                or not np.allclose(prediction.GetSpacing(), levels.GetSpacing(), atol=1e-5)
                or not np.allclose(prediction.GetDirection(), levels.GetDirection(), atol=1e-5)):
            raise ValueError(f"Vertebra {label} prediction geometry differs from level grid")
        step = np.linalg.solve(
            np.asarray(levels.GetDirection()).reshape(3, 3),
            np.asarray(prediction.GetOrigin()) - np.asarray(levels.GetOrigin()),
        ) / np.asarray(levels.GetSpacing())
        if not np.allclose(step, np.rint(step), atol=1e-3, rtol=0):
            raise ValueError(f"Vertebra {label} prediction is off the level voxel grid")
        indices = np.argwhere(source == label)
        lo = indices.min(axis=0)
        hi = indices.max(axis=0) + 1
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        roi = sitk.RegionOfInterest(
            levels, [int(v) for v in (hi - lo)[::-1]], [int(v) for v in lo[::-1]]
        )
        aligned = sitk.GetArrayFromImage(sitk.Resample(
            prediction, roi, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        ))
        owner = source[slices] == label
        local = out[slices]
        local[owner] = aligned[owner]
        clipped = int(np.count_nonzero(raw) - np.count_nonzero(aligned[owner]))
        missing = owner & (local == 0)
        eligible = owner & np.isin(local, (1, 5, 6, 7, 8, 9, 10))
        if not eligible.any():
            raise ValueError(f"Vertebra {label} has no body/process voxels to fill gaps")
        if missing.any():
            nearest = ndi.distance_transform_edt(
                ~eligible, return_distances=False, return_indices=True
            )
            local[missing] = local[tuple(axis[missing] for axis in nearest)]
        report[label] = {
            "owner_voxels": int(owner.sum()),
            "clipped_outside_bone_voxels": clipped,
            "filled_inside_bone_voxels": int(missing.sum()),
        }
    result = sitk.GetImageFromArray(out)
    result.CopyInformation(levels)
    return result, report


def _validate_model_folder(model_folder: Path) -> None:
    dataset = model_folder / "dataset.json"
    plans = model_folder / "plans.json"
    checkpoint = model_folder / "fold_0" / "checkpoint_best.pth"
    if not all(path.is_file() for path in (dataset, plans, checkpoint)):
        raise ValueError("Beta model needs dataset.json, plans.json and fold_0/checkpoint_best.pth")
    labels = json.loads(dataset.read_text(encoding="utf-8")).get("labels", {})
    if labels.get("background") != 0 or {name: value for name, value in labels.items() if name != "background"} != {
        name: index for index, name in CLASS_NAMES.items()
    }:
        raise ValueError("Beta model must use the exact Dataset737 class schema")


def _make_predictor(model_folder: Path, device: str):
    try:
        import torch
        from nnunetv2.inference import predict_from_raw_data as raw
        from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoDA import nnUNetTrainerNoDA
    except ImportError as exc:
        raise RuntimeError("Beta detailed labels require the optional nnunetv2 runtime") from exc
    chosen = torch.device(device if device in ("cpu", "cuda", "mps") else
                          "cuda" if torch.cuda.is_available() else "cpu")
    predictor = raw.nnUNetPredictor(device=chosen, perform_everything_on_device=chosen.type == "cuda")
    # RimPilotTrainer only changes training length from nnUNetTrainerNoDA.
    # Its released checkpoint therefore uses the parent architecture.
    original = raw.recursive_find_trainer_class_by_name
    def resolve(name):
        return nnUNetTrainerNoDA if name == "RimPilotTrainer" else original(name)
    raw.recursive_find_trainer_class_by_name = resolve
    try:
        predictor.initialize_from_trained_model_folder(
            str(model_folder), use_folds=(0,), checkpoint_name="checkpoint_best.pth"
        )
    finally:
        raw.recursive_find_trainer_class_by_name = original
    return predictor


def predict_detailed_subregions(
    levels: sitk.Image, model_folder: str | Path, *, device: str = "auto"
) -> tuple[sitk.Image, dict[int, dict[str, int]]]:
    """Run the opt-in Dataset737 model separately on each vertebra mask."""
    folder = Path(model_folder)
    _validate_model_folder(folder)
    crops = prepare_vertebra_crops(levels)
    predictor = _make_predictor(folder, device)
    with TemporaryDirectory(prefix="spine-segment-beta-") as temporary:
        root = Path(temporary)
        inputs = root / "inputs"
        outputs = root / "outputs"
        inputs.mkdir()
        outputs.mkdir()
        for label, crop in crops.items():
            sitk.WriteImage(crop, str(inputs / f"vertebra_{label:02d}_0000.nii.gz"))
        predictor.predict_from_files_sequential(str(inputs), str(outputs), False, True)
        predictions = {}
        for label in crops:
            path = outputs / f"vertebra_{label:02d}.nii.gz"
            if not path.is_file():
                raise RuntimeError(f"Missing beta prediction for vertebra {label}")
            predictions[label] = sitk.ReadImage(str(path))
        return assemble_predictions(levels, predictions)
