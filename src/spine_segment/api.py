from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from spine_segment.backend import SpineSegmentBackend, SpineSegmentBackendError
from spine_segment.compartments import CortTrabConfig, derive_cort_trab_labels
from spine_segment.device import resolve_device
from spine_segment.io import read_image, write_image
from spine_segment.outputs import OutputPaths, build_output_paths


@dataclass(frozen=True, slots=True)
class SegmentRunResult:
    input_path: Path
    output_paths: OutputPaths
    device: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CentroidArtifact:
    centroids: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    input_path: str | None = None
    coordinate_system: str = "voxel_xyz"


def segment_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    backend: SpineSegmentBackend,
    device: str = "auto",
    overwrite: bool = False,
    cort_trab_config: CortTrabConfig | None = None,
    level_only: bool = False,
    localization_only: bool = False,
    centroids_path: str | Path | None = None,
    beta_detailed_labels: bool = False,
    beta_detailed_model: str | Path | None = None,
) -> SegmentRunResult:
    if level_only and localization_only:
        raise ValueError("level_only and localization_only are mutually exclusive")
    if centroids_path is not None and not level_only:
        raise ValueError("centroids_path requires level_only=True")
    if centroids_path is not None and localization_only:
        raise ValueError("centroids_path cannot be combined with localization_only=True")
    if beta_detailed_labels and beta_detailed_model is not None:
        raise ValueError("Choose beta_detailed_labels or beta_detailed_model")
    if (beta_detailed_labels or beta_detailed_model is not None) and (level_only or localization_only):
        raise ValueError("Beta detailed labels require full segmentation")
    if beta_detailed_labels:
        from spine_segment.model_bundle import resolve_beta_detailed_model
        beta_detailed_model = resolve_beta_detailed_model(
            bundle_root=getattr(backend, "bundle_root", None)
        )

    source_path = Path(input_path)
    image = read_image(source_path, pixel_type=sitk.sitkFloat32)
    selected_device = resolve_device(device)
    output_paths = build_output_paths(source_path, output_dir)
    if beta_detailed_model is not None and not overwrite:
        for path in (output_paths.vertebral_level, output_paths.process_body,
                     output_paths.cort_trab, output_paths.centroids,
                     output_paths.detailed_subregions):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if localization_only:
        localization = backend.localize(
            image=image,
            source_path=source_path,
            device=selected_device,
        )
        centroids = localization.centroids
        metadata = dict(localization.metadata or {})
        metadata["device"] = selected_device
        metadata["localization_only"] = True
        write_json(
            {
                "input": str(source_path),
                "coordinate_system": "voxel_xyz",
                "centroids": centroids,
                "metadata": metadata,
            },
            output_paths.centroids,
            overwrite=overwrite,
        )
        return SegmentRunResult(
            input_path=source_path,
            output_paths=output_paths,
            device=selected_device,
            metadata=metadata,
        )

    centroid_artifact = None
    if centroids_path is not None:
        centroid_artifact = load_centroid_artifact(centroids_path, image=image)
        if not hasattr(backend, "segment_levels"):
            raise SpineSegmentBackendError(
                "The selected backend does not support centroid-driven level segmentation."
            )
        result = backend.segment_levels(
            image=image,
            source_path=source_path,
            device=selected_device,
            centroids=centroid_artifact.centroids,
        )
    else:
        result = backend.segment(
            image=image,
            source_path=source_path,
            device=selected_device,
            level_only=level_only,
        )

    detailed = report = None
    if beta_detailed_model is not None:
        if result.process_body is None:
            raise SpineSegmentBackendError(
                "Beta detailed labels require a full process/body segmentation."
            )
        from spine_segment.beta_subregions import predict_detailed_subregions
        detailed, report = predict_detailed_subregions(
            result.vertebral_level, beta_detailed_model, device=selected_device
        )

    write_image(result.vertebral_level, output_paths.vertebral_level, overwrite=overwrite)
    centroids = centroids_from_labelmap(result.vertebral_level)
    if level_only:
        metadata = dict(result.metadata or {})
        metadata["device"] = selected_device
        metadata["level_only"] = True
        if centroid_artifact is not None:
            centroids = merge_centroid_metadata(centroids, centroid_artifact.centroids)
            metadata["centroids_input"] = str(Path(centroids_path))
            metadata["input_centroids_metadata"] = centroid_artifact.metadata
        write_centroids_json(
            input_path=source_path,
            centroids=centroids,
            metadata=metadata,
            path=output_paths.centroids,
            overwrite=overwrite,
        )
        return SegmentRunResult(
            input_path=source_path,
            output_paths=output_paths,
            device=selected_device,
            metadata=metadata,
        )

    if result.process_body is None:
        raise SpineSegmentBackendError(
            "Standalone spine backend did not return a process/body labelmap."
        )

    cort_trab = result.cort_trab or derive_cort_trab_labels(
        image=image,
        process_body=result.process_body,
        vertebral_level=result.vertebral_level,
        config=cort_trab_config,
    )

    write_image(result.process_body, output_paths.process_body, overwrite=overwrite)
    write_image(cort_trab, output_paths.cort_trab, overwrite=overwrite)

    if beta_detailed_model is not None:
        write_image(detailed, output_paths.detailed_subregions, overwrite=overwrite)

    metadata = dict(result.metadata or {})
    metadata["device"] = selected_device
    if beta_detailed_model is not None:
        metadata["beta_detailed_subregions"] = {
            "schema": "dataset737", "anatomical_validation": False,
            "outgrowth_is_diagnostic": False, "levels": report,
        }
    write_centroids_json(
        input_path=source_path,
        centroids=centroids,
        metadata=metadata,
        path=output_paths.centroids,
        overwrite=overwrite,
    )
    return SegmentRunResult(
        input_path=source_path,
        output_paths=output_paths,
        device=selected_device,
        metadata=metadata,
    )


def segment_files(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    backend: SpineSegmentBackend,
    device: str = "auto",
    overwrite: bool = False,
    cort_trab_config: CortTrabConfig | None = None,
    level_only: bool = False,
    localization_only: bool = False,
    centroids_path: str | Path | None = None,
    beta_detailed_labels: bool = False,
    beta_detailed_model: str | Path | None = None,
) -> list[SegmentRunResult]:
    if centroids_path is not None and len(input_paths) != 1:
        raise ValueError("centroids_path requires exactly one input CT")
    return [
        segment_file(
            path,
            output_dir,
            backend=backend,
            device=device,
            overwrite=overwrite,
            cort_trab_config=cort_trab_config,
            level_only=level_only,
            localization_only=localization_only,
            centroids_path=centroids_path,
            beta_detailed_labels=beta_detailed_labels,
            beta_detailed_model=beta_detailed_model,
        )
        for path in input_paths
    ]


def load_centroid_artifact(
    path: str | Path,
    *,
    image: sitk.Image,
) -> CentroidArtifact:
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid centroid JSON: {artifact_path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Centroid artifact must contain a JSON object")
    if payload.get("coordinate_system") != "voxel_xyz":
        raise ValueError("Centroid artifact coordinate_system must be 'voxel_xyz'")

    raw_centroids = payload.get("centroids")
    if not isinstance(raw_centroids, dict) or not raw_centroids:
        raise ValueError("Centroid artifact must contain a non-empty 'centroids' object")
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("Centroid artifact 'metadata' must be an object")

    validated: dict[str, dict[str, Any]] = {}
    for key, raw_entry in raw_centroids.items():
        if not isinstance(key, str) or not isinstance(raw_entry, dict):
            raise ValueError("Each centroid must be keyed by a string label and contain an object")
        try:
            label = int(key)
        except ValueError as exc:
            raise ValueError(f"Centroid key is not a vertebral label: {key!r}") from exc
        expected_index = _verse_index_from_label(label)
        if expected_index is None:
            raise ValueError(f"Unsupported vertebral centroid label: {label}")
        if raw_entry.get("label") != label:
            raise ValueError(f"Centroid {key} has an inconsistent 'label'")
        if raw_entry.get("index") != expected_index:
            raise ValueError(f"Centroid {key} has an inconsistent VerSe 'index'")

        voxel_xyz = _coordinate_triplet(raw_entry.get("voxel_xyz"), key, "voxel_xyz")
        physical_xyz = _coordinate_triplet(
            raw_entry.get("physical_xyz"), key, "physical_xyz"
        )
        if any(
            value < 0.0 or value > float(size - 1)
            for value, size in zip(voxel_xyz, image.GetSize())
        ):
            raise ValueError(f"Centroid {key} is outside the input CT")
        expected_physical = image.TransformContinuousIndexToPhysicalPoint(voxel_xyz)
        if not np.allclose(physical_xyz, expected_physical, rtol=0.0, atol=1e-3):
            raise ValueError(
                f"Centroid {key} physical_xyz does not match the input CT geometry"
            )
        validated[key] = dict(raw_entry)

    input_path = payload.get("input")
    if input_path is not None and not isinstance(input_path, str):
        raise ValueError("Centroid artifact 'input' must be a string when present")
    return CentroidArtifact(
        centroids=validated,
        metadata=dict(raw_metadata),
        input_path=input_path,
    )


def _coordinate_triplet(value: Any, label: str, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Centroid {label} '{field}' must contain three numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Centroid {label} '{field}' must contain three numbers")
    result = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"Centroid {label} '{field}' must contain finite numbers")
    return result  # type: ignore[return-value]


def merge_centroid_metadata(
    segmented: dict[str, dict[str, Any]],
    supplied: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {
        label: {
            **entry,
            **segmented.get(label, {}),
            "segmentation_status": "segmented" if label in segmented else "missing",
        }
        for label, entry in supplied.items()
    }
    for label, entry in segmented.items():
        merged.setdefault(label, {**entry, "segmentation_status": "segmented"})
    return merged


def write_json(payload: dict[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output without --overwrite: {output_path}"
        )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_centroids_json(
    *,
    input_path: Path,
    centroids: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    path: str | Path,
    overwrite: bool,
) -> Path:
    return write_json(
        {
            "input": str(input_path),
            "coordinate_system": "voxel_xyz",
            "centroids": centroids,
            "metadata": metadata,
        },
        path,
        overwrite=overwrite,
    )


def centroids_from_labelmap(labelmap: sitk.Image) -> dict[str, dict[str, Any]]:
    labels = sitk.GetArrayFromImage(labelmap)
    centroids: dict[str, dict[str, Any]] = {}
    for raw_label in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
        coords_zyx = np.argwhere(labels == raw_label)
        if coords_zyx.size == 0:
            continue
        centroid_zyx = coords_zyx.mean(axis=0)
        voxel_xyz = (
            float(centroid_zyx[2]),
            float(centroid_zyx[1]),
            float(centroid_zyx[0]),
        )
        physical_xyz = labelmap.TransformContinuousIndexToPhysicalPoint(voxel_xyz)
        centroids[str(raw_label)] = {
            "label": raw_label,
            "index": _verse_index_from_label(raw_label),
            "voxel_xyz": [float(value) for value in voxel_xyz],
            "physical_xyz": [float(value) for value in physical_xyz],
            "voxel_count": int(coords_zyx.shape[0]),
            "source": "segmentation",
        }
    return centroids


def _verse_index_from_label(label: int) -> int | None:
    if 1 <= int(label) <= 25:
        return int(label) - 1
    if int(label) == 28:
        return 25
    return None
