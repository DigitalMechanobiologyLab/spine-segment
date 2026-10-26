from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from spine_segment import api
from spine_segment.api import segment_file
from spine_segment.backend import LocalizationResult, SegmentationResult
from spine_segment.labels import BODY_LABEL, PROCESS_LABEL


class _StubBackend:
    def __init__(self) -> None:
        self.localization_calls = 0
        self.level_centroids: dict[str, dict[str, object]] | None = None

    def localize(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
    ) -> LocalizationResult:
        self.localization_calls += 1
        return LocalizationResult(
            centroids={
                "20": {
                    "label": 20,
                    "index": 19,
                    "voxel_xyz": [2.0, 3.0, 4.0],
                    "physical_xyz": list(image.TransformContinuousIndexToPhysicalPoint((2.0, 3.0, 4.0))),
                    "score": 0.9,
                }
            },
            metadata={"backend": "stub", "device_seen": device},
        )

    def segment(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
        level_only: bool = False,
    ) -> SegmentationResult:
        shape = sitk.GetArrayFromImage(image).shape
        metadata = {
            "backend": "stub",
            "device_seen": device,
            "spacing_seen": image.GetSpacing(),
            "size_seen": image.GetSize(),
        }

        vertebral_arr = np.zeros(shape, dtype=np.uint8)
        vertebral_arr[1:4, 1:4, 1:4] = 20
        vertebral = sitk.GetImageFromArray(vertebral_arr)
        vertebral.CopyInformation(image)

        if level_only:
            return SegmentationResult(
                vertebral_level=vertebral,
                metadata={**metadata, "level_only_seen": True},
            )

        process_body_arr = np.zeros(shape, dtype=np.uint8)
        process_body_arr[1:4, 1:4, 1:4] = BODY_LABEL
        process_body_arr[1, 1:4, 1:4] = PROCESS_LABEL
        process_body = sitk.GetImageFromArray(process_body_arr)
        process_body.CopyInformation(image)

        return SegmentationResult(
            vertebral_level=vertebral,
            process_body=process_body,
            metadata=metadata,
        )

    def segment_levels(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
        centroids: dict[str, dict[str, object]],
    ) -> SegmentationResult:
        self.level_centroids = centroids
        return self.segment(
            image=image,
            source_path=source_path,
            device=device,
            level_only=True,
        )


def _write_centroid_artifact(path: Path, image: sitk.Image) -> None:
    voxel_xyz = (2.0, 3.0, 4.0)
    path.write_text(
        json.dumps(
            {
                "input": "localization-source.nii.gz",
                "coordinate_system": "voxel_xyz",
                "centroids": {
                    "20": {
                        "label": 20,
                        "index": 19,
                        "voxel_xyz": list(voxel_xyz),
                        "physical_xyz": list(
                            image.TransformContinuousIndexToPhysicalPoint(voxel_xyz)
                        ),
                        "score": 0.9,
                        "schema_tag": "preserve-me",
                    }
                },
                "metadata": {"backend": "localizer", "schema_version": 1},
            }
        ),
        encoding="utf-8",
    )


def test_segment_file_writes_default_outputs(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.full((6, 6, 6), 200.0, dtype=np.float32))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=_StubBackend(),
        device="cpu",
        overwrite=True,
    )

    assert result.output_paths.vertebral_level.exists()
    assert result.output_paths.process_body.exists()
    assert result.output_paths.cort_trab.exists()
    assert result.output_paths.centroids.exists()
    assert not result.output_paths.detailed_subregions.exists()
    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))
    assert payload["centroids"]["20"]["voxel_xyz"] == [2.0, 2.0, 2.0]
    assert payload["centroids"]["20"]["voxel_count"] == 27
    assert payload["centroids"]["20"]["source"] == "segmentation"
    assert not list((tmp_path / "out").glob(".spine-segment-tmp-*"))


def test_segment_file_beta_writes_detailed_labels_and_provenance(tmp_path, monkeypatch):
    from spine_segment import beta_subregions
    image = sitk.GetImageFromArray(np.full((6, 6, 6), 200.0, np.float32))
    source = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(source))

    def fake_predict(levels, model_folder, *, device):
        assert model_folder == "model"
        assert device == "cpu"
        detail = sitk.GetImageFromArray((sitk.GetArrayFromImage(levels) > 0).astype(np.uint8))
        detail.CopyInformation(levels)
        return detail, {20: {"owner_voxels": 27}}

    monkeypatch.setattr(beta_subregions, "predict_detailed_subregions", fake_predict)
    result = segment_file(source, tmp_path / "out", backend=_StubBackend(),
                          device="cpu", beta_detailed_model="model")
    assert result.output_paths.detailed_subregions.exists()
    payload = json.loads(result.output_paths.centroids.read_text())
    assert payload["metadata"]["beta_detailed_subregions"]["schema"] == "dataset737"
    assert payload["metadata"]["beta_detailed_subregions"]["anatomical_validation"] is False


def test_beta_prediction_failure_does_not_leave_standard_outputs(tmp_path, monkeypatch):
    from spine_segment import beta_subregions
    source = tmp_path / "case.nii.gz"
    sitk.WriteImage(sitk.Image([6, 6, 6], sitk.sitkFloat32), str(source))

    def fail_predict(*args, **kwargs):
        raise RuntimeError("predictor unavailable")

    monkeypatch.setattr(beta_subregions, "predict_detailed_subregions", fail_predict)
    with pytest.raises(RuntimeError, match="unavailable"):
        segment_file(source, tmp_path / "out", backend=_StubBackend(),
                     device="cpu", beta_detailed_model="model")
    assert not list((tmp_path / "out").glob("*.nii.gz"))


def test_segment_file_level_only_writes_only_vertebral_level(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.full((6, 6, 6), 200.0, dtype=np.float32))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=_StubBackend(),
        device="cpu",
        overwrite=True,
        level_only=True,
    )

    assert result.output_paths.vertebral_level.exists()
    assert result.output_paths.centroids.exists()
    assert not result.output_paths.process_body.exists()
    assert not result.output_paths.cort_trab.exists()
    assert result.metadata["level_only_seen"] is True
    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))
    assert payload["centroids"]["20"]["source"] == "segmentation"


def test_segment_file_localization_only_writes_centroid_json(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.full((8, 8, 8), 200.0, dtype=np.float32))
    image.SetSpacing((0.5, 0.5, 2.0))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=_StubBackend(),
        device="cpu",
        overwrite=True,
        localization_only=True,
    )
    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))

    assert result.output_paths.centroids.exists()
    assert not result.output_paths.vertebral_level.exists()
    assert payload["centroids"]["20"]["voxel_xyz"] == [2.0, 3.0, 4.0]
    assert payload["metadata"]["localization_only"] is True


def test_segment_file_rejects_level_only_with_localization_only_before_io(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="level_only and localization_only are mutually exclusive",
    ):
        segment_file(
            input_path=tmp_path / "does-not-exist.nii.gz",
            output_dir=tmp_path / "out",
            backend=_StubBackend(),
            level_only=True,
            localization_only=True,
        )


def test_segment_file_level_only_consumes_centroids_once_without_localization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = sitk.GetImageFromArray(np.full((8, 8, 8), 200.0, dtype=np.float32))
    image.SetSpacing((0.5, 0.75, 2.0))
    image.SetOrigin((-10.0, 5.0, 20.0))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)
    centroids_path = tmp_path / "localized.json"
    _write_centroid_artifact(centroids_path, image)
    backend = _StubBackend()
    read_count = 0
    original_read_text = Path.read_text

    def count_centroid_reads(path: Path, *args, **kwargs) -> str:
        nonlocal read_count
        if path == centroids_path:
            read_count += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_centroid_reads)

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=backend,
        device="cpu",
        overwrite=True,
        level_only=True,
        centroids_path=centroids_path,
    )

    assert read_count == 1
    assert backend.localization_calls == 0
    assert backend.level_centroids is not None
    assert backend.level_centroids["20"]["schema_tag"] == "preserve-me"
    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))
    assert payload["centroids"]["20"]["index"] == 19
    assert payload["centroids"]["20"]["score"] == 0.9
    assert payload["centroids"]["20"]["schema_tag"] == "preserve-me"
    assert payload["centroids"]["20"]["source"] == "segmentation"
    assert payload["centroids"]["20"]["segmentation_status"] == "segmented"
    assert payload["metadata"]["input_centroids_metadata"] == {
        "backend": "localizer",
        "schema_version": 1,
    }


def test_segment_file_preserves_supplied_centroid_missing_from_segmentation(
    tmp_path: Path,
) -> None:
    image = sitk.GetImageFromArray(np.full((8, 8, 8), 200.0, dtype=np.float32))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)
    centroids_path = tmp_path / "localized.json"
    _write_centroid_artifact(centroids_path, image)
    artifact = json.loads(centroids_path.read_text(encoding="utf-8"))
    artifact["centroids"]["21"] = {
        "label": 21,
        "index": 20,
        "voxel_xyz": [5.0, 5.0, 5.0],
        "physical_xyz": [5.0, 5.0, 5.0],
        "score": 0.4,
        "schema_tag": "missing-but-preserved",
    }
    centroids_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=_StubBackend(),
        device="cpu",
        overwrite=True,
        level_only=True,
        centroids_path=centroids_path,
    )

    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))
    assert payload["centroids"]["21"] == {
        "label": 21,
        "index": 20,
        "voxel_xyz": [5.0, 5.0, 5.0],
        "physical_xyz": [5.0, 5.0, 5.0],
        "score": 0.4,
        "schema_tag": "missing-but-preserved",
        "segmentation_status": "missing",
    }


def test_load_centroid_artifact_rejects_coordinates_outside_ct(tmp_path: Path) -> None:
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    path = tmp_path / "outside.json"
    _write_centroid_artifact(path, image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["centroids"]["20"]["voxel_xyz"] = [2.0, 3.0, 8.0]
    payload["centroids"]["20"]["physical_xyz"] = list(
        image.TransformContinuousIndexToPhysicalPoint((2.0, 3.0, 8.0))
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the input CT"):
        api.load_centroid_artifact(path, image=image)


def test_load_centroid_artifact_rejects_physical_geometry_mismatch(tmp_path: Path) -> None:
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    path = tmp_path / "mismatch.json"
    _write_centroid_artifact(path, image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["centroids"]["20"]["physical_xyz"] = [200.0, 300.0, 400.0]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the input CT geometry"):
        api.load_centroid_artifact(path, image=image)


@pytest.mark.parametrize(
    ("level_only", "localization_only", "message"),
    [
        (False, False, "requires level_only=True"),
        (True, True, "level_only and localization_only are mutually exclusive"),
    ],
)
def test_segment_file_rejects_invalid_centroid_mode_combinations(
    tmp_path: Path,
    level_only: bool,
    localization_only: bool,
    message: str,
) -> None:
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path))
    centroids_path = tmp_path / "centroids.json"
    _write_centroid_artifact(centroids_path, image)

    with pytest.raises(ValueError, match=message):
        segment_file(
            input_path=input_path,
            output_dir=tmp_path / "out",
            backend=_StubBackend(),
            device="cpu",
            level_only=level_only,
            localization_only=localization_only,
            centroids_path=centroids_path,
        )
