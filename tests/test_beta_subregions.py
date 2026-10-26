import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from spine_segment import cli
from spine_segment import beta_subregions
from spine_segment.beta_subregions import assemble_predictions, prepare_vertebra_crops


def _levels() -> sitk.Image:
    array = np.zeros((20, 20, 20), np.uint8)
    array[2:7, 3:8, 4:9] = 20
    array[11:17, 10:16, 10:16] = 21
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 1.1, 2.0))
    image.SetOrigin((10.0, -20.0, 30.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image


def test_beta_crops_are_binary_and_preserve_voxel_geometry():
    image = _levels()
    crops = prepare_vertebra_crops(image, margin_mm=2.0)
    assert set(crops) == {20, 21}
    for label, crop in crops.items():
        assert set(np.unique(sitk.GetArrayFromImage(crop))) <= {0, 1}
        assert crop.GetSpacing() == image.GetSpacing()
        assert crop.GetDirection() == image.GetDirection()
        assert np.count_nonzero(sitk.GetArrayFromImage(crop)) == np.count_nonzero(
            sitk.GetArrayFromImage(image) == label
        )


def test_beta_assembly_clips_exterior_and_fills_only_from_body_process():
    image = _levels()
    crops = prepare_vertebra_crops(image, margin_mm=2.0)
    predictions = {}
    for label, crop in crops.items():
        array = np.full(sitk.GetArrayFromImage(crop).shape, 4, np.uint8)
        owner = sitk.GetArrayFromImage(crop) != 0
        array[owner] = 1
        first = tuple(np.argwhere(owner)[0])
        array[owner & (np.indices(array.shape)[0] == np.argwhere(owner)[0, 0])] = 2
        array[first] = 0
        prediction = sitk.GetImageFromArray(array)
        prediction.CopyInformation(crop)
        predictions[label] = prediction
    output, report = assemble_predictions(image, predictions)
    source = sitk.GetArrayFromImage(image)
    result = sitk.GetArrayFromImage(output)
    assert np.array_equal(result > 0, source > 0)
    assert not np.any(result == 4)
    assert report[20]["clipped_outside_bone_voxels"] > 0
    assert report[20]["filled_inside_bone_voxels"] == 1
    assert output.GetOrigin() == image.GetOrigin()


def test_beta_rejects_unknown_class_and_missing_vertebra():
    image = _levels()
    crops = prepare_vertebra_crops(image)
    with pytest.raises(ValueError, match="match"):
        assemble_predictions(image, {20: crops[20]})
    bad = sitk.Image(crops[20])
    bad = bad * 11
    with pytest.raises(ValueError, match="unknown"):
        assemble_predictions(image, {20: bad, 21: crops[21]})


def test_beta_cli_rejects_incompatible_modes_before_model_load(tmp_path, monkeypatch):
    source = tmp_path / "scan.nii.gz"
    source.touch()
    monkeypatch.setattr(cli, "create_native_backend", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), "--output", str(tmp_path), "--level-only",
                  "--beta-detailed-model", str(tmp_path)])
    assert exc.value.code == 2


def test_beta_cli_resolves_shipped_model_without_folder(tmp_path, monkeypatch):
    source = tmp_path / "scan.nii.gz"
    source.touch()
    model = tmp_path / "beta_model"
    (model / "fold_0").mkdir(parents=True)
    (model / "fold_0" / "checkpoint_best.pth").touch()
    (model / "plans.json").write_text("{}")
    (model / "dataset.json").write_text(json.dumps({
        "labels": {"background": 0, **{name: value for value, name in
                                    beta_subregions.CLASS_NAMES.items()}}
    }))
    monkeypatch.setattr(cli, "create_native_backend", lambda *a, **k: object())
    monkeypatch.setattr(cli, "resolve_beta_detailed_model", lambda **k: model)
    def execute(**kwargs):
        assert kwargs["beta_detailed_model"] == model
        return []
    monkeypatch.setattr(cli, "segment_files", execute)
    assert cli.main([str(source), "--output", str(tmp_path / "out"),
                     "--beta-detailed-labels"]) == 0


def test_beta_cli_no_download_requires_cached_or_local_model(tmp_path, monkeypatch):
    source = tmp_path / "scan.nii.gz"
    source.touch()
    monkeypatch.setattr(cli, "create_native_backend", lambda *a, **k: object())
    from spine_segment.model_bundle import ModelBundleDownloadError
    def resolve(**kwargs):
        assert kwargs["allow_download"] is False
        raise ModelBundleDownloadError("not cached")
    monkeypatch.setattr(cli, "resolve_beta_detailed_model", resolve)
    assert cli.main([str(source), "--output", str(tmp_path / "out"),
                     "--beta-detailed-labels", "--no-model-download"]) == 2


def test_beta_inference_uses_mask_crops_and_reassembles(tmp_path, monkeypatch):
    model = tmp_path / "model"
    (model / "fold_0").mkdir(parents=True)
    (model / "fold_0" / "checkpoint_best.pth").touch()
    (model / "plans.json").write_text("{}")
    (model / "dataset.json").write_text(json.dumps({
        "labels": {"background": 0, **{name: value for value, name in
                                    beta_subregions.CLASS_NAMES.items()}}
    }))

    class FakePredictor:
        def predict_from_files_sequential(self, input_dir, output_dir, *args):
            for path in Path(input_dir).glob("*_0000.nii.gz"):
                image = sitk.ReadImage(str(path))
                values = sitk.GetArrayFromImage(image)
                assert set(np.unique(values)) <= {0, 1}
                prediction = sitk.GetImageFromArray(values)
                prediction.CopyInformation(image)
                sitk.WriteImage(prediction, str(Path(output_dir) / path.name.replace("_0000", "")))

    monkeypatch.setattr(beta_subregions, "_make_predictor", lambda *a: FakePredictor())
    output, report = beta_subregions.predict_detailed_subregions(_levels(), model, device="cpu")
    assert np.array_equal(sitk.GetArrayFromImage(output) > 0,
                          sitk.GetArrayFromImage(_levels()) > 0)
    assert set(report) == {20, 21}
