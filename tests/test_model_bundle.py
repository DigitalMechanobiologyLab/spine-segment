from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from spine_segment.model_bundle import (
    ModelBundleDownloadError,
    ensure_beta_detailed_model,
    ensure_default_native_torch_bundle,
    resolve_native_torch_bundle,
    validate_native_torch_bundle,
)


def test_resolve_native_torch_bundle_paths(tmp_path: Path) -> None:
    bundle = resolve_native_torch_bundle(tmp_path)
    assert bundle.spine_localization == tmp_path / "weights" / "spine-locator.pt"
    assert bundle.vertebrae_localization == tmp_path / "weights" / "vertebra-locator.pt"
    assert bundle.vertebrae_segmentation == tmp_path / "weights" / "vertebra-segmenter.pt"
    assert bundle.process_body_checkpoint == tmp_path / "weights" / "process-body-segmenter.pt"


def test_resolve_native_torch_bundle_supports_legacy_paths(tmp_path: Path) -> None:
    native = tmp_path / "native_torch"
    native.mkdir(parents=True)
    (native / "spine_localization.pt").touch()
    legacy_process = (
        tmp_path
        / "nnunet"
        / "Dataset005_VerseRelabel"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
    )
    legacy_process.mkdir(parents=True)
    (legacy_process / "checkpoint_final.pth").touch()

    bundle = resolve_native_torch_bundle(tmp_path)

    assert bundle.spine_localization == native / "spine_localization.pt"
    assert bundle.process_body_checkpoint == legacy_process / "checkpoint_final.pth"


def test_validate_native_torch_bundle_requires_process_body_checkpoint(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("spine-locator.pt", "vertebra-locator.pt", "vertebra-segmenter.pt"):
        (weights / name).touch()

    with pytest.raises(FileNotFoundError, match="process-body-segmenter.pt"):
        validate_native_torch_bundle(resolve_native_torch_bundle(tmp_path))


def test_ensure_default_native_torch_bundle_downloads_and_verifies_zip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    weights = source / "weights"
    weights.mkdir(parents=True)
    expected_files = {
        "weights/spine-locator.pt": b"spine",
        "weights/vertebra-locator.pt": b"vertebra",
        "weights/vertebra-segmenter.pt": b"segmenter",
        "weights/process-body-segmenter.pt": b"process-body",
    }
    for relative_path, contents in expected_files.items():
        path = source / relative_path
        path.write_bytes(contents)
    (source / "manifest.json").write_text(
        json.dumps({"version": 1, "contents": []}),
        encoding="utf-8",
    )
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for path in source.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(source).as_posix())

    bundle_root = ensure_default_native_torch_bundle(
        cache_root=tmp_path / "cache",
        download_url=archive.as_uri(),
        file_hashes={
            relative_path: hashlib.sha256(contents).hexdigest()
            for relative_path, contents in expected_files.items()
        },
    )

    assert bundle_root == tmp_path / "cache" / "v0.1.0"
    validate_native_torch_bundle(resolve_native_torch_bundle(bundle_root))
    assert (bundle_root / "weights" / "spine-locator.pt").read_bytes() == b"spine"


def test_ensure_default_native_torch_bundle_rejects_bad_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source"
    weights = source / "weights"
    weights.mkdir(parents=True)
    for name in (
        "spine-locator.pt",
        "vertebra-locator.pt",
        "vertebra-segmenter.pt",
        "process-body-segmenter.pt",
    ):
        (weights / name).write_bytes(b"wrong")
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for path in source.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(source).as_posix())

    with pytest.raises(ModelBundleDownloadError, match="checksum"):
        ensure_default_native_torch_bundle(
            cache_root=tmp_path / "cache",
            download_url=archive.as_uri(),
            file_hashes={"weights/spine-locator.pt": "0" * 64},
        )


def test_beta_model_downloads_once_and_verifies_all_files(tmp_path: Path, monkeypatch) -> None:
    from spine_segment import model_bundle
    from spine_segment.beta_subregions import CLASS_NAMES
    source = tmp_path / "source"
    (source / "fold_0").mkdir(parents=True)
    files = {
        "dataset.json": json.dumps({"labels": {"background": 0, **{
            name: index for index, name in CLASS_NAMES.items()
        }}}).encode(),
        "plans.json": b"{}",
        "fold_0/checkpoint_best.pth": b"checkpoint",
    }
    for relative, contents in files.items():
        (source / relative).write_bytes(contents)
    archive = tmp_path / "beta.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for path in source.rglob("*"):
            if path.is_file():
                zip_file.write(path, f"model/{path.relative_to(source).as_posix()}")
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    folder = ensure_beta_detailed_model(
        cache_root=tmp_path / "cache", download_url=archive.as_uri(), file_hashes=hashes
    )
    assert (folder / "fold_0" / "checkpoint_best.pth").read_bytes() == b"checkpoint"
    monkeypatch.setattr(model_bundle, "_download_file", lambda *a: pytest.fail("downloaded twice"))
    assert ensure_beta_detailed_model(cache_root=tmp_path / "cache", file_hashes=hashes) == folder


def test_beta_model_rejects_corrupted_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "fold_0").mkdir(parents=True)
    (source / "dataset.json").write_text(json.dumps({"labels": {"background": 0}}))
    (source / "plans.json").write_text("{}")
    (source / "fold_0" / "checkpoint_best.pth").write_bytes(b"wrong")
    archive = tmp_path / "beta.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for path in source.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(source).as_posix())
    with pytest.raises(ModelBundleDownloadError, match="checksum"):
        ensure_beta_detailed_model(
            cache_root=tmp_path / "cache", download_url=archive.as_uri(),
            file_hashes={"fold_0/checkpoint_best.pth": "0" * 64},
        )
