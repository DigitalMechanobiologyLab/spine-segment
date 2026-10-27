from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
import zipfile


DEFAULT_MODEL_BUNDLE_VERSION = "v0.1.0"
DEFAULT_MODEL_BUNDLE_ASSET = "spine-segment-model-bundle-v0.1.0.zip"
DEFAULT_MODEL_BUNDLE_URL = (
    "https://github.com/wallematthias/spine-segment/releases/download/"
    f"{DEFAULT_MODEL_BUNDLE_VERSION}/{DEFAULT_MODEL_BUNDLE_ASSET}"
)
DEFAULT_MODEL_BUNDLE_FILE_HASHES = {
    "weights/spine-locator.pt": "4797348e9251eeef48bebfa454292681ad652b58cdef44faaa5b41901f7a220d",
    "weights/vertebra-locator.pt": "a0c83558833801a1610b79be31a6a8dfbbfdbb6330c15160d37927e5ff776ac5",
    "weights/vertebra-segmenter.pt": "27507ab64e743bb030be66ec440dfa611f715736ae4b6c512c996bdb980b35ad",
    "weights/process-body-segmenter.pt": "fae180b16d090b9df96e9394b6a49e3ca2b8d1a7f80c25c72c413b3644ddb309",
}

# The experimental checkpoint is packaged in the beta wheel. The release asset
# remains a verified fallback for installations built without the model data.
BETA_DETAILED_MODEL_VERSION = "dataset737-v1"
BETA_DETAILED_MODEL_ASSET = "spine-segment-beta-detailed-dataset737-v1.zip"
BETA_DETAILED_MODEL_URL = (
    "https://github.com/DigitalMechanobiologyLab/spine-segment/releases/download/"
    f"beta-detailed-dataset737-v1/{BETA_DETAILED_MODEL_ASSET}"
)
BETA_DETAILED_MODEL_FILE_HASHES = {
    "dataset.json": "d60703e230eae787936a44bacda583a80f7f917b389d3299848b4f7ab70606eb",
    "plans.json": "a89d5b50d1363523aa22083a5c5c5b5a80902cb41677692e2fd0ffb47244c304",
    "fold_0/checkpoint_best.pth": "ac1e9637d32e2c6ab03b3588480b910a9ce8e9d76d723dff3d85f07877dfc6ef",
}


@dataclass(frozen=True, slots=True)
class NativeTorchBundlePaths:
    root: Path
    spine_localization: Path
    vertebrae_localization: Path
    vertebrae_segmentation: Path
    process_body_checkpoint: Path


class ModelBundleDownloadError(RuntimeError):
    """Raised when the default model bundle cannot be downloaded or verified."""


def resolve_native_torch_bundle(root: str | Path) -> NativeTorchBundlePaths:
    bundle_root = Path(root).expanduser().resolve()
    weights_root = bundle_root / "weights"
    legacy_native_root = bundle_root / "native_torch"
    legacy_process_ckpt = (
        bundle_root
        / "nnunet"
        / "Dataset005_VerseRelabel"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_final.pth"
    )
    return NativeTorchBundlePaths(
        root=bundle_root,
        spine_localization=_preferred_existing(
            weights_root / "spine-locator.pt",
            legacy_native_root / "spine_localization.pt",
        ),
        vertebrae_localization=_preferred_existing(
            weights_root / "vertebra-locator.pt",
            legacy_native_root / "vertebrae_localization.pt",
        ),
        vertebrae_segmentation=_preferred_existing(
            weights_root / "vertebra-segmenter.pt",
            legacy_native_root / "vertebrae_segmentation.pt",
        ),
        process_body_checkpoint=_preferred_existing(
            weights_root / "process-body-segmenter.pt",
            legacy_process_ckpt,
        ),
    )


def _preferred_existing(preferred: Path, legacy: Path) -> Path:
    return preferred if preferred.exists() or not legacy.exists() else legacy


def validate_native_torch_bundle(paths: NativeTorchBundlePaths) -> None:
    missing = [
        path
        for path in (
            paths.spine_localization,
            paths.vertebrae_localization,
            paths.vertebrae_segmentation,
            paths.process_body_checkpoint,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing converted native PyTorch checkpoint(s): "
            + ", ".join(str(path) for path in missing)
        )


def default_model_bundle_cache_dir(
    *,
    version: str = DEFAULT_MODEL_BUNDLE_VERSION,
    environ: dict[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    override = env.get("SPINE_SEGMENT_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "models" / version
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "spine-segment"
    elif os.name == "nt":
        base = Path(env.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "spine-segment" / "Cache"
    else:
        base = Path(env.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spine-segment"
    return base / "models" / version


def ensure_default_native_torch_bundle(
    *,
    cache_root: str | Path | None = None,
    download_url: str = DEFAULT_MODEL_BUNDLE_URL,
    file_hashes: dict[str, str] | None = None,
    version: str = DEFAULT_MODEL_BUNDLE_VERSION,
) -> Path:
    bundle_root = (
        (Path(cache_root).expanduser().resolve() / version)
        if cache_root is not None
        else default_model_bundle_cache_dir(version=version)
    )
    paths = resolve_native_torch_bundle(bundle_root)
    try:
        validate_native_torch_bundle(paths)
        _verify_bundle_hashes(bundle_root, file_hashes or DEFAULT_MODEL_BUNDLE_FILE_HASHES)
        return bundle_root
    except (FileNotFoundError, ModelBundleDownloadError):
        pass

    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spine-segment-model-", dir=str(bundle_root.parent)) as tmp_name:
        tmp_root = Path(tmp_name)
        archive_path = tmp_root / DEFAULT_MODEL_BUNDLE_ASSET
        try:
            _download_file(download_url, archive_path)
            extracted_root = tmp_root / "extracted"
            extracted_root.mkdir()
            _extract_zip_safely(archive_path, extracted_root)
            staged_root = _find_extracted_bundle_root(extracted_root)
            _verify_bundle_hashes(staged_root, file_hashes or DEFAULT_MODEL_BUNDLE_FILE_HASHES)
            validate_native_torch_bundle(resolve_native_torch_bundle(staged_root))
        except (OSError, URLError, zipfile.BadZipFile) as exc:
            raise ModelBundleDownloadError(
                f"Unable to download model bundle from {download_url}: {exc}"
            ) from exc
        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        shutil.move(str(staged_root), str(bundle_root))
    return bundle_root


def cached_beta_detailed_model(
    *, cache_root: str | Path | None = None,
    file_hashes: dict[str, str] | None = None,
) -> Path | None:
    """Return the verified Dataset737 checkpoint in the user cache, if present."""
    root = (
        Path(cache_root).expanduser().resolve() / BETA_DETAILED_MODEL_VERSION
        if cache_root is not None
        else default_model_bundle_cache_dir(version=BETA_DETAILED_MODEL_VERSION)
    )
    try:
        _verify_bundle_hashes(root, file_hashes or BETA_DETAILED_MODEL_FILE_HASHES)
        from spine_segment.beta_subregions import _validate_model_folder
        _validate_model_folder(root)
    except (FileNotFoundError, ModelBundleDownloadError, ValueError):
        return None
    return root


def ensure_beta_detailed_model(
    *,
    cache_root: str | Path | None = None,
    download_url: str = BETA_DETAILED_MODEL_URL,
    file_hashes: dict[str, str] | None = None,
) -> Path:
    """Fetch the opt-in Dataset737 model once and verify its three source files."""
    verified = cached_beta_detailed_model(cache_root=cache_root, file_hashes=file_hashes)
    if verified is not None:
        return verified
    root = (
        Path(cache_root).expanduser().resolve() / BETA_DETAILED_MODEL_VERSION
        if cache_root is not None
        else default_model_bundle_cache_dir(version=BETA_DETAILED_MODEL_VERSION)
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spine-segment-beta-", dir=str(root.parent)) as tmp_name:
        temporary = Path(tmp_name)
        archive = temporary / BETA_DETAILED_MODEL_ASSET
        try:
            _download_file(download_url, archive)
            extracted = temporary / "extracted"
            extracted.mkdir()
            _extract_zip_safely(archive, extracted)
            staged = _find_beta_model_root(extracted)
            _verify_bundle_hashes(staged, file_hashes or BETA_DETAILED_MODEL_FILE_HASHES)
            from spine_segment.beta_subregions import _validate_model_folder
            _validate_model_folder(staged)
        except (OSError, URLError, zipfile.BadZipFile, ValueError) as exc:
            raise ModelBundleDownloadError(
                f"Unable to install beta detailed model from {download_url}: {exc}"
            ) from exc
        if root.exists():
            shutil.rmtree(root)
        shutil.move(str(staged), str(root))
    return root


def _find_beta_model_root(extracted: Path) -> Path:
    if (extracted / "dataset.json").is_file():
        return extracted
    candidates = [path for path in extracted.iterdir()
                  if path.is_dir() and (path / "dataset.json").is_file()]
    if len(candidates) == 1:
        return candidates[0]
    raise ModelBundleDownloadError("Beta model archive must contain dataset.json at its root")


def resolve_beta_detailed_model(
    *, bundle_root: str | Path | None = None, allow_download: bool = True,
) -> Path:
    """Find a checked-out model or verified cache, downloading only when allowed."""
    from spine_segment.beta_subregions import _validate_model_folder
    candidates = [Path(__file__).resolve().parent / "models" / "beta_detailed"]
    if bundle_root is not None:
        candidates.append(Path(bundle_root) / "beta_detailed")
    for candidate in candidates:
        if candidate.is_dir():
            _verify_bundle_hashes(candidate, BETA_DETAILED_MODEL_FILE_HASHES)
            _validate_model_folder(candidate)
            return candidate
    cached = cached_beta_detailed_model()
    if cached is not None:
        return cached
    if not allow_download:
        raise ModelBundleDownloadError(
            "Beta detailed model is not cached; rerun without --no-model-download"
        )
    return ensure_beta_detailed_model()


def _download_file(url: str, output_path: Path) -> None:
    with urlopen(url) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_zip_safely(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as zip_file:
        for member in zip_file.infolist():
            target = (output_dir / member.filename).resolve()
            if not target.is_relative_to(output_dir.resolve()):
                raise ModelBundleDownloadError(f"Refusing unsafe archive member: {member.filename}")
        zip_file.extractall(output_dir)


def _find_extracted_bundle_root(output_dir: Path) -> Path:
    if (output_dir / "weights").is_dir():
        return output_dir
    candidates = [path for path in output_dir.iterdir() if path.is_dir() and (path / "weights").is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    raise ModelBundleDownloadError("Downloaded model bundle does not contain a weights/ directory.")


def _verify_bundle_hashes(bundle_root: Path, file_hashes: dict[str, str]) -> None:
    for relative_path, expected_hash in file_hashes.items():
        path = bundle_root / relative_path
        if not path.exists():
            raise ModelBundleDownloadError(f"Model bundle is missing {relative_path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ModelBundleDownloadError(
                f"Model bundle checksum mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
