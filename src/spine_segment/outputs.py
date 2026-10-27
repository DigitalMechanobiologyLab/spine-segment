from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputPaths:
    vertebral_level: Path
    process_body: Path
    cort_trab: Path
    centroids: Path
    detailed_subregions: Path


def _input_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if path.suffix:
        return path.stem
    return name


def build_output_paths(input_path: str | Path, output_dir: str | Path) -> OutputPaths:
    source = Path(input_path)
    out_dir = Path(output_dir)
    stem = _input_stem(source)
    return OutputPaths(
        vertebral_level=out_dir / f"{stem}_vertebral-level.nii.gz",
        process_body=out_dir / f"{stem}_process-body.nii.gz",
        cort_trab=out_dir / f"{stem}_cort-trab.nii.gz",
        centroids=out_dir / f"{stem}_centroids.json",
        detailed_subregions=out_dir / f"{stem}_detailed-subregions-beta.nii.gz",
    )
