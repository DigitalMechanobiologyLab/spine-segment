"""Create the opt-in Dataset737 release asset from the verified Git LFS model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spine_segment.beta_subregions import _validate_model_folder
from spine_segment.model_bundle import (
    BETA_DETAILED_MODEL_ASSET,
    BETA_DETAILED_MODEL_FILE_HASHES,
    _verify_bundle_hashes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build/model-bundle-pytorch/beta_detailed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    _verify_bundle_hashes(args.model_folder, BETA_DETAILED_MODEL_FILE_HASHES)
    _validate_model_folder(args.model_folder)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / BETA_DETAILED_MODEL_ASSET
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for relative in BETA_DETAILED_MODEL_FILE_HASHES:
            output.write(args.model_folder / relative, f"beta_detailed/{relative}")
        output.writestr(
            "beta_detailed/MODEL_CARD.txt",
            "Dataset737 mask-only vertebra subregions, experimental checkpoint.\n"
            "Trained on non-fractured VerSe19 vertebra masks with generated morphology labels.\n"
            "The focal-outgrowth class is a candidate region, not a CT-confirmed osteophyte.\n"
            "VerSe data: CC BY-SA 4.0; Sekuboyina et al., Medical Image Analysis 2021, "
            "doi:10.1016/j.media.2021.102166.\n"
            "VerSe source and license: https://github.com/anjany/verse\n",
        )
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
