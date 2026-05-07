from __future__ import annotations

import argparse
from pathlib import Path

import waveslab.imaging.sea_ice_segmentation as seg


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_image = repo_root / "data" / "sample" / "sea_ice" / "17-07-04_10-00-53.bmp"

    parser = argparse.ArgumentParser(description="Run the fair sea-ice segmentation comparison.")
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument("--sam-weights", type=Path, required=True, help="Path to SAM checkpoint, e.g. sam_vit_h_4b8939.pth.")
    parser.add_argument("--out", type=Path, default=Path("outputs/sea_ice_segmentation"))
    parser.add_argument("--no-orthorectify", action="store_true", help="Use this if --image is already rectified.")
    args = parser.parse_args()

    seg.RAW_IMAGE_PATH = args.image.expanduser().resolve()
    seg.SAM_WEIGHTS_PATH = args.sam_weights.expanduser().resolve()
    seg.OUT_DIR = args.out.expanduser().resolve()
    seg.USE_ORTHORECTIFICATION = not args.no_orthorectify
    seg.main()


if __name__ == "__main__":
    main()
