from __future__ import annotations

import argparse
import json
from pathlib import Path

from waveslab.cover_core import build_cover_npzs_from_sam_pipeline, json_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Build solver-ready cover NPZs from SAM sea-ice outputs.")
    parser.add_argument("--pipeline-dir", required=True, type=Path, help="Directory containing data/pipeline_arrays.npz and data/floe_catalog.csv.")
    parser.add_argument("--out", type=Path, default=Path("outputs/cover_models"))
    parser.add_argument("--direct-source", default="accepted_label_map")
    parser.add_argument("--crop-x0", type=int, default=256 * 3)
    parser.add_argument("--crop-y0", type=int, default=256)
    parser.add_argument("--crop-w", type=int, default=256 * 3)
    parser.add_argument("--crop-h", type=int, default=256 * 3)
    parser.add_argument("--smooth-width-px", type=float, default=2.0)
    parser.add_argument("--N", type=int, default=160)
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    meta = build_cover_npzs_from_sam_pipeline(
        pipeline_dir=args.pipeline_dir,
        out_dir=args.out,
        direct_source=args.direct_source,
        crop_x0=args.crop_x0,
        crop_y0=args.crop_y0,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
        smooth_width_px=args.smooth_width_px,
        N=args.N,
        M=args.M,
        write_full_binary_exports=True,
        write_preview=not args.no_preview,
    )
    summary_path = args.out / "cover_build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_ready(meta), indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
