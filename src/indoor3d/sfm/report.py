"""Báo cáo chất lượng ước lượng camera (COLMAP) cho một cảnh.

Đọc:
  <scene>/input/                 ảnh đầu vào
  <scene>/distorted/sparse/*/    các model COLMAP trước khi khử méo (để phát hiện bị tách mảnh)
  <scene>/sparse/0/              model đã khử méo, dùng cho 3DGS

Ngưỡng đánh giá là kinh nghiệm ban đầu, sẽ hiệu chỉnh sau các lần quay thử (mốc M2).

Cách dùng:
  python -m indoor3d.sfm.report --scene /workspace/data/scenes/phong
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indoor3d.sfm.colmap_model import (
    model_exists,
    read_cameras_binary,
    read_images_binary,
    read_points3d_summary,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
REGISTRATION_PASS = 0.95
REGISTRATION_WARN = 0.80
REPROJECTION_WARN_PX = 1.5


def list_models(sparse_dir: Path) -> list[dict]:
    """Các model con trong một thư mục sparse/, kèm số ảnh đã đăng ký, xếp giảm dần."""
    models = []
    if sparse_dir.is_dir():
        for child in sorted(sparse_dir.iterdir()):
            if child.is_dir() and model_exists(child):
                models.append({"path": str(child), "registered": len(read_images_binary(child / "images.bin"))})
    return sorted(models, key=lambda m: m["registered"], reverse=True)


def analyze_model(model_dir: Path) -> dict:
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")
    points = read_points3d_summary(model_dir / "points3D.bin")
    observations = [image.num_observations for image in images.values()]
    return {
        "path": str(model_dir),
        "registered_images": len(images),
        "registered_names": sorted(image.name for image in images.values()),
        "points3d": points.count,
        "mean_reprojection_error_px": points.mean_error,
        "median_reprojection_error_px": points.median_error,
        "mean_track_length": points.mean_track_length,
        "mean_observations_per_image": (sum(observations) / len(observations)) if observations else None,
        "cameras": [
            {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params}
            for cam in cameras.values()
        ],
    }


def analyze_scene(scene_dir: Path) -> dict:
    input_dir = scene_dir / "input"
    input_names = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) \
        if input_dir.is_dir() else []
    distorted = list_models(scene_dir / "distorted" / "sparse")
    final_dir = scene_dir / "sparse" / "0"

    report: dict = {"scene": str(scene_dir), "input_images": len(input_names),
                    "distorted_models": distorted, "warnings": []}
    if model_exists(final_dir):
        model = analyze_model(final_dir)
    elif distorted:
        model = analyze_model(Path(distorted[0]["path"]))
        report["warnings"].append("undistorted model sparse/0 not found; statistics use the largest distorted model")
    else:
        report["model"] = None
        report["verdict"] = "FAIL"
        report["warnings"].append("no COLMAP reconstruction found")
        return report

    registered_names = set(model.pop("registered_names"))
    report["model"] = model
    report["unregistered_images"] = [name for name in input_names if name not in registered_names]
    ratio = model["registered_images"] / len(input_names) if input_names else None
    report["registration_ratio"] = ratio

    if ratio is None:
        verdict = "WARN"
        report["warnings"].append("input/ folder not found: registration ratio unknown")
    elif ratio >= REGISTRATION_PASS:
        verdict = "PASS"
    elif ratio >= REGISTRATION_WARN:
        verdict = "WARN"
        report["warnings"].append(f"only {ratio:.0%} of input images registered")
    else:
        verdict = "FAIL"
        report["warnings"].append(f"only {ratio:.0%} of input images registered")

    if len(distorted) > 1:
        report["warnings"].append(
            f"reconstruction split into {len(distorted)} models; only the largest is used")
        verdict = "WARN" if verdict == "PASS" else verdict
    error = model["mean_reprojection_error_px"]
    if error is not None and error > REPROJECTION_WARN_PX:
        report["warnings"].append(f"mean reprojection error {error:.2f} px > {REPROJECTION_WARN_PX} px")
        verdict = "WARN" if verdict == "PASS" else verdict
    report["verdict"] = verdict
    return report


def format_report(report: dict) -> str:
    lines = [f"scene: {report['scene']}", f"input images: {report['input_images']}"]
    model = report.get("model")
    if model:
        ratio = report.get("registration_ratio")
        lines += [
            f"registered images: {model['registered_images']}"
            + (f" ({ratio:.1%})" if ratio is not None else ""),
            f"3D points: {model['points3d']}",
            f"mean reprojection error: {model['mean_reprojection_error_px']:.3f} px"
            if model["mean_reprojection_error_px"] is not None else "mean reprojection error: -",
            f"mean track length: {model['mean_track_length']:.2f}"
            if model["mean_track_length"] is not None else "mean track length: -",
            f"models before undistortion: {len(report['distorted_models'])}",
        ]
    for warning in report["warnings"]:
        lines.append(f"WARN: {warning}")
    lines.append(f"SFM VERDICT: {report['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Báo cáo chất lượng COLMAP cho một cảnh")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="mặc định <scene>/sfm_report.json")
    args = parser.parse_args(argv)

    report = analyze_scene(args.scene)
    out = args.out or args.scene / "sfm_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(report))
    print(f"report: {out}")
    return 0 if report["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
