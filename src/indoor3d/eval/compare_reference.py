"""So sánh kết quả 3DGS tự chạy với số liệu tham chiếu đã công bố.

Mỗi thư mục mô hình (output của train.py / render.py / metrics.py của Inria) cần có:

    results.json      chỉ số do metrics.py ghi, theo từng "ours_<iteration>"
    run_meta.json     thông tin lượt chạy (thời gian huấn luyện, GPU, commit) do script chạy ghi
    gpu_log.csv       nhật ký nvidia-smi trong lúc huấn luyện, để lấy VRAM đỉnh (không bắt buộc)
    point_cloud/iteration_<iteration>/point_cloud.ply   để lấy số Gaussian và dung lượng

Cách dùng:

    python -m indoor3d.eval.compare_reference \\
        --config configs/baseline_deepblending.json \\
        --model-root /workspace/outputs/baseline_inria/db \\
        --out summary.json

Kết luận PASS chỉ khi mọi cảnh có kết quả và trung bình mỗi chỉ số lệch so với tham chiếu
không vượt ngưỡng trong file cấu hình.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

METRICS = ("PSNR", "SSIM", "LPIPS")


def read_ply_vertex_count(path: Path) -> int | None:
    """Đọc số đỉnh (số Gaussian) từ header của file PLY; None nếu không đọc được."""
    try:
        with open(path, "rb") as handle:
            for _ in range(256):
                raw = handle.readline()
                if not raw:
                    break
                line = raw.decode("ascii", errors="replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[2])
                if line == "end_header":
                    break
    except (OSError, ValueError, IndexError):
        return None
    return None


def peak_gpu_memory_mib(csv_path: Path) -> float | None:
    """VRAM đỉnh (MiB) từ log `nvidia-smi --query-gpu=timestamp,memory.used,... --format=csv,noheader,nounits`."""
    peak = None
    try:
        with open(csv_path, encoding="utf-8", errors="replace") as handle:
            for row in handle:
                parts = [part.strip() for part in row.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    used = float(parts[1])
                except ValueError:
                    continue
                peak = used if peak is None else max(peak, used)
    except OSError:
        return None
    return peak


@dataclass
class SceneResult:
    scene: str
    model_dir: str
    status: str = "MISSING"  # OK | MISSING
    metrics: dict = field(default_factory=dict)
    train_seconds: float | None = None
    peak_vram_mib: float | None = None
    num_gaussians: int | None = None
    ply_size_mb: float | None = None
    run_meta: dict = field(default_factory=dict)


def load_scene(scene: str, model_dir: Path, method: str, iteration: int) -> SceneResult:
    result = SceneResult(scene=scene, model_dir=str(model_dir))

    results_path = model_dir / "results.json"
    if results_path.is_file():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        values = data.get(method, {})
        if all(name in values for name in METRICS):
            result.metrics = {name: float(values[name]) for name in METRICS}
            result.status = "OK"

    meta_path = model_dir / "run_meta.json"
    if meta_path.is_file():
        result.run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seconds = result.run_meta.get("train_seconds")
        result.train_seconds = float(seconds) if seconds is not None else None

    result.peak_vram_mib = peak_gpu_memory_mib(model_dir / "gpu_log.csv")

    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if ply_path.is_file():
        result.num_gaussians = read_ply_vertex_count(ply_path)
        result.ply_size_mb = round(ply_path.stat().st_size / 2**20, 2)
    return result


def compare(scenes: list[SceneResult], reference: dict, tolerance: dict) -> dict:
    """Tính trung bình các cảnh và so với tham chiếu. Trả về dict tóm tắt."""
    summary: dict = {
        "mean": {},
        "reference": reference,
        "tolerance": tolerance,
        "delta": {},
        "within_tolerance": {},
    }
    if not scenes or any(scene.status != "OK" for scene in scenes):
        summary["verdict"] = "INCOMPLETE"
        return summary

    for name in METRICS:
        value = sum(scene.metrics[name] for scene in scenes) / len(scenes)
        delta = value - reference[name]
        summary["mean"][name] = value
        summary["delta"][name] = delta
        summary["within_tolerance"][name] = abs(delta) <= tolerance[name]
    summary["verdict"] = "PASS" if all(summary["within_tolerance"].values()) else "FAIL"
    return summary


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def format_table(scenes: list[SceneResult], summary: dict) -> str:
    lines = [
        "| Cảnh | PSNR | SSIM | LPIPS | Train (phút) | VRAM đỉnh (GiB) | Số Gaussian |",
        "|---|---|---|---|---|---|---|",
    ]
    for scene in scenes:
        minutes = None if scene.train_seconds is None else scene.train_seconds / 60
        vram = None if scene.peak_vram_mib is None else scene.peak_vram_mib / 1024
        gaussians = "-" if scene.num_gaussians is None else f"{scene.num_gaussians:,}"
        if scene.status == "OK":
            m = scene.metrics
            lines.append(f"| {scene.scene} | {_fmt(m['PSNR'])} | {_fmt(m['SSIM'])} | {_fmt(m['LPIPS'])} "
                         f"| {_fmt(minutes, 1)} | {_fmt(vram, 1)} | {gaussians} |")
        else:
            lines.append(f"| {scene.scene} | THIẾU KẾT QUẢ | | | {_fmt(minutes, 1)} | {_fmt(vram, 1)} | {gaussians} |")
    if summary.get("mean"):
        mean, ref, delta = summary["mean"], summary["reference"], summary["delta"]
        lines.append(f"| **Trung bình** | {_fmt(mean['PSNR'])} | {_fmt(mean['SSIM'])} | {_fmt(mean['LPIPS'])} | | | |")
        lines.append(f"| Tham chiếu | {_fmt(ref['PSNR'])} | {_fmt(ref['SSIM'])} | {_fmt(ref['LPIPS'])} | | | |")
        lines.append(f"| Chênh lệch | {delta['PSNR']:+.3f} | {delta['SSIM']:+.3f} | {delta['LPIPS']:+.3f} | | | |")
    lines.append("")
    lines.append(f"Kết luận: **{summary['verdict']}**")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path,
                        help="thư mục chứa một thư mục mô hình cho mỗi cảnh")
    parser.add_argument("--out", type=Path, help="ghi tóm tắt JSON vào file này")
    parser.add_argument("--strict", action="store_true", help="trả mã 1 nếu kết luận không phải PASS")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    method = config.get("method", "ours_30000")
    iteration = int(config.get("iteration", 30000))
    scenes = [load_scene(name, args.model_root / name, method, iteration) for name in config["scenes"]]
    summary = compare(scenes, config["reference"]["metrics"], config["tolerance"])

    report = {
        "config": config.get("name"),
        "method": method,
        "reference_source": config["reference"].get("source"),
        "scenes": [asdict(scene) for scene in scenes],
        **summary,
    }
    print(format_table(scenes, summary))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nĐã ghi tóm tắt: {args.out}")
    if args.strict and summary["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
