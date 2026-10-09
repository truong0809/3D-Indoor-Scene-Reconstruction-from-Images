"""Trích và chọn frame từ video (hoặc thư mục ảnh) để đưa vào ước lượng camera.

Cách làm:
  1. Trích frame ứng viên bằng ffmpeg với tốc độ `--fps` (ffmpeg tự xoay theo metadata video).
  2. Chấm độ nét mỗi frame bằng phương sai Laplacian (ảnh xám, thu nhỏ cạnh dài về 640 px).
  3. Chia dãy frame thành `--target` đoạn đều nhau theo thời gian, lấy frame nét nhất mỗi đoạn.
  4. Bỏ frame được chọn nếu độ nét < `--blur-ratio` x trung vị độ nét của cả video.
  5. Ghi frame đã chọn vào `<scene-dir>/input/` (đúng cấu trúc COLMAP/3DGS mong đợi)
     và báo cáo vào `<scene-dir>/frames_report.json`.

Cách dùng:
  python -m indoor3d.data.frames --video phong.mp4 --scene-dir /workspace/data/scenes/phong
  python -m indoor3d.data.frames --images thu_muc_anh --scene-dir /workspace/data/scenes/phong
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def sharpness(path: Path, long_side: int = 640) -> float:
    """Độ nét: phương sai của Laplacian trên ảnh xám đã thu nhỏ (càng lớn càng nét)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image {path}")
    height, width = gray.shape
    scale = long_side / max(height, width)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(video: Path, out_dir: Path, fps: float, max_side: int = 0,
                   ffmpeg: str = "ffmpeg") -> list[Path]:
    """Trích frame JPEG từ video vào out_dir; trả về danh sách file theo thứ tự thời gian."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = [f"fps={fps}"]
    if max_side > 0:
        # Giới hạn cạnh dài, giữ tỉ lệ, kích thước chẵn.
        filters.append(
            f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
        )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vf", ",".join(filters), "-q:v", "2", str(out_dir / "frame_%06d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def select_frames(scores: list[float], target: int, blur_ratio: float) -> tuple[list[int], dict]:
    """Chọn frame nét nhất trong mỗi đoạn; bỏ frame quá mờ so với trung vị.

    Trả về (chỉ số các frame được chọn theo thứ tự, thống kê).
    """
    count = len(scores)
    if count == 0:
        return [], {"candidates": 0, "windows": 0, "selected": 0, "rejected_blurry": 0}
    values = np.asarray(scores, dtype=float)
    median = float(np.median(values))
    threshold = blur_ratio * median
    windows = max(1, min(target, count))
    edges = np.linspace(0, count, windows + 1).round().astype(int)

    selected: list[int] = []
    rejected = 0
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        best = int(start + np.argmax(values[start:end]))
        if values[best] >= threshold:
            selected.append(best)
        else:
            rejected += 1
    stats = {
        "candidates": count,
        "windows": windows,
        "selected": len(selected),
        "rejected_blurry": rejected,
        "blur_ratio": blur_ratio,
        "blur_threshold": threshold,
        "sharpness_min": float(values.min()),
        "sharpness_median": median,
        "sharpness_max": float(values.max()),
    }
    return selected, stats


def prepare_from_video(video: Path, scene_dir: Path, fps: float, target: int, blur_ratio: float,
                       max_side: int = 0, keep_candidates: bool = False,
                       ffmpeg: str = "ffmpeg") -> dict:
    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    work = scene_dir / "frames_candidates" if keep_candidates else Path(tempfile.mkdtemp(prefix="frames_"))
    try:
        candidates = extract_frames(video, work, fps, max_side=max_side, ffmpeg=ffmpeg)
        if not candidates:
            raise RuntimeError(f"ffmpeg extracted no frames from {video}")
        scores = [sharpness(path) for path in candidates]
        selected, stats = select_frames(scores, target, blur_ratio)
        records = []
        for order, index in enumerate(selected, start=1):
            name = f"{order:05d}.jpg"
            shutil.copy2(candidates[index], input_dir / name)
            records.append({"name": name, "candidate_index": index,
                            "time_s": round(index / fps, 3), "sharpness": scores[index]})
    finally:
        if not keep_candidates:
            shutil.rmtree(work, ignore_errors=True)
    return {"source": str(video), "source_type": "video", "fps": fps, "max_side": max_side,
            **stats, "frames": records}


def prepare_from_images(images_dir: Path, scene_dir: Path, target: int, blur_ratio: float) -> dict:
    """Dùng ảnh chụp sẵn: đọc bằng OpenCV (áp dụng xoay EXIF), chọn ảnh nét, ghi JPEG vào input/."""
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"no .jpg/.jpeg/.png images in {images_dir}")
    scores = [sharpness(path) for path in paths]
    selected, stats = select_frames(scores, target, blur_ratio)
    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for order, index in enumerate(selected, start=1):
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        name = f"{order:05d}.jpg"
        if image is None or not cv2.imwrite(str(input_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"cannot convert {paths[index]}")
        records.append({"name": name, "source_name": paths[index].name, "sharpness": scores[index]})
    return {"source": str(images_dir), "source_type": "images", **stats, "frames": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trích và chọn frame nét cho COLMAP/3DGS")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="file video đầu vào")
    source.add_argument("--images", type=Path, help="thư mục ảnh đầu vào")
    parser.add_argument("--scene-dir", type=Path, required=True, help="thư mục cảnh; frame ghi vào <scene-dir>/input")
    parser.add_argument("--fps", type=float, default=4.0, help="tốc độ trích frame ứng viên (mặc định 4)")
    parser.add_argument("--target", type=int, default=250, help="số đoạn chia = số frame tối đa được chọn (mặc định 250)")
    parser.add_argument("--blur-ratio", type=float, default=0.4,
                        help="bỏ frame có độ nét < tỉ lệ này x trung vị (mặc định 0.4)")
    parser.add_argument("--max-side", type=int, default=0, help="giới hạn cạnh dài khi trích (0 = giữ nguyên)")
    parser.add_argument("--keep-candidates", action="store_true", help="giữ lại toàn bộ frame ứng viên")
    parser.add_argument("--overwrite", action="store_true", help="cho phép ghi vào input/ đã có ảnh")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="đường dẫn ffmpeg")
    args = parser.parse_args(argv)

    input_dir = args.scene_dir / "input"
    if input_dir.is_dir() and any(input_dir.iterdir()):
        if not args.overwrite:
            parser.error(f"{input_dir} already contains files (use --overwrite to replace them)")
        shutil.rmtree(input_dir)

    if args.video:
        report = prepare_from_video(args.video, args.scene_dir, args.fps, args.target, args.blur_ratio,
                                    max_side=args.max_side, keep_candidates=args.keep_candidates,
                                    ffmpeg=args.ffmpeg)
    else:
        report = prepare_from_images(args.images, args.scene_dir, args.target, args.blur_ratio)

    report_path = args.scene_dir / "frames_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"candidates: {report['candidates']} | selected: {report['selected']} "
          f"| rejected (blurry): {report['rejected_blurry']} "
          f"| sharpness median: {report['sharpness_median']:.1f}")
    print(f"frames: {input_dir}\nreport: {report_path}")
    return 0 if report["selected"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
