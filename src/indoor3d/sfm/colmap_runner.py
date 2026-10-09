"""Chạy COLMAP cho một cảnh và đưa kết quả về cấu trúc mà 3DGS gốc (Inria) đọc được.

Các bước giống `convert.py` của Inria: trích đặc trưng (camera OPENCV, một camera chung)
-> so khớp -> dựng model (Mapper.ba_global_function_tolerance = 1e-6) -> khử méo về PINHOLE
-> chuyển model vào `<scene>/sparse/0`. Khác biệt có chủ đích:

  * Tự nhận tên tùy chọn GPU theo phiên bản COLMAP: bản mới dùng `FeatureExtraction.*` /
    `FeatureMatching.*`, bản cũ dùng `SiftExtraction.*` / `SiftMatching.*`. `convert.py`
    gốc chỉ biết tên cũ nên không chạy được với COLMAP mới.
  * Nếu COLMAP tách thành nhiều model, khử méo model có nhiều ảnh nhất
    (`convert.py` luôn lấy model 0).
  * Tùy chọn sequential matcher (video dài) và global mapper (COLMAP >= 4.0) cho thực nghiệm.

Đầu vào:  `<scene>/input/*.jpg`
Đầu ra:   `<scene>/images/`, `<scene>/sparse/0/`, `<scene>/distorted/` (database + model gốc),
          `<scene>/colmap_logs/*.log`, `<scene>/sfm_report.json`

Cách dùng:
  python -m indoor3d.sfm.colmap_runner --scene /workspace/data/scenes/phong \\
      --colmap /workspace/miniforge3/envs/gs-tools/bin/colmap
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from indoor3d.sfm.report import analyze_scene, format_report, list_models


def detect_gpu_option(colmap: str, command: str, candidates: tuple[str, ...]) -> str:
    """Tìm tên tùy chọn use_gpu mà phiên bản COLMAP này hỗ trợ (dựa trên `colmap <command> -h`)."""
    proc = subprocess.run([colmap, command, "-h"], capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    for option in candidates:
        if option in text:
            return option
    raise RuntimeError(f"cannot find any of {candidates} in 'colmap {command} -h' output")


def run_step(name: str, cmd: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[colmap] {name}: {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise RuntimeError(f"COLMAP step '{name}' failed (exit {proc.returncode}); see {log_path}\n"
                           + "\n".join(tail))


def run_colmap(scene: Path, colmap: str = "colmap", use_gpu: bool = True, matcher: str = "exhaustive",
               sequential_overlap: int = 20, mapper: str = "incremental",
               camera_model: str = "OPENCV", overwrite: bool = False) -> dict:
    input_dir = scene / "input"
    if not input_dir.is_dir() or not any(input_dir.iterdir()):
        raise RuntimeError(f"{input_dir} is missing or empty")
    outputs = [scene / "distorted", scene / "images", scene / "sparse", scene / "stereo"]
    if any(path.exists() for path in outputs):
        if not overwrite:
            raise RuntimeError(f"previous COLMAP outputs exist in {scene} (use --overwrite)")
        for path in outputs:
            shutil.rmtree(path, ignore_errors=True)

    distorted = scene / "distorted"
    sparse_raw = distorted / "sparse"
    sparse_raw.mkdir(parents=True)
    database = distorted / "database.db"
    logs = scene / "colmap_logs"
    gpu = "1" if use_gpu else "0"

    extract_opt = detect_gpu_option(colmap, "feature_extractor",
                                    ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"))
    match_command = "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher"
    match_opt = detect_gpu_option(colmap, match_command,
                                  ("FeatureMatching.use_gpu", "SiftMatching.use_gpu"))

    run_step("1_feature_extractor", [
        colmap, "feature_extractor", "--database_path", str(database), "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", camera_model,
        f"--{extract_opt}", gpu], logs)

    match_cmd = [colmap, match_command, "--database_path", str(database), f"--{match_opt}", gpu]
    if matcher == "sequential":
        match_cmd += ["--SequentialMatching.overlap", str(sequential_overlap)]
    run_step("2_matcher", match_cmd, logs)

    if mapper == "incremental":
        run_step("3_mapper", [
            colmap, "mapper", "--database_path", str(database), "--image_path", str(input_dir),
            "--output_path", str(sparse_raw), "--Mapper.ba_global_function_tolerance=0.000001"], logs)
    else:
        run_step("3_global_mapper", [
            colmap, "global_mapper", "--database_path", str(database), "--image_path", str(input_dir),
            "--output_path", str(sparse_raw)], logs)

    models = list_models(sparse_raw)
    if not models:
        raise RuntimeError("COLMAP produced no model (no images could be registered)")
    chosen = Path(models[0]["path"])
    run_step("4_image_undistorter", [
        colmap, "image_undistorter", "--image_path", str(input_dir), "--input_path", str(chosen),
        "--output_path", str(scene), "--output_type", "COLMAP"], logs)

    # Same layout as Inria convert.py: move undistorted model files into sparse/0.
    final_dir = scene / "sparse" / "0"
    final_dir.mkdir(parents=True, exist_ok=True)
    for item in (scene / "sparse").iterdir():
        if item.is_file():
            shutil.move(str(item), str(final_dir / item.name))

    report = analyze_scene(scene)
    report["colmap_settings"] = {
        "colmap": colmap, "use_gpu": use_gpu, "matcher": matcher, "mapper": mapper,
        "camera_model": camera_model, "extract_gpu_option": extract_opt,
        "match_gpu_option": match_opt, "undistorted_model": str(chosen),
        "sequential_overlap": sequential_overlap if matcher == "sequential" else None,
    }
    (scene / "sfm_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy COLMAP cho một cảnh (tương đương convert.py của Inria)")
    parser.add_argument("--scene", type=Path, required=True, help="thư mục cảnh chứa input/")
    parser.add_argument("--colmap", default="colmap", help="đường dẫn chương trình colmap")
    parser.add_argument("--no-gpu", action="store_true", help="chạy SIFT trên CPU")
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--mapper", choices=("incremental", "global"), default="incremental")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--overwrite", action="store_true", help="xóa kết quả COLMAP cũ của cảnh")
    args = parser.parse_args(argv)

    try:
        report = run_colmap(args.scene, colmap=args.colmap, use_gpu=not args.no_gpu, matcher=args.matcher,
                            sequential_overlap=args.sequential_overlap, mapper=args.mapper,
                            camera_model=args.camera_model, overwrite=args.overwrite)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    print(f"report: {args.scene / 'sfm_report.json'}")
    return 0 if report["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
