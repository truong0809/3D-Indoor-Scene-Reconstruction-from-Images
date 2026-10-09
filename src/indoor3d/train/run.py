"""Huấn luyện một cảnh bằng gsplat theo một chế độ trong configs/train_modes.json, rồi đánh giá và tóm tắt.

Chạy bằng Python của env gs-gsplat (trên pod có GPU):

    python -m indoor3d.train.run --mode default --scene /workspace/data/scenes/phong01 \\
        --out /workspace/outputs/modes/phong01/default
    python -m indoor3d.train.run --mode mcmc --scene /workspace/data/scenes/phong01 \\
        --out /workspace/outputs/modes/phong01/mcmc --cap-max-from /workspace/outputs/modes/phong01/default

Các bước: kiểm tra cảnh → chia train/test (split.json) → huấn luyện qua gsplat_launcher → đánh giá
checkpoint cuối (lệnh riêng, như benchmark của gsplat) → summary.json. Lệnh đã chạy, commit, GPU, thời gian
và mọi sai khác so với protocol được ghi vào run_meta.json.

`--dry-run` chỉ in kế hoạch (không ghi file, không cần GPU). Một thư mục kết quả đã có lượt chạy OK chỉ được
bỏ qua khi cấu hình giống hệt yêu cầu mới; lượt dở dang được chạy lại với `--rerun-incomplete`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from indoor3d.eval.gsplat_results import load_run, load_summary
from indoor3d.sfm.colmap_model import model_exists, read_cameras_binary, read_images_binary
from indoor3d.train.profiles import (
    eval_args,
    load_config,
    parse_assignments,
    resolve_cap_max,
    resolve_mode,
    trainer_args,
)
from indoor3d.train.split import make_split, make_train_view, write_split

REPO_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_DIR / "configs" / "train_modes.json"
UNDISTORTED_MODELS = ("PINHOLE", "SIMPLE_PINHOLE")
INRIA_MAX_WIDTH = 1600  # train.py của Inria tự thu nhỏ ảnh rộng hơn 1600 px (khi --resolution mặc định)
PROGRESS_INTERVAL = 30.0  # giây giữa hai dòng tiến độ tqdm được ghi vào log


class SceneError(RuntimeError):
    pass


def inspect_scene(scene_dir: Path) -> dict:
    """Kiểm tra cảnh có `images/` và `sparse/0/` dùng được; trả về thông tin và cảnh báo."""
    images_dir = scene_dir / "images"
    model_dir = scene_dir / "sparse" / "0"
    if not images_dir.is_dir():
        raise SceneError(f"{images_dir} not found")
    if not model_exists(model_dir):
        raise SceneError(f"{model_dir} must contain cameras.bin, images.bin and points3D.bin")
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")
    if not images:
        raise SceneError(f"no registered images in {model_dir}")
    names = sorted(image.name for image in images.values())
    missing = [name for name in names if not (images_dir / name).is_file()]
    if missing:
        raise SceneError(f"{len(missing)} registered image(s) are missing in {images_dir}, e.g. {missing[:3]}")

    models = sorted({camera.model for camera in cameras.values()})
    sizes = sorted({(camera.width, camera.height) for camera in cameras.values()})
    warnings = []
    if any(model not in UNDISTORTED_MODELS for model in models):
        warnings.append(f"camera model(s) {models} are not undistorted: gsplat undistorts internally, the Inria "
                        "baseline needs PINHOLE images (use colmap_runner, which runs image_undistorter)")
    if max(width for width, _ in sizes) > INRIA_MAX_WIDTH:
        warnings.append(f"images are wider than {INRIA_MAX_WIDTH} px: Inria train.py downsamples them, gsplat with "
                        "data_factor 1 does not, so resolutions would differ from B0 (extract frames with "
                        f"--max-side {INRIA_MAX_WIDTH})")
    image_files = [path for path in images_dir.iterdir() if path.is_file()]
    return {
        "num_registered": len(names),
        "num_image_files": len(image_files),
        "num_cameras": len(cameras),
        "camera_models": models,
        "image_sizes": [list(size) for size in sizes],
        "warnings": warnings,
    }


def git_state(path: Path) -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def gpu_name(index: str) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", index, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.CalledProcessError):
        return None


def torch_info() -> dict:
    try:
        import torch  # có trong env gs-gsplat; không có trong môi trường test CPU
    except ImportError:
        return {}
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda}


def start_gpu_monitor(csv_path: Path, index: str):
    """Ghi memory.used mỗi 2 giây, cùng định dạng với run_baseline_deepblending.sh."""
    if shutil.which("nvidia-smi") is None:
        return None
    handle = open(csv_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        ["nvidia-smi", "-i", index, "--query-gpu=timestamp,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-l", "2"],
        stdout=handle, stderr=subprocess.DEVNULL)
    process.log_handle = handle  # đóng cùng lúc với tiến trình
    return process


def stop_gpu_monitor(process) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    process.log_handle.close()


def run_command(cmd: list[str], cwd: Path, env: dict, log_path: Path,
                progress_interval: float = PROGRESS_INTERVAL) -> int:
    """Chạy lệnh, in ra màn hình và ghi log; trả về mã thoát.

    Thanh tiến độ tqdm của gsplat cập nhật mỗi vòng lặp bằng ký tự '\\r'. Các đoạn kết thúc bằng '\\r'
    chỉ được ghi tối đa một lần mỗi `progress_interval` giây; dòng kết thúc bằng '\\n' luôn được ghi.
    """
    last_progress = float("-inf")
    with open(log_path, "a", encoding="utf-8") as log:
        def emit(raw: bytes, newline: bool) -> None:
            nonlocal last_progress
            if not newline:
                if not raw.strip():
                    return  # đoạn rỗng trước '\r' đầu tiên của tqdm
                now = time.monotonic()
                if now - last_progress < progress_interval:
                    return
                last_progress = now
            text = raw.decode("utf-8", errors="replace") + "\n"
            sys.stdout.write(text)
            log.write(text)

        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        buffer = b""
        while True:
            chunk = process.stdout.read1(65536)
            buffer += chunk
            while buffer:
                cut = min((i for i in (buffer.find(b"\n"), buffer.find(b"\r")) if i >= 0), default=-1)
                if cut < 0:
                    break
                if buffer[cut:cut + 1] == b"\r":
                    if cut + 1 == len(buffer) and chunk:
                        break  # chờ thêm dữ liệu để biết có phải "\r\n" không
                    if buffer[cut + 1:cut + 2] == b"\n":
                        emit(buffer[:cut], True)
                        buffer = buffer[cut + 2:]
                        continue
                    emit(buffer[:cut], False)
                else:
                    emit(buffer[:cut], True)
                buffer = buffer[cut + 1:]
            if not chunk:
                break
        if buffer:
            emit(buffer, True)
        sys.stdout.flush()
        return process.wait()


def request_differences(meta: dict, spec, seed: int, cap_max: int | None, split_digest: str) -> list[str]:
    """Các khác biệt giữa một lượt chạy đã có (run_meta.json) và yêu cầu hiện tại."""
    previous = meta.get("spec", {})
    pairs = {
        "preset": (previous.get("preset"), spec.preset),
        "protocol": (previous.get("protocol"), spec.protocol),
        "args": (previous.get("args"), spec.args),
        "seed": (meta.get("seed"), seed),
        "cap_max": ((meta.get("cap_max") or {}).get("value"), cap_max),
        "split": (meta.get("split", {}).get("sha256"), split_digest),
    }
    return [name for name, (old, new) in pairs.items() if old != new]


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", required=True, help="tên chế độ trong file cấu hình (default, mcmc, ...)")
    parser.add_argument("--scene", required=True, type=Path, help="thư mục cảnh có images/ và sparse/0/")
    parser.add_argument("--scene-name", help="tên cảnh ghi vào kết quả (mặc định: tên thư mục cảnh)")
    parser.add_argument("--out", required=True, type=Path, help="thư mục kết quả của lượt chạy này")
    parser.add_argument("--gsplat-dir", type=Path, default=Path(os.environ.get("WS", "/workspace")) / "code" / "gsplat")
    parser.add_argument("--seed", type=int, default=None, help="mặc định: default_seed trong file cấu hình")
    parser.add_argument("--test-list", type=Path, help="file danh sách ảnh test (thay cho mỗi ảnh thứ test_every)")
    cap = parser.add_mutually_exclusive_group()
    cap.add_argument("--cap-max", type=int, help="giới hạn số Gaussian cho MCMC (ghi đè chính sách trong cấu hình)")
    cap.add_argument("--cap-max-from", type=Path, help="thư mục lượt chạy baseline cùng cảnh để lấy số Gaussian")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="tham số riêng của chế độ, vd. --set opacity_reg=0.001 (lặp lại được)")
    parser.add_argument("--protocol", action="append", default=[], metavar="KEY=VALUE",
                        help="đổi protocol (bị ghi là sai khác, vd. max_steps=3000 để chạy thử)")
    parser.add_argument("--gpu", default="0", help="chỉ số GPU khi CUDA_VISIBLE_DEVICES chưa đặt")
    parser.add_argument("--python", default=sys.executable, help="Python của env gs-gsplat")
    parser.add_argument("--force", action="store_true", help="chạy lại; thư mục cũ được đổi tên, không xóa")
    parser.add_argument("--rerun-incomplete", action="store_true",
                        help="chạy lại nếu thư mục kết quả là lượt dở dang / lỗi (không đụng tới lượt OK)")
    parser.add_argument("--allow-gsplat-mismatch", action="store_true",
                        help="cho phép checkout gsplat khác commit trong cấu hình (ghi vào run_meta)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None, runner: Callable[..., int] = run_command) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        protocol_overrides = parse_assignments(args.protocol)
        spec = resolve_mode(config, args.mode, parse_assignments(args.set), protocol_overrides)
        scene_dir = args.scene.resolve()
        scene_info = inspect_scene(scene_dir)
        split = make_split(scene_dir, spec.protocol["test_every"], args.test_list)
        seed = args.seed if args.seed is not None else int(config.get("default_seed", 42))
        baseline = load_summary(args.cap_max_from).to_dict() if args.cap_max_from else None
        cap_max, cap_source = resolve_cap_max(spec, args.cap_max, baseline, seed)
    except (ValueError, SceneError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if baseline is not None and baseline.get("split_sha256") != split.digest():
        print("ERROR: the baseline run used a different train/test split; the Gaussian budget would not be "
              "comparable", file=sys.stderr)
        return 2

    out = args.out.resolve()
    examples = args.gsplat_dir.resolve() / "examples"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and "," in visible:
        print("ERROR: CUDA_VISIBLE_DEVICES lists several GPUs; training must use exactly one", file=sys.stderr)
        return 2
    gpu_index = visible if visible not in (None, "") else str(args.gpu)

    def launcher(record: str, resolved: str, trainer: list[str]) -> list[str]:
        cmd = [args.python, "-m", "indoor3d.train.gsplat_launcher", "--examples-dir", str(examples),
               "--seed", str(seed), "--record-split", str(out / record), "--resolved-config", str(out / resolved)]
        if split.rule == "test_list":
            cmd += ["--split-file", str(out / "split.json")]
        return cmd + ["--", *trainer]

    # Với danh sách test riêng, gsplat đọc một bản dữ liệu đã lọc điểm SfM (xem split.make_train_view).
    data_dir = out / "data" if split.rule == "test_list" else scene_dir
    train_cmd = launcher("split_applied.json", "resolved_config.json", trainer_args(spec, data_dir, out, cap_max))
    eval_cmd = launcher("split_applied_eval.json", "resolved_config_eval.json",
                        eval_args(spec, data_dir, out, cap_max))
    warnings = list(scene_info["warnings"])
    if spec.deviations:
        warnings.append(f"protocol deviations: {sorted(spec.deviations)} - not comparable with protocol runs")

    if args.dry_run:
        plan = {"spec": spec.to_dict(), "seed": seed, "cap_max": cap_source, "scene_info": scene_info,
                "data_dir": str(data_dir),
                "split": {k: v for k, v in split.to_dict().items() if k not in ("train", "test")},
                "train": shlex.join(train_cmd), "eval": shlex.join(eval_cmd), "warnings": warnings}
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if not (examples / "simple_trainer.py").is_file():
        print(f"ERROR: {examples / 'simple_trainer.py'} not found (run scripts/setup_gsplat.sh)", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()):
        previous = load_summary(out)
        previous_meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8")) \
            if (out / "run_meta.json").is_file() else {}
        differences = request_differences(previous_meta, spec, seed, cap_max, split.digest())
        if previous.status == "OK" and not args.force:
            if not differences:
                print(f"[run] {out} already has an OK run with the same configuration; skipping")
                return 0
            print(f"ERROR: {out} holds an OK run with a different configuration ({', '.join(differences)}); "
                  "use another --out, or --force to replace it (the old folder is renamed)", file=sys.stderr)
            return 2
        if not (args.force or args.rerun_incomplete):
            print(f"ERROR: {out} exists but is not a finished run (status {previous.status}); "
                  "use --rerun-incomplete or --force (the old folder is renamed)", file=sys.stderr)
            return 2
        stamp = f"{dt.datetime.now():%Y%m%d_%H%M%S}"
        backup, counter = out.with_name(f"{out.name}.prev_{stamp}"), 1
        while backup.exists():
            backup, counter = out.with_name(f"{out.name}.prev_{stamp}_{counter}"), counter + 1
        out.rename(backup)
        print(f"[run] previous results moved to {backup}")

    gsplat_commit, gsplat_dirty = git_state(args.gsplat_dir)
    expected = config.get("trainer", {}).get("commit")
    if expected and gsplat_commit != expected:
        message = f"gsplat checkout is at {gsplat_commit}, configuration expects {expected}"
        if not args.allow_gsplat_mismatch:
            print(f"ERROR: {message} (run scripts/setup_gsplat.sh or pass --allow-gsplat-mismatch)", file=sys.stderr)
            return 2
        warnings.append(message)

    out.mkdir(parents=True, exist_ok=True)
    write_split(split, out / "split.json")
    train_view = None
    if split.rule == "test_list":
        try:
            train_view = make_train_view(scene_dir, data_dir, split)
        except ValueError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(f"[run] test list: {train_view['points_kept']}/{train_view['points_total']} SfM points are seen by "
              f">= {train_view['min_train_views']} training images and kept for initialisation")

    repo_commit, repo_dirty = git_state(REPO_DIR)
    for message in warnings:
        print(f"WARN: {message}")

    meta = {
        "scene": args.scene_name or scene_dir.name,
        "scene_dir": str(scene_dir),
        "mode": spec.mode,
        "seed": seed,
        "status": "RUNNING",
        "failed_step": None,
        "spec": spec.to_dict(),
        "cap_max": cap_source,
        "split": {k: v for k, v in split.to_dict().items() if k not in ("train", "test")},
        "scene_info": scene_info,
        "data_dir": str(data_dir),
        "train_view": train_view,
        "commands": {"train": train_cmd, "eval": eval_cmd},
        "gsplat_dir": str(args.gsplat_dir.resolve()),
        "gsplat_commit": gsplat_commit,
        "gsplat_dirty": gsplat_dirty,
        "gsplat_commit_expected": expected,
        "repo_commit": repo_commit,
        "repo_dirty": repo_dirty,
        "gpu": gpu_name(gpu_index),
        "cuda_visible_devices": gpu_index,
        "python": sys.version.split()[0],
        **torch_info(),
        "warnings": warnings,
        "started": _now(),
    }
    meta_path = out / "run_meta.json"
    _write_json(meta_path, meta)

    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # cùng thứ tự với chỉ số mà nvidia-smi -i dùng
    env["CUDA_VISIBLE_DEVICES"] = gpu_index
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    for step, cmd, log_name in (("train", train_cmd, "train.log"), ("eval", eval_cmd, "eval.log")):
        print(f"\n===== [{dt.datetime.now():%H:%M:%S}] {step}: {spec.mode} / {meta['scene']} =====")
        monitor = start_gpu_monitor(out / "gpu_log.csv", gpu_index) if step == "train" else None
        started = time.monotonic()
        try:
            code = runner(cmd, examples, env, out / log_name)
        finally:
            stop_gpu_monitor(monitor)
        meta[f"{step}_seconds"] = round(time.monotonic() - started, 1)
        if code != 0:
            meta.update(status="FAILED", failed_step=step, finished=_now(), exit_code=code)
            _write_json(meta_path, meta)
            print(f"ERROR: {step} failed with exit code {code}; see {out / log_name}", file=sys.stderr)
            return 1
        _write_json(meta_path, meta)

    meta["finished"] = _now()
    meta["status"] = "DONE"
    _write_json(meta_path, meta)
    result = load_run(out)
    if result.status != "OK":
        meta["status"] = "FAILED"
        meta["failed_step"] = "collect"
        _write_json(meta_path, meta)
        result = load_run(out)
    _write_json(out / "summary.json", result.to_dict())
    metrics = result.metrics or {}
    print(f"\n[run] {result.status} | {spec.mode} / {meta['scene']} | PSNR {metrics.get('PSNR', float('nan')):.3f} "
          f"SSIM {metrics.get('SSIM', float('nan')):.4f} LPIPS {metrics.get('LPIPS', float('nan')):.4f} | "
          f"Gaussians {result.num_gaussians} | train {meta['train_seconds'] / 60:.1f} min | {out / 'summary.json'}")
    if result.split_applied_ok is False:
        print("WARN: the images gsplat used differ from split.json", file=sys.stderr)
    return 0 if result.status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
