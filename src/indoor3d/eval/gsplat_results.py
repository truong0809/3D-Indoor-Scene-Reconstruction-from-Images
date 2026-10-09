"""Đọc kết quả một lượt huấn luyện gsplat (một chế độ, một cảnh, một seed).

Thư mục kết quả (`--result-dir` của simple_trainer.py) sau khi `indoor3d.train.run` chạy xong:

    run_meta.json                     do indoor3d.train.run ghi (cấu hình, lệnh, commit, GPU, thời gian)
    split.json / split_applied.json   tập train/test dự kiến và tập gsplat thực sự dùng
    resolved_config.json              cấu hình simple_trainer sau khi phân giải
    gpu_log.csv                       nhật ký nvidia-smi trong lúc huấn luyện
    stats/train_step<N>_rank0.json    gsplat: mem (GiB, max_memory_allocated), ellipse_time, num_GS
    stats/val_step<N>.json            gsplat: psnr, ssim, lpips, ellipse_time (giây/ảnh), num_GS
    ckpts/ckpt_<N>_rank0.pt, ply/point_cloud_<N>.ply

Số liệu trong summary.json chỉ lấy từ các file trên, không ước lượng.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from indoor3d.eval.compare_reference import SceneResult, peak_gpu_memory_mib, read_ply_vertex_count

_VAL = re.compile(r"^val_step(\d+)\.json$")
_TRAIN = re.compile(r"^train_step(\d+)_rank0\.json$")


def latest_stats(stats_dir: Path, kind: str) -> tuple[int, dict] | None:
    """File stats có bước lớn nhất: kind = 'val' hoặc 'train'."""
    pattern = _VAL if kind == "val" else _TRAIN
    best = None
    if not stats_dir.is_dir():
        return None
    for path in stats_dir.iterdir():
        match = pattern.match(path.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    if best is None:
        return None
    return best[0], json.loads(best[1].read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _size_mb(path: Path) -> float | None:
    return round(path.stat().st_size / 2**20, 2) if path.is_file() else None


@dataclass
class RunResult:
    run_dir: str
    scene: str | None = None
    mode: str | None = None
    seed: int | None = None
    status: str = "MISSING"  # OK | MISSING | FAILED | RUNNING
    metrics: dict = field(default_factory=dict)  # PSNR, SSIM, LPIPS (lượt đánh giá cuối)
    eval_step: int | None = None
    num_gaussians: int | None = None
    cap_max: int | None = None
    train_seconds: float | None = None  # thời gian thực của tiến trình huấn luyện (gồm nạp dữ liệu, ghi file)
    train_loop_seconds: float | None = None  # ellipse_time của gsplat: chỉ vòng lặp huấn luyện
    eval_seconds: float | None = None
    peak_vram_mib: float | None = None  # nvidia-smi (gồm cả bộ nhớ đệm của PyTorch và CUDA context)
    torch_peak_gib: float | None = None  # torch.cuda.max_memory_allocated() do gsplat ghi
    render_ms_per_image: float | None = None
    ply_size_mb: float | None = None
    ckpt_size_mb: float | None = None
    protocol: dict = field(default_factory=dict)
    deviations: dict = field(default_factory=dict)
    args: dict = field(default_factory=dict)  # tham số riêng của chế độ (--set)
    cap_policy: str | None = None
    split_sha256: str | None = None
    split_applied_ok: bool | None = None
    gsplat_commit: str | None = None
    gpu: str | None = None
    lpips_net: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_scene_result(self) -> SceneResult:
        """Chuyển sang SceneResult để dùng lại compare_reference.compare / format_table."""
        return SceneResult(
            scene=self.scene or Path(self.run_dir).name,
            model_dir=self.run_dir,
            status="OK" if self.status == "OK" else "MISSING",
            metrics=dict(self.metrics),
            train_seconds=self.train_seconds,
            peak_vram_mib=self.peak_vram_mib,
            num_gaussians=self.num_gaussians,
            ply_size_mb=self.ply_size_mb,
        )


def _split_applied_ok(run_dir: Path) -> bool | None:
    """So tập ảnh gsplat thực sự dùng với tập dự kiến (split.json).

    Tập train lấy từ lượt huấn luyện (split_applied.json), tập val lấy từ lượt đánh giá
    (split_applied_eval.json) nếu có, vì chỉ số được tính ở lượt đó.
    """
    planned = _read_json(run_dir / "split.json")
    trained = _read_json(run_dir / "split_applied.json").get("splits", {})
    evaluated = _read_json(run_dir / "split_applied_eval.json").get("splits", {}) or trained
    if not planned or "train" not in trained or "val" not in evaluated:
        return None
    return (sorted(trained["train"]) == sorted(planned.get("train", []))
            and sorted(evaluated["val"]) == sorted(planned.get("test", [])))


def load_run(run_dir: Path) -> RunResult:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "run_meta.json")
    spec = meta.get("spec", {})
    protocol = spec.get("protocol", {})
    result = RunResult(
        run_dir=str(run_dir),
        scene=meta.get("scene"),
        mode=meta.get("mode"),
        seed=meta.get("seed"),
        cap_max=(meta.get("cap_max") or {}).get("value"),
        cap_policy=(meta.get("cap_max") or {}).get("policy"),
        train_seconds=meta.get("train_seconds"),
        eval_seconds=meta.get("eval_seconds"),
        protocol=protocol,
        deviations=spec.get("deviations", {}),
        args=spec.get("args", {}),
        split_sha256=meta.get("split", {}).get("sha256"),
        gsplat_commit=meta.get("gsplat_commit"),
        gpu=meta.get("gpu"),
        lpips_net=protocol.get("lpips_net"),
    )
    final_step = int(protocol["max_steps"]) - 1 if "max_steps" in protocol else None

    val = latest_stats(run_dir / "stats", "val")
    if val is not None:
        step, stats = val
        if all(key in stats for key in ("psnr", "ssim", "lpips")):
            result.metrics = {"PSNR": float(stats["psnr"]), "SSIM": float(stats["ssim"]),
                              "LPIPS": float(stats["lpips"])}
            result.eval_step = step
            if stats.get("num_GS") is not None:
                result.num_gaussians = int(stats["num_GS"])
            if stats.get("ellipse_time") is not None:
                result.render_ms_per_image = round(float(stats["ellipse_time"]) * 1000, 3)

    train = latest_stats(run_dir / "stats", "train")
    if train is not None:
        _, stats = train
        if stats.get("mem") is not None:
            result.torch_peak_gib = round(float(stats["mem"]), 3)
        if stats.get("ellipse_time") is not None:
            result.train_loop_seconds = round(float(stats["ellipse_time"]), 1)
        if result.num_gaussians is None and stats.get("num_GS") is not None:
            result.num_gaussians = int(stats["num_GS"])

    result.peak_vram_mib = peak_gpu_memory_mib(run_dir / "gpu_log.csv")
    if final_step is not None:
        ply = run_dir / "ply" / f"point_cloud_{final_step}.ply"
        result.ply_size_mb = _size_mb(ply)
        if result.num_gaussians is None and ply.is_file():
            result.num_gaussians = read_ply_vertex_count(ply)
        result.ckpt_size_mb = _size_mb(run_dir / "ckpts" / f"ckpt_{final_step}_rank0.pt")
    result.split_applied_ok = _split_applied_ok(run_dir)

    meta_status = meta.get("status")
    if meta_status in ("FAILED", "RUNNING"):
        result.status = meta_status
    elif result.metrics and (final_step is None or result.eval_step == final_step):
        result.status = "OK"
    return result


def load_summary(run_dir: Path) -> RunResult:
    """Đọc summary.json nếu có, nếu không thì tính lại từ các file gốc."""
    path = Path(run_dir) / "summary.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in RunResult.__dataclass_fields__}
        return RunResult(**{key: value for key, value in data.items() if key in known})
    return load_run(Path(run_dir))
