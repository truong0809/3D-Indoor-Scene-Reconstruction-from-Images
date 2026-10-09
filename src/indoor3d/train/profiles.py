"""Cấu hình các chế độ huấn luyện (configs/train_modes.json) và tham số dòng lệnh cho gsplat.

File cấu hình gồm:

- `protocol`: điều kiện chung cho mọi chế độ (độ phân giải, cách chia test, số vòng lặp, mạng LPIPS,
  chế độ raster...). Hai chế độ chỉ được so sánh khi protocol giống nhau.
- `modes`: mỗi chế độ chọn preset của `simple_trainer.py` (`default` hoặc `mcmc`), tham số riêng (`args`)
  và cách đặt giới hạn số Gaussian (`cap_max`) cho MCMC.

Mọi thay đổi protocol (trong `modes.<tên>.protocol` hoặc qua dòng lệnh) được ghi thành `deviations`
để bước so sánh đánh dấu lượt chạy đó là không cùng điều kiện.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# khóa protocol -> tên tham số của simple_trainer.py (tyro dùng gạch ngang)
PROTOCOL_FLAGS = {
    "data_factor": "data-factor",
    "test_every": "test-every",
    "max_steps": "max-steps",
    "lpips_net": "lpips-net",
    "antialiased": "antialiased",
    "save_ply": "save-ply",
    "disable_video": "disable-video",
    "init_type": "init-type",
}
PROTOCOL_TYPES = {
    "data_factor": int,
    "test_every": int,
    "max_steps": int,
    "lpips_net": str,
    "antialiased": bool,
    "save_ply": bool,
    "disable_video": bool,
    "init_type": str,
}
# Tham số do wrapper quản lý; không được đặt qua `args` để tránh lệch điều kiện mà không ai biết.
RESERVED_FLAGS = set(PROTOCOL_FLAGS.values()) | {
    "data-dir", "result-dir", "eval-steps", "save-steps", "ply-steps", "disable-viewer",
    "ckpt", "strategy.cap-max", "steps-scaler", "batch-size",
}
PRESETS = ("default", "mcmc")
CAP_POLICIES = ("match_baseline", "fixed")


def normalize_flag(key: str) -> str:
    """'strategy.refine_stop_iter' -> 'strategy.refine-stop-iter'; bỏ '--' ở đầu nếu có."""
    key = key.strip().lstrip("-")
    if not key or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        raise ValueError(f"invalid trainer argument name: {key!r}")
    return ".".join(part.replace("_", "-") for part in key.split("."))


def parse_value(text: str):
    """Chuỗi dòng lệnh -> bool / int / float / str."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if re.fullmatch(r"[+-]?\d+", text.strip()):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def parse_assignments(items: list[str] | None) -> dict:
    """['a=1', 'strategy.b=x'] -> {'a': 1, 'strategy.b': 'x'}."""
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = parse_value(value)
    return result


def to_cli(flag: str, value) -> list[str]:
    """Một cặp (tham số, giá trị) -> token dòng lệnh theo quy ước của tyro."""
    if isinstance(value, bool):
        if value:
            return [f"--{flag}"]
        prefix, _, leaf = flag.rpartition(".")
        return [f"--{prefix}.no-{leaf}" if prefix else f"--no-{leaf}"]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"empty list for --{flag}")
        return [f"--{flag}", *(str(item) for item in value)]
    if value is None:
        raise ValueError(f"None is not allowed for --{flag}")
    return [f"--{flag}", str(value)]


def _coerce_protocol(key: str, value):
    if key not in PROTOCOL_TYPES:
        raise ValueError(f"unknown protocol key {key!r} (known: {sorted(PROTOCOL_TYPES)})")
    expected = PROTOCOL_TYPES[key]
    if expected is bool and not isinstance(value, bool):
        raise ValueError(f"protocol {key} must be true/false")
    if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"protocol {key} must be an integer")
    if expected is str and not isinstance(value, str):
        raise ValueError(f"protocol {key} must be a string")
    return value


@dataclass
class ModeSpec:
    config_name: str
    mode: str
    description: str
    preset: str
    protocol: dict
    base_protocol: dict
    deviations: dict = field(default_factory=dict)
    args: dict = field(default_factory=dict)  # tham số đã chuẩn hóa -> giá trị
    cap_max_policy: dict | None = None
    baseline_mode: str = "default"

    @property
    def max_steps(self) -> int:
        return int(self.protocol["max_steps"])

    @property
    def final_step(self) -> int:
        """gsplat lưu checkpoint / chỉ số của vòng cuối với chỉ số max_steps - 1."""
        return self.max_steps - 1

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "mode": self.mode,
            "description": self.description,
            "preset": self.preset,
            "protocol": self.protocol,
            "base_protocol": self.base_protocol,
            "deviations": self.deviations,
            "args": self.args,
            "cap_max_policy": self.cap_max_policy,
            "baseline_mode": self.baseline_mode,
        }


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "protocol", "modes"):
        if key not in config:
            raise ValueError(f"{path}: missing '{key}'")
    for key, value in config["protocol"].items():
        _coerce_protocol(key, value)
    missing = sorted(set(PROTOCOL_TYPES) - set(config["protocol"]))
    if missing:
        raise ValueError(f"{path}: protocol is missing {missing}")
    baseline = config.get("baseline_mode", "default")
    if baseline not in config["modes"]:
        raise ValueError(f"{path}: baseline_mode {baseline!r} is not defined in modes")
    return config


def resolve_mode(config: dict, mode: str, set_args: dict | None = None,
                 protocol_overrides: dict | None = None) -> ModeSpec:
    """Ghép protocol chung + cấu hình của chế độ + tham số dòng lệnh thành một ModeSpec."""
    modes = config["modes"]
    if mode not in modes:
        raise ValueError(f"unknown mode {mode!r} (available: {sorted(modes)})")
    entry = modes[mode]
    preset = entry.get("preset")
    if preset not in PRESETS:
        raise ValueError(f"mode {mode}: preset must be one of {PRESETS}")

    base = dict(config["protocol"])
    protocol = dict(base)
    deviations = {}
    for source, overrides in (("mode", entry.get("protocol", {})), ("cli", protocol_overrides or {})):
        for key, value in overrides.items():
            value = _coerce_protocol(key, value)
            if value != base[key]:
                deviations[key] = {"base": base[key], "used": value, "source": source}
            else:
                deviations.pop(key, None)
            protocol[key] = value

    args = {}
    for source, items in (("mode", entry.get("args", {})), ("cli", set_args or {})):
        for key, value in items.items():
            flag = normalize_flag(key)
            if flag in RESERVED_FLAGS:
                raise ValueError(f"--{flag} is controlled by the protocol / wrapper; "
                                 f"change it through the protocol instead ({source})")
            args[flag] = value

    cap = entry.get("cap_max")
    if cap is not None:
        if preset != "mcmc":
            raise ValueError(f"mode {mode}: cap_max only applies to the mcmc preset")
        if cap.get("policy") not in CAP_POLICIES:
            raise ValueError(f"mode {mode}: cap_max.policy must be one of {CAP_POLICIES}")
        if cap["policy"] == "fixed" and not isinstance(cap.get("value"), int):
            raise ValueError(f"mode {mode}: cap_max.value must be an integer")

    return ModeSpec(
        config_name=config["name"],
        mode=mode,
        description=entry.get("description", ""),
        preset=preset,
        protocol=protocol,
        base_protocol=base,
        deviations=deviations,
        args=args,
        cap_max_policy=cap,
        baseline_mode=config.get("baseline_mode", "default"),
    )


def resolve_cap_max(spec: ModeSpec, cli_value: int | None = None, baseline_summary: dict | None = None,
                    seed: int | None = None) -> tuple[int | None, dict | None]:
    """Giá trị --strategy.cap-max và nguồn gốc của nó (ghi vào run_meta).

    Với chính sách match_baseline, lượt baseline phải đã xong (status OK), đúng chế độ baseline,
    cùng protocol với lượt đang chạy, không có sai khác protocol của riêng nó và cùng seed.
    (Kiểm tra cùng tập test nằm ở run.py vì cần split của lượt đang chạy.)
    """
    if cli_value is not None:
        if spec.preset != "mcmc":
            raise ValueError("--cap-max only applies to the mcmc preset")
        return int(cli_value), {"policy": "cli", "value": int(cli_value)}
    policy = spec.cap_max_policy
    if policy is None:
        return None, None
    if policy["policy"] == "fixed":
        return int(policy["value"]), {"policy": "fixed", "value": int(policy["value"])}
    # match_baseline: cùng số Gaussian cuối của lượt baseline trên cùng cảnh (giao thức của 3DGS-MCMC)
    if baseline_summary is None:
        raise ValueError(f"mode {spec.mode} needs the final Gaussian count of a '{spec.baseline_mode}' run "
                         "(pass --cap-max-from <baseline run dir> or --cap-max N)")
    count = baseline_summary.get("num_gaussians")
    if baseline_summary.get("status") != "OK" or not isinstance(count, int) or count <= 0:
        raise ValueError(f"the baseline run {baseline_summary.get('run_dir')} is not finished "
                         f"(status {baseline_summary.get('status')}; status must be OK)")
    if baseline_summary.get("mode") != spec.baseline_mode:
        raise ValueError(f"--cap-max-from points to a '{baseline_summary.get('mode')}' run, "
                         f"expected '{spec.baseline_mode}'")
    if baseline_summary.get("protocol") != spec.protocol:
        raise ValueError("the baseline run used a different protocol; its Gaussian count is not a comparable budget")
    if baseline_summary.get("deviations"):
        raise ValueError(f"the baseline run deviates from the protocol {sorted(baseline_summary['deviations'])}")
    if seed is not None and baseline_summary.get("seed") != seed:
        raise ValueError(f"the baseline run used seed {baseline_summary.get('seed')}, this run uses seed {seed}")
    return count, {
        "policy": "match_baseline",
        "value": count,
        "baseline_mode": spec.baseline_mode,
        "baseline_run_dir": baseline_summary.get("run_dir"),
        "baseline_seed": baseline_summary.get("seed"),
        "baseline_split_sha256": baseline_summary.get("split_sha256"),
    }


def trainer_args(spec: ModeSpec, scene_dir: Path, result_dir: Path, cap_max: int | None = None) -> list[str]:
    """Tham số cho `simple_trainer.py` khi huấn luyện (không đánh giá giữa chừng)."""
    p = spec.protocol
    argv = [spec.preset]
    argv += to_cli("data-dir", str(scene_dir))
    argv += to_cli("result-dir", str(result_dir))
    for key, flag in PROTOCOL_FLAGS.items():
        argv += to_cli(flag, p[key])
    argv += to_cli("eval-steps", [-1])  # đánh giá tách riêng sau khi huấn luyện (như benchmark của gsplat)
    argv += to_cli("save-steps", [spec.max_steps])
    argv += to_cli("ply-steps", [spec.max_steps])
    argv += to_cli("disable-viewer", True)
    if cap_max is not None:
        argv += to_cli("strategy.cap-max", int(cap_max))
    for flag, value in spec.args.items():
        argv += to_cli(flag, value)
    return argv


def checkpoint_path(spec: ModeSpec, result_dir: Path) -> Path:
    return Path(result_dir) / "ckpts" / f"ckpt_{spec.final_step}_rank0.pt"


def eval_args(spec: ModeSpec, scene_dir: Path, result_dir: Path, cap_max: int | None = None) -> list[str]:
    """Đánh giá checkpoint cuối với đúng cấu hình đã huấn luyện (gsplat bỏ qua huấn luyện khi có --ckpt)."""
    return trainer_args(spec, scene_dir, result_dir, cap_max) + to_cli("ckpt", [str(checkpoint_path(spec, result_dir))])
