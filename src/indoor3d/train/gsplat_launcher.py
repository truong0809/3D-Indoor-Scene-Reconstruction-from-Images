"""Chạy `examples/simple_trainer.py` của gsplat kèm vài mở rộng nhỏ, KHÔNG sửa file nào của gsplat.

Các mở rộng được áp dụng bằng cách thay thuộc tính của module trước khi simple_trainer được nạp:

- `--seed N`: simple_trainer gán cứng seed `42 + rank`; launcher đổi thành `N + rank`
  (cần cho các lần chạy lặp, docs/plan.md mục 4.2).
- `--split-file F`: dùng đúng danh sách ảnh train/test trong F (JSON do `indoor3d.train.split` ghi)
  thay cho quy tắc "mỗi ảnh thứ test_every". Dùng cho đoạn quay kiểm tra riêng.
- `--record-split F`: ghi danh sách ảnh train/val mà gsplat thực sự dùng, làm bằng chứng cho báo cáo.
- `--resolved-config F`: ghi toàn bộ cấu hình sau khi phân giải (kể cả tham số của strategy).
- `--resolve-only`: chỉ phân giải cấu hình rồi thoát; không cần GPU, dùng để kiểm tra tham số.

Launcher dừng nếu gói `gsplat` đang cài khác mã nguồn trong thư mục checkout (so SHA-256 các file strategy
và rasterization). Lý do: v1.5.3 và commit được ghim cùng báo phiên bản "1.5.3", nên cài nhầm sẽ không
bị phát hiện qua số phiên bản.

Chỉ hỗ trợ 1 GPU: với nhiều GPU, gsplat tự tạo tiến trình con và các thay đổi trên không theo sang.

Cách gọi (bằng Python của env gs-gsplat):

    python -m indoor3d.train.gsplat_launcher --examples-dir /workspace/code/gsplat/examples \\
        --seed 42 --record-split split_applied.json --resolved-config resolved_config.json \\
        -- default --data-dir <cảnh> --result-dir <thư mục kết quả> ...
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

GSPLAT_DEFAULT_SEED = 42  # simple_trainer.Runner: set_random_seed(42 + local_rank)
GUARDED_FILES = ("strategy/default.py", "strategy/mcmc.py", "strategy/ops.py", "rendering.py")


def _write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def config_to_dict(cfg) -> dict:
    """Config (dataclass) của simple_trainer -> dict có thể ghi JSON, kèm tên lớp strategy."""
    data = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(vars(cfg))
    strategy = getattr(cfg, "strategy", None)
    if strategy is not None:
        data["strategy_type"] = type(strategy).__name__
    return data


def package_fingerprint(package_dir: Path) -> dict:
    result = {}
    for name in GUARDED_FILES:
        path = Path(package_dir) / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


def check_installed_gsplat(examples: Path) -> dict:
    """Gói gsplat được import phải trùng mã nguồn với checkout (<checkout>/gsplat) chứa examples/."""
    import gsplat

    installed = Path(gsplat.__file__).resolve().parent
    checkout = examples.parent / "gsplat"
    info = {"installed_dir": str(installed), "version": getattr(gsplat, "__version__", None),
            "fingerprint": package_fingerprint(installed)}
    if checkout.is_dir():
        expected = package_fingerprint(checkout)
        differing = [name for name in GUARDED_FILES if info["fingerprint"][name] != expected[name]]
        if differing:
            raise RuntimeError(f"the installed gsplat ({installed}) differs from the checkout {checkout} in "
                               f"{differing}; reinstall it with scripts/setup_gsplat.sh (FORCE_REBUILD=1)")
        info["matches_checkout"] = str(checkout)
    return info


def _check_origin(module, examples: Path) -> None:
    """Module phải được nạp từ examples/ của gsplat (`utils`, `datasets` là tên rất dễ trùng;
    `datasets` còn là namespace package nên một gói cùng tên cài trong env sẽ che mất nó)."""
    origin = Path(getattr(module, "__file__", "") or "").resolve()
    if examples not in origin.parents:
        raise RuntimeError(f"{module.__name__} was imported from {origin}, not from {examples}")


def patch_seed(seed: int, examples: Path) -> None:
    import utils as gs_utils  # examples/utils.py

    _check_origin(gs_utils, examples)
    original = gs_utils.set_random_seed

    def set_random_seed(value: int) -> None:
        original(int(seed) + (int(value) - GSPLAT_DEFAULT_SEED))

    gs_utils.set_random_seed = set_random_seed


def patch_split(split_spec: dict | None, record_path: Path | None, examples: Path) -> None:
    """Thay chỉ số ảnh của `datasets.colmap.Dataset` theo `split_spec` và/hoặc ghi lại chỉ số đã dùng."""
    import numpy as np

    import datasets.colmap as gs_colmap  # examples/datasets/colmap.py

    _check_origin(gs_colmap, examples)
    original_init = gs_colmap.Dataset.__init__
    recorded: dict = {"rule": split_spec["rule"] if split_spec else "gsplat test_every", "splits": {}}

    # Giữ đúng tên tham số `split` của Dataset.__init__ vì simple_trainer truyền nó dạng keyword.
    def init(self, parser, split="train", *args, **kwargs):
        original_init(self, parser, split, *args, **kwargs)
        names = list(parser.image_names)
        if split_spec is not None:
            wanted = split_spec["train"] if split == "train" else split_spec["test"]
            position = {name: index for index, name in enumerate(names)}
            missing = [name for name in wanted if name not in position]
            if missing:
                raise RuntimeError(f"{len(missing)} image(s) of the split file are not known to gsplat, "
                                   f"e.g. {missing[:3]}")
            self.indices = np.array(sorted(position[name] for name in wanted), dtype=np.int64)
        if record_path is not None:
            recorded["splits"][split] = [names[int(index)] for index in self.indices]
            _write_json(record_path, recorded)

    gs_colmap.Dataset.__init__ = init


def patch_cli(resolved_path: Path | None, resolve_only: bool, launcher_info: dict | None = None) -> None:
    import gsplat.distributed as gs_distributed

    original_cli = gs_distributed.cli

    def cli(fn, args, verbose: bool = False):
        if resolved_path is not None:
            _write_json(resolved_path, {**config_to_dict(args), "_launcher": launcher_info or {}})
        if resolve_only:
            print(f"[launcher] resolved config written to {resolved_path}; not training (--resolve-only)")
            return True
        import torch

        if torch.cuda.device_count() > 1:
            raise RuntimeError("more than one GPU is visible; set CUDA_VISIBLE_DEVICES to a single GPU "
                               "(the launcher patches do not reach gsplat's distributed workers)")
        return original_cli(fn, args, verbose=verbose)

    gs_distributed.cli = cli


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split_at = argv.index("--")
        own, trainer = argv[:split_at], argv[split_at + 1:]
    else:
        own, trainer = argv, []
    parser = argparse.ArgumentParser(description="Run gsplat simple_trainer.py with small extensions")
    parser.add_argument("--examples-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--record-split", type=Path, default=None)
    parser.add_argument("--resolved-config", type=Path, default=None)
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args(own)
    if not trainer:
        parser.error("trainer arguments are required after '--'")

    examples = args.examples_dir.resolve()
    script = examples / "simple_trainer.py"
    if not script.is_file():
        parser.error(f"{script} not found")
    split = json.loads(args.split_file.read_text(encoding="utf-8")) if args.split_file else None
    if split is not None and not (split.get("train") and split.get("test")):
        parser.error(f"{args.split_file} must contain non-empty 'train' and 'test' lists")
    if args.resolve_only and args.resolved_config is None:
        parser.error("--resolve-only needs --resolved-config")

    os.chdir(examples)
    sys.path.insert(0, str(examples))
    launcher_info = {"seed": args.seed, "split_file": str(args.split_file) if args.split_file else None,
                     "gsplat": check_installed_gsplat(examples)}
    if args.seed is not None:
        patch_seed(args.seed, examples)
    if split is not None or args.record_split is not None:
        patch_split(split, args.record_split.resolve() if args.record_split else None, examples)
    patch_cli(args.resolved_config.resolve() if args.resolved_config else None, args.resolve_only, launcher_info)

    sys.argv = [str(script), *trainer]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
