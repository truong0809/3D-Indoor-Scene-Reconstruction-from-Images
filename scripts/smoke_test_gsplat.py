"""Smoke test cho env gs-gsplat trên GPU (gọi ở bước cuối của setup_gsplat.sh).

Kiểm tra:
1. gsplat có extension CUDA biên dịch sẵn (gsplat.csrc), không phải biên dịch JIT lúc chạy.
2. rasterization chạy xuôi/ngược; DefaultStrategy và MCMCStrategy chạy vài bước (kể cả nhân relocation).
3. fused_ssim: SSIM(x, x) ~ 1 và có gradient.
4. LPIPS (VGG, như Inria) tính được; lần đầu sẽ tải trọng số.
5. examples/simple_trainer.py import được và phân giải đúng tham số của chế độ default và mcmc.

    python scripts/smoke_test_gsplat.py --examples-dir /workspace/code/gsplat/examples
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool]] = []


def check(name):
    def wrap(fn):
        def run(*args, **kwargs):
            print(f"\n--- {name}")
            try:
                fn(*args, **kwargs)
            except Exception:  # noqa: BLE001 - báo lỗi rồi chạy tiếp các mục khác
                traceback.print_exc()
                RESULTS.append((name, False))
                print(f"FAIL {name}")
            else:
                RESULTS.append((name, True))
                print(f"PASS {name}")
        return run
    return wrap


@check("gsplat compiled extension")
def check_extension():
    import importlib

    import gsplat

    importlib.import_module("gsplat.csrc")  # chỉ có khi extension được build sẵn bằng setup.py
    print("gsplat", gsplat.__version__, "|", gsplat.__file__)


def make_params(n: int, opacity: float, device: str):
    torch.manual_seed(0)
    quats = F.normalize(torch.randn(n, 4, device=device), dim=-1)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.randn(n, 3, device=device) * 0.5 + torch.tensor([0.0, 0.0, 3.0], device=device)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.03), device=device)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(opacity / (1 - opacity)), device=device)),
        "sh0": torch.nn.Parameter(torch.rand(n, 1, 3, device=device) * 0.5),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })


@check("rasterization + DefaultStrategy + MCMCStrategy")
def check_strategies():
    from gsplat import DefaultStrategy, MCMCStrategy, rasterization

    device = "cuda"
    width, height, n = 320, 240, 3000
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor([[[300.0, 0.0, width / 2], [0.0, 300.0, height / 2], [0.0, 0.0, 1.0]]], device=device)
    target = torch.rand(1, height, width, 3, device=device)
    strategies = {
        # ngưỡng 0 để chắc chắn có nhân bản / tách Gaussian trong vài bước
        "default": DefaultStrategy(refine_start_iter=0, refine_every=2, grow_grad2d=0.0, verbose=True),
        "mcmc": MCMCStrategy(cap_max=int(n * 1.2), refine_start_iter=0, refine_every=2, verbose=True),
    }
    for name, strategy in strategies.items():
        params = make_params(n, 0.1 if name == "default" else 0.5, device)
        optimizers = {key: torch.optim.Adam([value], lr=1e-3) for key, value in params.items()}
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=1.0) if name == "default" else strategy.initialize_state()
        start = len(params["means"])
        for step in range(7):
            colors = torch.cat([params["sh0"], params["shN"]], dim=1)
            render, _, info = rasterization(
                params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                colors, viewmats, Ks, width, height, sh_degree=3, packed=False)
            if name == "default":
                strategy.step_pre_backward(params, optimizers, state, step, info)
            loss = (render - target).abs().mean()
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if name == "default":
                strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
            else:
                strategy.step_post_backward(params, optimizers, state, step, info, lr=1e-3)
        end = len(params["means"])
        print(f"{name}: {start} -> {end} Gaussians, last loss {loss.item():.4f}")
        assert torch.isfinite(loss), "loss is not finite"
        assert end > start, f"{name} strategy did not add Gaussians"
        if name == "mcmc":
            assert end <= strategy.cap_max, "MCMC exceeded cap_max"


@check("fused_ssim")
def check_fused_ssim():
    from fused_ssim import fused_ssim

    x = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    value = fused_ssim(x, x.detach())
    assert abs(value.item() - 1.0) < 1e-3, f"SSIM(x, x) = {value.item()}"
    other = torch.rand(1, 3, 64, 64, device="cuda")
    (1 - fused_ssim(x, other)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@check("LPIPS (VGG) from torchmetrics")
def check_lpips():
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to("cuda")
    a = torch.rand(1, 3, 64, 64, device="cuda")
    assert lpips(a, a).item() < 1e-6
    print("LPIPS(a, b) =", lpips(a, torch.rand_like(a)).item())


@check("simple_trainer.py arguments for default and mcmc")
def check_trainer_cli(examples_dir: Path):
    sys.path.insert(0, str(REPO_DIR / "src"))
    from indoor3d.train.profiles import load_config, resolve_mode, trainer_args

    config = load_config(REPO_DIR / "configs" / "train_modes.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_DIR / "src"), env.get("PYTHONPATH")]))
    with tempfile.TemporaryDirectory() as tmp:
        for mode, cap in (("default", None), ("mcmc", 1_000_000)):
            spec = resolve_mode(config, mode)
            out = Path(tmp) / f"{mode}.json"
            cmd = [sys.executable, "-m", "indoor3d.train.gsplat_launcher", "--examples-dir", str(examples_dir),
                   "--resolve-only", "--resolved-config", str(out), "--",
                   *trainer_args(spec, Path(tmp) / "scene", Path(tmp) / mode, cap)]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout[-2000:] + proc.stderr[-4000:])
            cfg = json.loads(out.read_text(encoding="utf-8"))
            expected = "DefaultStrategy" if mode == "default" else "MCMCStrategy"
            assert cfg["strategy_type"] == expected, cfg["strategy_type"]
            assert cfg["max_steps"] == spec.max_steps and cfg["lpips_net"] == spec.protocol["lpips_net"]
            if cap is not None:
                assert cfg["strategy"]["cap_max"] == cap
            print(f"{mode}: {expected}, max_steps {cfg['max_steps']}, lpips {cfg['lpips_net']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples-dir", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("FAIL: PyTorch cannot see a GPU")
        return 1
    print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
    check_extension()
    check_strategies()
    check_fused_ssim()
    check_lpips()
    check_trainer_cli(args.examples_dir.resolve())
    print("\n===== SUMMARY =====")
    for name, ok in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    passed = all(ok for _, ok in RESULTS)
    print("SMOKE TEST", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
