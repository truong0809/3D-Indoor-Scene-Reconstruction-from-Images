"""Test cấu hình chế độ huấn luyện và tham số dòng lệnh sinh cho simple_trainer.py của gsplat.

Dùng file cấu hình thật configs/train_modes.json; các số Gaussian là dữ liệu giả.
"""

from pathlib import Path

import pytest

from indoor3d.train.profiles import (
    checkpoint_path,
    eval_args,
    load_config,
    normalize_flag,
    parse_assignments,
    resolve_cap_max,
    resolve_mode,
    to_cli,
    trainer_args,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train_modes.json"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def flag_value(argv, flag):
    index = argv.index(flag)
    return argv[index + 1]


def baseline_summary(config, count=2_500_000, mode="default", status="OK", seed=42, deviations=None, protocol=None):
    return {"mode": mode, "status": status, "num_gaussians": count, "run_dir": "/x/default", "seed": seed,
            "split_sha256": "abc", "protocol": protocol or dict(config["protocol"]), "deviations": deviations or {}}


def test_real_config_defines_both_modes(config):
    assert config["baseline_mode"] == "default"
    assert {"default", "mcmc"} <= set(config["modes"])
    assert config["protocol"]["lpips_net"] == "vgg"  # cùng mạng LPIPS với Inria
    assert len(config["trainer"]["commit"]) == 40


def test_default_mode_arguments(config):
    spec = resolve_mode(config, "default")
    argv = trainer_args(spec, Path("/data/scene"), Path("/out/run"))
    assert argv[0] == "default"
    assert flag_value(argv, "--data-dir") == "/data/scene"
    assert flag_value(argv, "--result-dir") == "/out/run"
    assert flag_value(argv, "--data-factor") == "1"
    assert flag_value(argv, "--test-every") == "8"
    assert flag_value(argv, "--max-steps") == "30000"
    assert flag_value(argv, "--lpips-net") == "vgg"
    assert flag_value(argv, "--eval-steps") == "-1"
    assert flag_value(argv, "--save-steps") == "30000"
    assert flag_value(argv, "--ply-steps") == "30000"
    assert "--no-antialiased" in argv and "--save-ply" in argv and "--no-disable-video" in argv
    assert flag_value(argv, "--init-type") == "sfm"  # cả hai chế độ khởi tạo từ điểm SfM
    assert "--disable-viewer" in argv
    assert not any(token.startswith("--strategy.") for token in argv)
    assert spec.deviations == {} and spec.final_step == 29999


def test_mcmc_needs_baseline_budget(config):
    spec = resolve_mode(config, "mcmc")
    with pytest.raises(ValueError, match="cap-max-from"):
        resolve_cap_max(spec)
    cap, source = resolve_cap_max(spec, baseline_summary=baseline_summary(config, 2_345_678), seed=42)
    assert cap == 2_345_678 and source["policy"] == "match_baseline" and source["baseline_seed"] == 42
    argv = trainer_args(spec, Path("/s"), Path("/o"), cap)
    assert argv[0] == "mcmc" and flag_value(argv, "--strategy.cap-max") == "2345678"
    with pytest.raises(ValueError, match="expected 'default'"):
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, mode="mcmc"))
    with pytest.raises(ValueError, match="status must be OK"):
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, status="FAILED"))
    with pytest.raises(ValueError, match="different protocol"):  # vd. lượt chạy thử 3000 vòng
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, protocol={**config["protocol"],
                                                                                  "max_steps": 3000}))
    with pytest.raises(ValueError, match="deviates"):
        resolve_cap_max(spec, baseline_summary=baseline_summary(
            config, deviations={"max_steps": {"base": 30000, "used": 30000, "source": "cli"}}))
    with pytest.raises(ValueError, match="seed 43"):
        resolve_cap_max(spec, baseline_summary=baseline_summary(config, seed=43), seed=42)
    cap, source = resolve_cap_max(spec, cli_value=1_000_000)
    assert cap == 1_000_000 and source == {"policy": "cli", "value": 1_000_000}


def test_mcmc_demo_reproduces_notebook_and_is_marked(config):
    spec = resolve_mode(config, "mcmc_demo")
    assert set(spec.deviations) == {"max_steps", "antialiased"}
    cap, source = resolve_cap_max(spec)
    assert cap == 3_500_000 and source["policy"] == "fixed"
    argv = trainer_args(spec, Path("/s"), Path("/o"), cap)
    assert flag_value(argv, "--max-steps") == "40000"
    assert flag_value(argv, "--strategy.refine-stop-iter") == "35000"
    assert flag_value(argv, "--opacity-reg") == "0.01" and flag_value(argv, "--scale-reg") == "0.01"
    assert "--antialiased" in argv and flag_value(argv, "--save-steps") == "40000"
    assert checkpoint_path(spec, Path("/o")) == Path("/o/ckpts/ckpt_39999_rank0.pt")


def test_eval_arguments_reuse_training_configuration(config):
    spec = resolve_mode(config, "mcmc")
    train = trainer_args(spec, Path("/s"), Path("/o"), 123)
    evaluation = eval_args(spec, Path("/s"), Path("/o"), 123)
    assert evaluation[:len(train)] == train
    assert evaluation[len(train):] == ["--ckpt", "/o/ckpts/ckpt_29999_rank0.pt"]


def test_reserved_flags_and_protocol_overrides(config):
    for key in ("max_steps", "strategy.cap_max", "--data-dir", "lpips_net", "batch_size"):
        with pytest.raises(ValueError, match="controlled by the protocol"):
            resolve_mode(config, "default", set_args={key: 1})
    spec = resolve_mode(config, "mcmc", set_args={"opacity_reg": 0.001},
                        protocol_overrides={"max_steps": 3000})
    assert spec.args == {"opacity-reg": 0.001}
    assert spec.deviations == {"max_steps": {"base": 30000, "used": 3000, "source": "cli"}}
    same = resolve_mode(config, "mcmc_demo", protocol_overrides={"max_steps": 30000})
    assert set(same.deviations) == {"antialiased"}  # đặt lại đúng giá trị gốc thì không còn là sai khác
    with pytest.raises(ValueError, match="unknown protocol key"):
        resolve_mode(config, "default", protocol_overrides={"steps": 1})
    with pytest.raises(ValueError, match="integer"):
        resolve_mode(config, "default", protocol_overrides={"max_steps": "many"})
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_mode(config, "nope")


def test_cli_helpers():
    assert normalize_flag("--strategy.refine_stop_iter") == "strategy.refine-stop-iter"
    assert to_cli("strategy.verbose", False) == ["--strategy.no-verbose"]
    assert to_cli("antialiased", False) == ["--no-antialiased"]
    assert to_cli("save-ply", True) == ["--save-ply"]
    assert to_cli("eval-steps", [7000, 30000]) == ["--eval-steps", "7000", "30000"]
    assert parse_assignments(["a=1", "b=0.5", "c=true", "d=vgg", "e=-2"]) == {
        "a": 1, "b": 0.5, "c": True, "d": "vgg", "e": -2}
    with pytest.raises(ValueError):
        parse_assignments(["novalue"])
    with pytest.raises(ValueError):
        normalize_flag("bad flag")
