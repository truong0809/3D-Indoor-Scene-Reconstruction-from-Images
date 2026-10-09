"""Test so sánh các chế độ huấn luyện. Mọi con số là dữ liệu giả để kiểm tra logic, KHÔNG phải kết quả."""

import json

import pytest

from indoor3d.eval.compare_modes import b0_check, build_report, discover, format_markdown, main

PROTOCOL = {"data_factor": 1, "test_every": 8, "max_steps": 30000, "lpips_net": "vgg",
            "antialiased": False, "save_ply": True, "disable_video": False}
REFERENCE = {"PSNR": 29.690, "SSIM": 0.906, "LPIPS": 0.238}
TOLERANCE = {"PSNR": 0.5, "SSIM": 0.01, "LPIPS": 0.01}


def write_run(root, scene, mode, seed, psnr, *, count=2_000_000, split="s1", deviations=None,
              protocol=None, gpu="RTX A6000", status="OK", args=None, tag=""):
    run = root / scene / f"{mode}{tag}_s{seed}"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text("{}", encoding="utf-8")
    summary = {"run_dir": str(run), "scene": scene, "mode": mode, "seed": seed, "status": status,
               "metrics": {"PSNR": psnr, "SSIM": 0.9, "LPIPS": 0.25}, "num_gaussians": count,
               "train_seconds": 1800.0, "peak_vram_mib": 10240.0, "protocol": protocol or PROTOCOL,
               "deviations": deviations or {}, "split_sha256": split, "split_applied_ok": True,
               "gsplat_commit": "6e8c837", "gpu": gpu, "args": args or {}}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def checks(report, scene, mode):
    return {c["check"]: c["status"] for c in report["fairness"][scene][mode]}


def test_fair_comparison_with_seeds(tmp_path):
    for scene in ("drjohnson", "playroom"):
        for seed, (base, mcmc) in zip((42, 43), ((29.5, 29.7), (29.7, 29.9))):
            write_run(tmp_path, scene, "default", seed, base)
            write_run(tmp_path, scene, "mcmc", seed, mcmc)
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    assert report["complete_scenes"] == ["drjohnson", "playroom"]
    agg = report["scenes"]["playroom"]["mcmc"]
    assert agg["n"] == 2 and agg["seeds"] == [42, 43]
    assert agg["metrics"]["PSNR"] == pytest.approx(29.8)
    assert agg["metrics_std"]["PSNR"] == pytest.approx(0.1414, abs=1e-3)
    assert report["deltas"]["playroom"]["mcmc"]["PSNR"] == pytest.approx(0.2)
    assert checks(report, "playroom", "mcmc") == {"protocol": "OK", "split": "OK", "gsplat_commit": "OK",
                                                   "gpu": "OK", "split_applied": "OK", "seeds": "FEW",
                                                   "gaussian_budget": "EQUAL"}
    assert report["mean_over_scenes"]["default"]["PSNR"] == pytest.approx(29.6)
    text = format_markdown(report)
    assert "đủ điều kiện" in text and "ngân sách EQUAL" in text and "< 3 seed" in text
    assert "29.800 ± 0.141" in text


def test_unfair_runs_are_flagged(tmp_path):
    write_run(tmp_path, "room", "default", 42, 30.0)
    write_run(tmp_path, "room", "mcmc", 42, 30.5, split="other", gpu="RTX 4090")
    write_run(tmp_path, "room", "mcmc", 43, 30.4)
    deviation = {"max_steps": {"base": 30000, "used": 40000, "source": "mode"}}
    write_run(tmp_path, "room", "mcmc_demo", 42, 31.0, count=3_500_000, deviations=deviation,
              protocol={**PROTOCOL, "max_steps": 40000})
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc", "mcmc_demo"])
    mcmc = checks(report, "room", "mcmc")
    assert mcmc["split"] == "DIFF" and mcmc["gpu"] == "DIFF" and mcmc["protocol"] == "OK"
    assert mcmc["seeds"] == "DIFF"  # baseline [42], mcmc [42, 43]
    assert "KHÁC" in format_markdown(report)
    # lượt sai khác protocol bị loại mặc định, không trộn vào bảng
    assert report["excluded"][0]["variant"] == "mcmc_demo{max_steps=40000}"
    assert "mcmc_demo" not in json.dumps(report["scenes"])
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc_demo"], include_deviating=True)
    demo = checks(report, "room", "mcmc_demo{max_steps=40000}")
    assert demo["protocol"] == "DIFF" and demo["gaussian_budget"] == "UNEQUAL"


def test_mode_arguments_make_separate_variants(tmp_path):
    write_run(tmp_path, "playroom", "default", 42, 30.0)
    write_run(tmp_path, "playroom", "default", 42, 31.0, args={"strategy.absgrad": True}, tag="_abs")
    write_run(tmp_path, "playroom", "mcmc", 42, 30.2, args={"opacity-reg": 0.001})
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    variants = report["scenes"]["playroom"]
    assert variants["default"]["n"] == 1 and variants["default"]["metrics"]["PSNR"] == 30.0  # không bị trộn
    assert "default[strategy.absgrad=True]" in variants and "mcmc[opacity-reg=0.001]" in variants
    assert report["deltas"]["playroom"]["mcmc[opacity-reg=0.001]"]["PSNR"] == pytest.approx(0.2)


def test_incomplete_and_failed_runs(tmp_path):
    write_run(tmp_path, "a", "default", 42, 30.0)
    write_run(tmp_path, "b", "default", 42, 30.0)
    write_run(tmp_path, "b", "mcmc", 42, 30.0, status="FAILED")
    report = build_report(discover([tmp_path]), "default", ["default", "mcmc"])
    assert report["complete_scenes"] == []
    assert {"scene": "a", "missing_variants": ["mcmc"]} in report["incomplete"]
    assert any(item.get("status") == "FAILED" for item in report["incomplete"])


def test_backup_folders_are_ignored(tmp_path):
    write_run(tmp_path, "a", "default", 42, 30.0)
    old = write_run(tmp_path / "x", "a", "default", 42, 20.0)
    old.rename(old.with_name(old.name + ".prev_20261009_120000"))
    runs = discover([tmp_path])
    assert len(runs) == 1 and runs[0].metrics["PSNR"] == 30.0


def test_reference_check_and_cli(tmp_path):
    write_run(tmp_path, "drjohnson", "default", 42, 29.4)
    write_run(tmp_path, "playroom", "default", 42, 30.0)
    report = build_report(discover([tmp_path]), "default", ["default"], REFERENCE, TOLERANCE,
                          ["drjohnson", "playroom"])
    assert report["reference_check"]["verdict"] == "FAIL"  # LPIPS giả 0.25 lệch 0.012 > ngưỡng 0.01
    assert report["reference_check"]["delta"]["PSNR"] == pytest.approx(0.01)

    config = tmp_path / "ref.json"
    config.write_text(json.dumps({"reference": {"metrics": REFERENCE}, "tolerance": TOLERANCE,
                                  "scenes": ["drjohnson", "playroom", "missing"]}), encoding="utf-8")
    out, markdown = tmp_path / "report.json", tmp_path / "report.md"
    assert main(["--root", str(tmp_path), "--modes", "default", "--reference-config", str(config),
                 "--out", str(out), "--markdown", str(markdown)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["reference_check"]["verdict"] == "INCOMPLETE"  # thiếu cảnh "missing"
    assert "INCOMPLETE" in markdown.read_text(encoding="utf-8")


def write_b0(root, scene, psnr, ssim=0.905, lpips=0.24):
    model = root / scene
    model.mkdir(parents=True)
    (model / "results.json").write_text(json.dumps({"ours_30000": {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips}}),
                                        encoding="utf-8")


def test_b0_paired_check(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "drjohnson", "default", 42, 29.30)
    write_run(runs, "playroom", "default", 42, 30.10)
    b0 = tmp_path / "b0"
    write_b0(b0, "drjohnson", 29.20, ssim=0.9, lpips=0.25)
    write_b0(b0, "playroom", 30.00, ssim=0.9, lpips=0.25)
    report = build_report(discover([runs]), "default", ["default"])
    result = b0_check(report, b0, ["drjohnson", "playroom"], "ours_30000", 30000, TOLERANCE)
    assert result["verdict"] == "PASS"
    assert result["scenes"][0]["delta"]["PSNR"] == pytest.approx(0.1)
    write_b0(b0, "room", 31.0)
    assert b0_check(report, b0, ["drjohnson", "room"], "ours_30000", 30000, TOLERANCE)["verdict"] == "INCOMPLETE"
    (b0 / "playroom" / "results.json").write_text(json.dumps({"ours_30000": {"PSNR": 31.0, "SSIM": 0.9,
                                                                             "LPIPS": 0.25}}))
    assert b0_check(report, b0, ["drjohnson", "playroom"], "ours_30000", 30000, TOLERANCE)["verdict"] == "FAIL"
