"""So sánh các chế độ huấn luyện (vd. default và mcmc) trên cùng các cảnh, kèm kiểm tra điều kiện so sánh.

    python -m indoor3d.eval.compare_modes --root /workspace/outputs/modes_db \\
        --baseline default --modes default mcmc \\
        --reference-config configs/baseline_deepblending.json \\
        --b0-root /workspace/outputs/baseline_inria/db --out report.json --markdown report.md

- Tìm mọi lượt chạy (thư mục có run_meta.json) dưới --root.
- Nhóm theo (cảnh, biến thể). Biến thể = chế độ + tham số riêng (`--set`). Ví dụ "mcmc[opacity-reg=0.001]"
  là biến thể khác "mcmc". Các seed của cùng biến thể được lấy trung bình.
- Lượt chạy có sai khác protocol bị loại khỏi bảng (liệt kê riêng), trừ khi --include-deviating.
  Khi đó chúng thành biến thể riêng, không bao giờ trộn với lượt đúng protocol.
- Baseline là chế độ --baseline không có tham số riêng và không sai khác protocol.

Báo cáo chỉ ghi số liệu, chênh lệch và các mục kiểm tra; không tự kết luận chế độ nào tốt hơn.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from indoor3d.eval.compare_reference import METRICS, SceneResult, compare, load_scene
from indoor3d.eval.gsplat_results import RunResult, load_summary

BUDGET_TOLERANCE = 0.01  # chênh lệch tương đối của số Gaussian cuối vẫn coi là "cùng ngân sách"
MIN_SEEDS = 3  # docs/plan.md mục 4.2


def discover(roots: list[Path]) -> list[RunResult]:
    runs = []
    for root in roots:
        for meta in sorted(Path(root).rglob("run_meta.json")):
            if any(".prev_" in part for part in meta.parent.parts):
                continue  # thư mục cũ do --force / --rerun-incomplete đổi tên
            runs.append(load_summary(meta.parent))
    return runs


def variant_of(run: RunResult) -> str:
    label = run.mode or "?"
    if run.args:
        label += "[" + ", ".join(f"{key}={value}" for key, value in sorted(run.args.items())) + "]"
    if run.deviations:
        label += "{" + ", ".join(f"{key}={item.get('used')}" for key, item in sorted(run.deviations.items())) + "}"
    return label


def mode_of(variant: str) -> str:
    return variant.split("[")[0].split("{")[0]


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _std(values):
    values = [value for value in values if value is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def aggregate(runs: list[RunResult]) -> dict:
    return {
        "n": len(runs),
        "seeds": sorted(run.seed for run in runs if run.seed is not None),
        "metrics": {name: _mean(run.metrics.get(name) for run in runs) for name in METRICS},
        "metrics_std": {name: _std([run.metrics.get(name) for run in runs]) for name in METRICS},
        "num_gaussians": _mean(run.num_gaussians for run in runs),
        "train_minutes": _mean(None if run.train_seconds is None else run.train_seconds / 60 for run in runs),
        "train_loop_minutes": _mean(None if run.train_loop_seconds is None else run.train_loop_seconds / 60
                                    for run in runs),
        "peak_vram_gib": _mean(None if run.peak_vram_mib is None else run.peak_vram_mib / 1024 for run in runs),
        "torch_peak_gib": _mean(run.torch_peak_gib for run in runs),
        "render_ms_per_image": _mean(run.render_ms_per_image for run in runs),
        "ply_size_mb": _mean(run.ply_size_mb for run in runs),
        "run_dirs": [run.run_dir for run in runs],
    }


def _same(values) -> bool:
    return len({json.dumps(value, sort_keys=True) for value in values}) <= 1


def fairness(baseline_runs: list[RunResult], runs: list[RunResult]) -> list[dict]:
    """Các điều kiện phải giống nhau để so sánh một biến thể với baseline trên cùng cảnh."""
    both = baseline_runs + runs
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    deviating = [run.run_dir for run in both if run.deviations]
    add("protocol", "OK" if _same([run.protocol for run in both]) and not deviating else "DIFF",
        {"deviating_runs": deviating} if deviating else "same protocol, no deviations")
    add("split", "OK" if _same([run.split_sha256 for run in both]) else "DIFF",
        sorted({str(run.split_sha256)[:12] for run in both}))
    add("gsplat_commit", "OK" if _same([run.gsplat_commit for run in both]) else "DIFF",
        sorted({str(run.gsplat_commit) for run in both}))
    add("gpu", "OK" if _same([run.gpu for run in both]) else "DIFF", sorted({str(run.gpu) for run in both}))
    applied = [run.split_applied_ok for run in both]
    if any(value is False for value in applied):
        add("split_applied", "DIFF", "gsplat used other images")
    elif any(value is None for value in applied):
        add("split_applied", "UNKNOWN", "split_applied file missing")
    else:
        add("split_applied", "OK", "images used == split.json")
    base_seeds = sorted(run.seed for run in baseline_runs)
    seeds = sorted(run.seed for run in runs)
    if base_seeds != seeds:
        add("seeds", "DIFF", {"baseline": base_seeds, "mode": seeds})
    else:
        add("seeds", "OK" if len(seeds) >= MIN_SEEDS else "FEW", {"seeds": seeds, "min": MIN_SEEDS})
    base_count = _mean(run.num_gaussians for run in baseline_runs)
    count = _mean(run.num_gaussians for run in runs)
    if base_count and count:
        ratio = count / base_count
        add("gaussian_budget", "EQUAL" if abs(ratio - 1) <= BUDGET_TOLERANCE else "UNEQUAL",
            {"baseline": round(base_count), "mode": round(count), "ratio": round(ratio, 4)})
    return checks


def b0_check(report: dict, b0_root: Path, scenes: list[str], method: str, iteration: int,
             tolerance: dict) -> dict:
    """Đối chiếu baseline gsplat với lượt Inria (B0) tự chạy, từng cảnh, cùng tập test."""
    rows, verdict = [], "PASS"
    for scene in scenes:
        b0 = load_scene(scene, Path(b0_root) / scene, method, iteration)
        b1 = report["scenes"].get(scene, {}).get(report["baseline"])
        if b0.status != "OK" or b1 is None:
            rows.append({"scene": scene, "status": "MISSING", "b0": b0.metrics, "b1": b1 and b1["metrics"]})
            verdict = "INCOMPLETE"
            continue
        delta = {name: b1["metrics"][name] - b0.metrics[name] for name in METRICS}
        within = {name: abs(delta[name]) <= tolerance[name] for name in METRICS}
        rows.append({"scene": scene, "status": "OK" if all(within.values()) else "FAIL", "b0": b0.metrics,
                     "b1": b1["metrics"], "delta": delta, "within_tolerance": within})
        if verdict == "PASS" and not all(within.values()):
            verdict = "FAIL"
    return {"b0_root": str(b0_root), "tolerance": tolerance, "scenes": rows, "verdict": verdict}


def build_report(runs: list[RunResult], baseline: str, modes: list[str],
                 reference: dict | None = None, tolerance: dict | None = None,
                 reference_scenes: list[str] | None = None, include_deviating: bool = False) -> dict:
    if baseline not in modes:
        modes = [baseline, *modes]
    grouped: dict[str, dict[str, list[RunResult]]] = {}
    report = {"baseline": baseline, "modes": modes, "variants": [], "scenes": {}, "deltas": {}, "fairness": {},
              "incomplete": [], "excluded": [], "mean_over_scenes": {}, "complete_scenes": []}
    for run in runs:
        if run.mode not in modes:
            continue
        if run.status != "OK":
            report["incomplete"].append({"run_dir": run.run_dir, "scene": run.scene, "mode": run.mode,
                                         "status": run.status})
            continue
        if run.deviations and not include_deviating:
            report["excluded"].append({"run_dir": run.run_dir, "scene": run.scene, "variant": variant_of(run),
                                       "reason": "protocol deviation"})
            continue
        grouped.setdefault(run.scene, {}).setdefault(variant_of(run), []).append(run)

    variants = sorted({variant for by_variant in grouped.values() for variant in by_variant},
                      key=lambda v: (v != baseline, modes.index(mode_of(v)), v))
    report["variants"] = variants
    for scene in sorted(grouped):
        by_variant = grouped[scene]
        report["scenes"][scene] = {variant: aggregate(by_variant[variant]) for variant in variants
                                   if variant in by_variant}
        missing = [variant for variant in variants if variant not in by_variant]
        # chế độ được yêu cầu nhưng không có lượt OK nào (ở bất kỳ biến thể nào) cũng là thiếu
        missing += [mode for mode in modes if mode not in {mode_of(v) for v in by_variant} and mode not in missing]
        if missing:
            report["incomplete"].append({"scene": scene, "missing_variants": missing})
        if baseline not in by_variant:
            continue
        if not missing:
            report["complete_scenes"].append(scene)
        base = report["scenes"][scene][baseline]
        for variant in variants:
            if variant == baseline or variant not in by_variant:
                continue
            current = report["scenes"][scene][variant]
            report["deltas"].setdefault(scene, {})[variant] = {
                **{name: current["metrics"][name] - base["metrics"][name] for name in METRICS},
                "num_gaussians_ratio": (current["num_gaussians"] / base["num_gaussians"]
                                        if current["num_gaussians"] and base["num_gaussians"] else None),
                "train_time_ratio": (current["train_minutes"] / base["train_minutes"]
                                     if current["train_minutes"] and base["train_minutes"] else None),
            }
            report["fairness"].setdefault(scene, {})[variant] = fairness(by_variant[baseline], by_variant[variant])

    for variant in variants:
        values = [report["scenes"][scene][variant]["metrics"] for scene in report["complete_scenes"]]
        if values:
            report["mean_over_scenes"][variant] = {name: statistics.fmean(v[name] for v in values)
                                                   for name in METRICS}

    if reference is not None:
        scenes = reference_scenes or sorted(grouped)
        results = []
        for scene in scenes:
            agg = report["scenes"].get(scene, {}).get(baseline)
            if agg is None:
                results.append(SceneResult(scene=scene, model_dir=""))
            else:
                results.append(SceneResult(scene=scene, model_dir=";".join(agg["run_dirs"]), status="OK",
                                           metrics=dict(agg["metrics"])))
        report["reference_check"] = {"mode": baseline, "scenes": scenes,
                                     **compare(results, reference, tolerance or {})}
    return report


def _f(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _metric(agg: dict, name: str, digits: int) -> str:
    value, std = agg["metrics"][name], agg["metrics_std"][name]
    return _f(value, digits) + ("" if std is None else f" ± {std:.{digits}f}")


def format_markdown(report: dict) -> str:
    baseline = report["baseline"]
    lines = [f"Baseline: `{baseline}`. Số liệu là trung bình ± độ lệch chuẩn theo seed (n = số lượt chạy).", "",
             "| Cảnh | Biến thể | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Số Gaussian | Train (phút) | VRAM đỉnh (GiB) "
             "| Render (ms/ảnh) | PLY (MB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for scene, by_variant in report["scenes"].items():
        for variant, agg in by_variant.items():
            count = "-" if agg["num_gaussians"] is None else f"{agg['num_gaussians']:,.0f}"
            lines.append(f"| {scene} | {variant} | {agg['n']} | {_metric(agg, 'PSNR', 3)} | {_metric(agg, 'SSIM', 4)} "
                         f"| {_metric(agg, 'LPIPS', 4)} | {count} | {_f(agg['train_minutes'], 1)} "
                         f"| {_f(agg['peak_vram_gib'], 1)} | {_f(agg['render_ms_per_image'], 2)} "
                         f"| {_f(agg['ply_size_mb'], 0)} |")
    if report["deltas"]:
        lines += ["", f"Chênh lệch so với `{baseline}` (dương ở PSNR/SSIM và âm ở LPIPS là tốt hơn):", "",
                  "| Cảnh | Biến thể | ΔPSNR | ΔSSIM | ΔLPIPS | Tỉ lệ số Gaussian | Tỉ lệ thời gian | Điều kiện |",
                  "|---|---|---|---|---|---|---|---|"]
        for scene, by_variant in report["deltas"].items():
            for variant, delta in by_variant.items():
                checks = report["fairness"][scene][variant]
                bad = [c["check"] for c in checks if c["status"] in ("DIFF", "UNKNOWN")]
                status = {c["check"]: c["status"] for c in checks}
                verdict = "đủ điều kiện" if not bad else "KHÁC: " + ", ".join(bad)
                verdict += f"; ngân sách {status.get('gaussian_budget', '-')}"
                if status.get("seeds") == "FEW":
                    verdict += f"; < {MIN_SEEDS} seed"
                lines.append(f"| {scene} | {variant} | {delta['PSNR']:+.3f} | {delta['SSIM']:+.4f} "
                             f"| {delta['LPIPS']:+.4f} | {_f(delta['num_gaussians_ratio'], 3)} "
                             f"| {_f(delta['train_time_ratio'], 2)} | {verdict} |")
    if report["mean_over_scenes"]:
        lines += ["", "Trung bình trên các cảnh có đủ mọi biến thể: " + ", ".join(report["complete_scenes"]), ""]
        for variant, m in report["mean_over_scenes"].items():
            lines.append(f"- `{variant}`: PSNR {_f(m['PSNR'])}, SSIM {_f(m['SSIM'])}, LPIPS {_f(m['LPIPS'])}")
    if report.get("reference_check"):
        ref = report["reference_check"]
        lines += ["", f"Đối chiếu `{ref['mode']}` với số liệu công bố ({', '.join(ref['scenes'])}): "
                      f"**{ref['verdict']}**"]
        if ref.get("mean"):
            lines.append("- chênh lệch trung bình: " + ", ".join(f"{k} {v:+.3f}" for k, v in ref["delta"].items()))
    if report.get("b0_check"):
        b0 = report["b0_check"]
        lines += ["", f"Đối chiếu `{baseline}` (gsplat) với lượt Inria tự chạy (B0), từng cảnh: **{b0['verdict']}**"]
        for row in b0["scenes"]:
            if row["status"] == "MISSING":
                lines.append(f"- {row['scene']}: thiếu kết quả")
            else:
                lines.append(f"- {row['scene']}: " + ", ".join(f"Δ{k} {v:+.3f}" for k, v in row["delta"].items())
                             + f" ({row['status']})")
    for title, key in (("Bị loại (sai khác protocol)", "excluded"), ("Chưa đủ / lỗi", "incomplete")):
        if report[key]:
            lines += ["", f"{title}:"] + [f"- {json.dumps(item, ensure_ascii=False)}" for item in report[key]]
    lines += ["", "Ghi chú: chỉ nói \"cải thiện\" khi mọi mục điều kiện là OK, ngân sách Gaussian EQUAL, "
                  f"cùng bộ ≥ {MIN_SEEDS} seed và kết quả nhất quán trên các cảnh "
                  "(docs/design/training_modes.md, mục 4.3)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path, action="append",
                        help="thư mục chứa các lượt chạy (lặp lại được)")
    parser.add_argument("--baseline", default="default")
    parser.add_argument("--modes", nargs="+", default=["default", "mcmc"])
    parser.add_argument("--include-deviating", action="store_true",
                        help="giữ lượt sai khác protocol thành biến thể riêng thay vì loại bỏ")
    parser.add_argument("--reference-config", type=Path,
                        help="file cấu hình có reference/tolerance/scenes (vd. configs/baseline_deepblending.json)")
    parser.add_argument("--b0-root", type=Path,
                        help="thư mục kết quả Inria tự chạy (mỗi cảnh một thư mục có results.json) để đối chiếu từng cảnh")
    parser.add_argument("--out", type=Path, help="ghi báo cáo JSON")
    parser.add_argument("--markdown", type=Path, help="ghi bảng Markdown")
    args = parser.parse_args(argv)

    config = json.loads(args.reference_config.read_text(encoding="utf-8")) if args.reference_config else None
    reference = config["reference"]["metrics"] if config else None
    tolerance = config["tolerance"] if config else None
    scenes = config.get("scenes") if config else None
    report = build_report(discover(args.root), args.baseline, args.modes, reference, tolerance, scenes,
                          args.include_deviating)
    if args.b0_root:
        if config is None:
            parser.error("--b0-root needs --reference-config (scenes, method, iteration, tolerance)")
        report["b0_check"] = b0_check(report, args.b0_root, scenes or sorted(report["scenes"]),
                                      config.get("method", "ours_30000"), int(config.get("iteration", 30000)),
                                      tolerance)
    text = format_markdown(report)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nĐã ghi: {args.out}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
