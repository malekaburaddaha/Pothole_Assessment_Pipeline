"""Paper 2 result figures, generated from the result JSONs so every point traces
to a produced artefact.

  fig_frontier.pdf   quality-cost frontier: agreement with human consensus vs
                     latency per pothole, log x-axis, cloud vs local engines
  fig_safety.pdf     advisory safety: lane-departure rate and judge ratings,
                     unconstrained vs constrained decoding
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANN = Path(os.environ.get("POTHOLE_ROOT", "."), "Annotations_Results")
RES = Path(os.environ.get("POTHOLE_ROOT", "."), "Paper2_Improvements/results")
OUT = Path(os.environ.get("POTHOLE_ROOT", "."), "Paper2_Improvements/figures")
from matplotlib import font_manager as fm
_FONT_DIR = os.environ.get("ROBOTO_DIR", "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF")
for _w in ("Regular", "Medium", "Bold", "Italic"):
    try:
        fm.fontManager.addfont(f"{_FONT_DIR}/Roboto-{_w}.ttf")
    except (FileNotFoundError, OSError):  # falls back to the matplotlib default font
        pass
plt.rcParams.update({"font.family": "Roboto", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def frontier():
    """Quality-cost frontier.

    Every point is READ from a results file. The earlier version of this figure
    hard-coded the distilled ensemble at kappa 0.404 - a number later withdrawn
    when deterministic retraining showed it was a favourable training draw
    (0.268 +/- 0.056 across independent ensembles). A hand-typed figure cannot go
    stale visibly, which is exactly why it must not be hand-typed.
    """
    ev = json.loads((RES / "ensemble_variance.json").read_text())["summary"]
    ens = ev["ensemble_kappa_across_independent_ensembles"]
    single = ev["single_model_kappa"]

    def vlm(tag):
        return json.loads((RES / f"depth_eval_{tag}.json").read_text())["metrics"][
            "kappa_vs_human_consensus"]

    # (label, latency_ms, kappa, yerr, colour, marker, offset, ha)
    pts = [
        # Cloud point = the released cloud labels (gemini-2_5-pro_full_dataset.json), which are
        # the cloud pipeline's actual outputs and the distillation teacher: kappa 0.321 on the
        # 38-item set. The 0.479 used earlier came from an internal spreadsheet column that the
        # Paper-1 revision corrected (it differs from the released labels on 8 of 38 items).
        ("Gemini 2.5 Pro (cloud)", 4800, 0.321, None, "#c0392b", "o", (-12, 12), "right"),
        ("Qwen3-VL-8B", 14000, vlm("qwen3vl8b-resized-rf"), None,
         "#7f8c8d", "s", (-12, 6), "right"),
        ("Qwen3-VL-30B-A3B FP8", 13000, vlm("qwen3vl30b-fp8-resized-rf"), None,
         "#95a5a6", "s", (-12, -14), "right"),
        ("Distilled single", 4.4, single["mean"], single["sd"],
         "#2980b9", "^", (0, -22), "center"),
        ("Distilled ensemble\n(deployed)", 28, ens["mean"], ens["sd"],
         "#1a5490", "*", (12, 6), "left"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.axhline(0.453, ls="--", lw=1, color="#7f8c8d", zorder=1)
    ax.text(2.6, 0.462, "mean pairwise human agreement", fontsize=7.2,
            color="#7f8c8d", va="bottom")
    for name, ms, k, err, c, m, off, ha in pts:
        if err:
            # Error bars are the SPREAD ACROSS INDEPENDENT TRAINING RUNS, not an
            # item bootstrap. Training variance is the larger term here, and the
            # figure would misrepresent the result if it showed the smaller one.
            ax.errorbar(ms, k, yerr=err, fmt="none", ecolor=c, elinewidth=1.2,
                        capsize=3, zorder=2)
        ax.scatter(ms, k, s=260 if m == "*" else 85, c=c, marker=m, zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(name, (ms, k), textcoords="offset points", xytext=off,
                    ha=ha, va="center", fontsize=7.6, color=c, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Latency per pothole (ms, log scale)")
    ax.set_ylabel(r"Agreement with human consensus ($\kappa$)")
    ax.set_ylim(0.0, 0.55)
    ax.set_xlim(2, 60000)
    ax.set_title("Severity engines: quality against cost", fontsize=10, pad=8)
    ax.grid(alpha=0.22, ls=":")
    fig.text(0.5, -0.03, "Error bars: standard deviation across independent "
             "training runs (n=3 ensembles, 18 models).", ha="center", fontsize=6.8,
             color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "fig_frontier.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_frontier.png", dpi=300, bbox_inches="tight")
    print(f"wrote fig_frontier  (ensemble kappa {ens['mean']} +/- {ens['sd']})")


def safety():
    u = json.loads((RES / "advisory_safety.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    # (a) lane-departure rate
    labels = ["Cloud\n(Paper 1)", "Local\nunconstrained", "Local\nconstrained"]
    vals = [u["lane_departure_pct"]["cloud"], u["lane_departure_pct"]["local_unconstrained"],
            u["lane_departure_pct"]["local_constrained"]]
    cols = ["#c0392b", "#e67e22", "#27ae60"]
    axes[0].bar(labels, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("Scenes advising a lane departure (%)")
    axes[0].set_ylim(0, max(vals) * 1.25 + 4)
    axes[0].set_title("(a) Unsupportable advice", fontsize=9.5)
    axes[0].grid(axis="y", alpha=0.25, ls=":")
    # (b) judge vs HUMAN, on the system the human study actually rated.
    #
    # The earlier version of this panel plotted judge ratings alone, which buries
    # the paper's methodological finding: the judge is not merely noisy, it is
    # wrong by a wide margin in a consistent direction. Humans rated the local
    # system's safety 3.95; the judge rated the same system 2.43. Plotting them
    # side by side is the point of the panel.
    #
    # The human study rated the UNCONSTRAINED local generator (see the provenance
    # warning in v1_paper2_by_system.json), so the local bars here are the
    # unconstrained system for both raters - a like-for-like comparison.
    j = u["judge"]
    hp = json.loads((ANN / "v1_paper2_by_system.json").read_text())["per_system"]
    # Same pairing as the human study (unconstrained local generator), read from the
    # original judge run so the panel matches the human-vs-judge table exactly.
    jm = json.loads((RES / "llm_judge_warnings.json").read_text())["mean_ratings"]
    jc = {"clarity": jm["cloud_clarity"], "safety": jm["cloud_safety"]}
    ju = {"clarity": jm["local_clarity"], "safety": jm["local_safety"]}
    x = np.arange(2)
    w = 0.2
    axes[1].bar(x - 1.5 * w, [jc["clarity"], jc["safety"]], w,
                label="Cloud, judge", color="#c0392b", alpha=0.55)
    axes[1].bar(x - 0.5 * w, [hp["cloud"]["clarity_mean"], hp["cloud"]["safety_mean"]], w,
                label="Cloud, human", color="#c0392b")
    axes[1].bar(x + 0.5 * w, [ju["clarity"], ju["safety"]], w,
                label="Local, judge", color="#e67e22", alpha=0.55)
    axes[1].bar(x + 1.5 * w, [hp["local"]["clarity_mean"], hp["local"]["safety_mean"]], w,
                label="Local, human", color="#e67e22")
    # annotate the gap that matters
    gap = hp["local"]["safety_mean"] - ju["safety"]
    axes[1].annotate("", xy=(1 + 1.5 * w, hp["local"]["safety_mean"]),
                     xytext=(1 + 0.5 * w, ju["safety"]),
                     arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0))
    axes[1].text(1 + 1.0 * w, (hp["local"]["safety_mean"] + ju["safety"]) / 2,
                 f"  {gap:+.2f}", fontsize=8, fontweight="bold", va="center")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Clarity", "Safety"])
    axes[1].set_ylabel("Rating (1--5)")
    axes[1].set_ylim(0, 5.9)
    axes[1].legend(fontsize=6.2, frameon=False, loc="upper right", ncol=2)
    axes[1].set_title("(b) Automated judge vs. human raters", fontsize=9.5)
    axes[1].grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig_safety.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_safety.png", dpi=300, bbox_inches="tight")
    print("wrote fig_safety")


def segmentation():
    """Stage-2 candidates: mask fidelity against the published ViT-H masks vs time."""
    spec = [("ViT-H fp32 (published)", "baseline", "#2d3748", "o", False, None, (0, 11)),
            ("MobileSAM", "mobilesam", "#c0392b", "s", False, "1", (1, -10)),
            ("TinySAM", "tinysam", "#c0392b", "s", False, "2", (-8, 6)),
            ("EfficientViT-L2", "efficientvit-l2", "#c0392b", "s", False, "3", (-9, -2)),
            ("EfficientViT-XL1", "efficientvit-xl1", "#c0392b", "s", False, "4", (9, 3)),
            ("SAM 2.1 small", "sam2_1-small", "#c0392b", "s", False, "5", (9, 0)),
            ("TinySAM windowed", "windowed_tinysam", "#c0392b", "s", True, "6", (-9, 3)),
            ("ViT-H windowed", "windowed_vith", "#2d3748", "o", True, "7", (9, 0)),
            ("ViT-H fp16", "vith-fp16", "#1e8449", "D", False, None, (22, -13)),
            ("ViT-H fp16 + compile", "vith-fp16-comp", "#1e8449", "*", False, None, (-4, 16))]
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.axhline(0.85, ls="--", lw=1, color="#7f8c8d", zorder=1)
    ax.text(1.25, 0.857, "acceptance threshold (mean IoU 0.85)", fontsize=7.2, color="#7f8c8d",
            ha="right", va="bottom")
    legend = []
    for name, tag, c, m, hollow, num, off in spec:
        d = json.loads((RES / f"bench_stage2_{tag}.json").read_text())
        t = d["timing"]["t_total"]["mean"] if "t_total" in d["timing"] else d["timing"]["mean"]
        if tag == "baseline":
            iou, flip = 1.0, 0.0
        else:
            key = next(k for k in d if k.startswith("fidelity_iou"))
            iou, flip = d[key]["mean"], d["risk_category_flip_rate"]
        ax.scatter(t, iou, s=230 if m == "*" else 70, marker=m, zorder=3,
                   facecolors="white" if hollow else c, edgecolors=c, linewidths=1.5)
        if num:
            ax.annotate(num, (t, iou), textcoords="offset points", xytext=off, ha="center",
                        va="center", fontsize=7.2, color=c, fontweight="bold", zorder=4)
            legend.append(f"{num}  {name}   {iou:.2f}  {100 * flip:.1f}%")
        else:
            ax.annotate(name + ("  (adopted)" if tag == "vith-fp16-comp" else ""), (t, iou),
                        textcoords="offset points", xytext=off, ha="center", va="center",
                        fontsize=7.2, color=c, fontweight="bold" if m == "*" else "normal", zorder=4)
    ax.text(0.50, 0.28, "IoU · risk flips\n" + "\n".join(legend), transform=ax.transAxes,
            fontsize=6.6, va="center", color="#c0392b", linespacing=1.4)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 1.3)
    ax.set_ylim(0.66, 1.06)
    ax.set_xlabel("Stage-2 time per scene (s, log scale)")
    ax.set_ylabel("Mean mask IoU against published ViT-H")
    ax.set_title("Segmentation: efficient variants lose fidelity, half precision does not",
                 fontsize=9.5, pad=8)
    ax.grid(alpha=0.22, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig_segmentation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_segmentation.png", dpi=300, bbox_inches="tight")
    print("wrote fig_segmentation")


def domain_shift():
    """Out-of-domain severity on PothRGBD (151 potholes, class-balanced by measured depth)."""
    from scipy.stats import spearmanr
    dm = json.loads((RES / "distilled_vs_metric.json").read_text())
    sa = json.loads((RES / "distilled_vs_metric_scaleaug.json").read_text())
    gm = json.loads((ROOT / "Paper1_Resubmission/m1_vlm_vs_metric.json").read_text())
    order = {"low": 0, "medium": 1, "high": 2}
    rho_c = spearmanr([order[p["vlm_category"]] for p in gm["predictions"]],
                      [p["metric_mm"] for p in gm["predictions"]])
    dist = {"Measured\ndepth": gm["metrics"]["metric_distribution"],
            "Cloud VLM": gm["metrics"]["vlm_distribution"],
            "Distilled\nensemble": dm["metrics"]["prediction_distribution"],
            "Distilled +\nscale aug.": sa["metrics"]["prediction_distribution"]}
    kap = [gm["metrics"]["kappa"], dm["metrics"]["kappa"], sa["metrics"]["kappa"]]
    rho = [rho_c[0], dm["metrics"]["spearman_pred_vs_mm"][0], sa["metrics"]["spearman_pred_vs_mm"][0]]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
    cols = {"low": "#2e9e5b", "medium": "#e0a21b", "high": "#c0392b"}
    names = list(dist)
    bottom = np.zeros(len(names))
    for cls in ("low", "medium", "high"):
        v = np.array([dist[n].get(cls, 0) for n in names], float) / 151 * 100
        axes[0].bar(names, v, bottom=bottom, color=cols[cls], width=0.62, label=cls)
        for i, (b, h) in enumerate(zip(bottom, v)):
            if h >= 8:
                axes[0].text(i, b + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=7.4,
                             color="white", fontweight="bold")
        bottom += v
    axes[0].set_ylabel("Share of 151 potholes (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("(a) Predicted severity class", fontsize=9.5)
    axes[0].legend(fontsize=6.8, frameon=False, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, -0.2))
    axes[0].tick_params(axis="x", labelsize=7.4)
    lab = ["Cloud\nVLM", "Distilled\nensemble", "Distilled +\nscale aug."]
    x = np.arange(3)
    axes[1].bar(x - 0.19, kap, 0.36, color="#6b46c1", label=r"$\kappa$ vs measured depth")
    axes[1].bar(x + 0.19, rho, 0.36, color="#117a8b", label="Spearman $\\rho$")
    for xi, k, r in zip(x, kap, rho):
        axes[1].text(xi - 0.19, k + (0.012 if k >= 0 else -0.012), f"{k:+.2f}", ha="center",
                     va="bottom" if k >= 0 else "top", fontsize=6.8)
        axes[1].text(xi + 0.19, r + (0.012 if r >= 0 else -0.012), f"{r:+.2f}", ha="center",
                     va="bottom" if r >= 0 else "top", fontsize=6.8)
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(lab, fontsize=7.4)
    axes[1].set_ylim(-0.3, 0.34)
    axes[1].set_title("(b) Agreement with depth", fontsize=9.5)
    axes[1].legend(fontsize=6.6, frameon=False, loc="upper center", ncol=1,
                   bbox_to_anchor=(0.5, -0.2))
    axes[1].grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig_domainshift.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_domainshift.png", dpi=300, bbox_inches="tight")
    print("wrote fig_domainshift", [round(v, 3) for v in kap], [round(v, 3) for v in rho])

def safety_round2():
    """Round-2 version of fig_safety: panel (b) now compares the judge with humans on
    the system the paper reports (constrained local generator). Humans: 50 scenes,
    3 raters (human_eval_round2.json). Judge: 150 random scenes, both orders
    (llm_judge_warnings_constrained.json). The two samples overlap in 5 scenes, so
    this is a comparison of means, not an item-by-item one."""
    u = json.loads((RES / "advisory_safety.json").read_text())
    h = json.loads((RES / "human_eval_round2.json").read_text())["descriptive"]
    jm = json.loads((RES / "llm_judge_warnings_constrained.json").read_text())["mean_ratings"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    labels = ["Cloud\n(prior work)", "Local\nunconstrained", "Local\nconstrained"]
    vals = [u["lane_departure_pct"]["cloud"], u["lane_departure_pct"]["local_unconstrained"],
            u["lane_departure_pct"]["local_constrained"]]
    axes[0].bar(labels, vals, color=["#c0392b", "#e67e22", "#27ae60"], width=0.6)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("Scenes advising a lane departure (%)")
    axes[0].set_ylim(0, max(vals) * 1.25 + 4)
    axes[0].set_title("(a) Unsupportable advice", fontsize=9.5)
    axes[0].grid(axis="y", alpha=0.25, ls=":")
    x, w = np.arange(2), 0.2
    jc = [jm["cloud_clarity"], jm["cloud_safety"]]
    jl = [jm["local_clarity"], jm["local_safety"]]
    hc = [h["cloud"]["clarity"]["mean"], h["cloud"]["safety"]["mean"]]
    hl = [h["local"]["clarity"]["mean"], h["local"]["safety"]["mean"]]
    axes[1].bar(x - 1.5 * w, jc, w, label="Cloud, judge", color="#c0392b", alpha=0.45)
    axes[1].bar(x - 0.5 * w, hc, w, label="Cloud, humans", color="#c0392b")
    axes[1].bar(x + 0.5 * w, jl, w, label="Local, judge", color="#27ae60", alpha=0.45)
    axes[1].bar(x + 1.5 * w, hl, w, label="Local, humans", color="#27ae60")
    for xi, vals_ in ((x - 1.5 * w, jc), (x - 0.5 * w, hc), (x + 0.5 * w, jl), (x + 1.5 * w, hl)):
        for a, b in zip(xi, vals_):
            axes[1].text(a, b + 0.06, f"{b:.2f}", ha="center", fontsize=6.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Clarity", "Safety"])
    axes[1].set_ylabel("Mean rating (1-5)")
    axes[1].set_ylim(0, 6.0)
    axes[1].legend(fontsize=6.4, frameon=False, loc="upper center", ncol=2)
    axes[1].set_title("(b) Automated judge vs. human raters", fontsize=9.5)
    axes[1].grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig_safety_r2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_safety_r2.png", dpi=300, bbox_inches="tight")
    print("wrote fig_safety_r2")


def human_round2():
    """Per-rater view of the round-2 human study: (a) local-minus-cloud difference
    with 95% bootstrap intervals over scenes, for each rater and pooled; (b) the
    preference split for each rater."""
    import sys
    sys.path.insert(0, str(RES.parent / "benchmarks"))
    from analyze_human_round2 import load, RATERS
    d = load()
    rng = np.random.default_rng(7)
    names = {"R1": "Rater 1\n(researcher)", "R2": "Rater 2\n(researcher)",
             "R3": "Rater 3\n(driver)", "all": "All raters"}
    order = ["R1", "R2", "R3", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    for k, (metric, col, off) in enumerate((("clarity", "#1d6fb8", -0.13), ("safety", "#27ae60", 0.13))):
        for i, r in enumerate(order):
            g = d if r == "all" else d[d.rater == r]
            s = g.pivot_table(index="scene", columns="system", values=metric, aggfunc="mean")
            diff = (s["local"] - s["cloud"]).to_numpy()
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            axes[0].errorbar(i + off, diff.mean(), yerr=[[diff.mean() - lo], [hi - diff.mean()]],
                             fmt="o" if r != "all" else "D", color=col, ms=5 if r != "all" else 6.5,
                             capsize=3, lw=1.3, label=metric.capitalize() if i == 0 else None)
    axes[0].axhline(0, color="#555", lw=0.8)
    axes[0].axvline(2.5, color="#bbb", lw=0.6, ls=":")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([names[r] for r in order], fontsize=7.6)
    axes[0].set_ylabel("Local minus cloud (rating points)")
    axes[0].set_title("(a) Rating difference, 95% CI over scenes", fontsize=9.5)
    axes[0].legend(fontsize=7.2, frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=0.25, ls=":")
    p = d[d.system == "local"]
    rows = []
    for r in order:
        g = p if r == "all" else p[p.rater == r]
        n = len(g)
        rows.append([100 * (g.pref == c).sum() / n for c in ("local", "tie", "cloud")])
    rows = np.array(rows)
    left = np.zeros(len(order))
    for j, (lab, col) in enumerate((("Local preferred", "#27ae60"), ("Tie", "#c9ccd1"),
                                    ("Cloud preferred", "#c0392b"))):
        axes[1].barh(range(len(order)), rows[:, j], left=left, color=col, label=lab, height=0.62)
        for i, v in enumerate(rows[:, j]):
            if v >= 9:
                axes[1].text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                             color="white" if j != 1 else "#333")
        left += rows[:, j]
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([names[r].replace("\n", " ") for r in order], fontsize=7.6)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share of scenes (%)")
    axes[1].set_title("(b) Which warning would you rather get?", fontsize=9.5)
    axes[1].legend(fontsize=6.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    axes[1].spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_human_r2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_human_r2.png", dpi=300, bbox_inches="tight")
    print("wrote fig_human_r2")


ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README

if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:] == ["round2"]:      # MDPI version: only the round-2 figures
        safety_round2()
        human_round2()
        raise SystemExit
    segmentation()
    domain_shift()
    frontier()
    if (RES / "advisory_safety.json").exists():
        safety()
    else:
        print("advisory_safety.json not present yet - run measure_advisory_safety.py first")
