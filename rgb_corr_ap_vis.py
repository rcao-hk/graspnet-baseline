import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = [
    ("mmgnet_scene_none", "point-only"),
    ("mmgnet_scene", "early"),
    ("mmgnet_scene_concat", "concat"),
    ("mmgnet_scene_add", "add"),
    ("mmgnet_scene_gate", "gate"),
    ("mmgnet_scene_intermediate", "intermediate"),
]

CORRUPTIONS = ["cutout", "blur", "brightness", "contrast"]
SPLITS = ["seen", "similar", "novel"]


def split_tag(split: str) -> str:
    return f"test_{split}"


def find_dump_dir(experiment_root: str, method: str, split: str, noise: str, severity: int) -> str | None:
    t = split_tag(split)
    cands = [
        os.path.join(experiment_root, f"{method}.{noise}.s{severity}"),
        os.path.join(experiment_root, f"{method}.{t}.{noise}.s{severity}"),
        os.path.join(experiment_root, f"{method}.{split}.{noise}.s{severity}"),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return None


def load_ap_topk(dump_dir: str, split: str, camera: str, topk: int = 50) -> float:
    ap_path = os.path.join(dump_dir, f"ap_test_{split}_{camera}.npy")
    if not os.path.isfile(ap_path):
        return np.nan
    res = np.load(ap_path)
    return float(np.mean(res[:, :, :topk, :]))


def get_ap(experiment_root: str, method: str, split: str, noise: str, severity: int,
           camera: str, topk: int) -> float:
    if severity == 0:
        noise_load, sev_load = "none", 0
    else:
        noise_load, sev_load = noise, severity

    dump_dir = find_dump_dir(experiment_root, method, split, noise_load, sev_load)
    if dump_dir is None:
        return np.nan
    return load_ap_topk(dump_dir, split, camera, topk=topk)


def build_table(experiment_root: str, camera: str, topk: int = 50) -> pd.DataFrame:
    rows = []
    for method, method_alias in METHODS:
        for corr in CORRUPTIONS:
            for sev in [0, 1, 2, 3, 4, 5]:
                ap_splits = []
                for sp in SPLITS:
                    ap = get_ap(experiment_root, method, sp, corr, sev, camera, topk)
                    ap_splits.append(ap)
                    rows.append({
                        "method": method,
                        "method_alias": method_alias,
                        "corruption": corr,
                        "severity": sev,
                        "split": sp,
                        "ap": ap,
                    })

                ap_mean = float(np.nanmean(ap_splits)) if np.any(np.isfinite(ap_splits)) else np.nan
                rows.append({
                    "method": method,
                    "method_alias": method_alias,
                    "corruption": corr,
                    "severity": sev,
                    "split": "mean",
                    "ap": ap_mean,
                })

    df = pd.DataFrame(rows)

    # baseline = severity 0 (none.s0)
    baseline_df = df[df["severity"] == 0][["method", "split", "ap"]].drop_duplicates()
    baseline_df = baseline_df.rename(columns={"ap": "ap_base"})
    df = df.merge(baseline_df, on=["method", "split"], how="left")

    df["drop_pp"] = (df["ap_base"] - df["ap"]) * 100.0
    df["drop_pct"] = np.where(
        np.isfinite(df["ap_base"]) & (df["ap_base"] > 1e-12) & np.isfinite(df["ap"]),
        (df["ap_base"] - df["ap"]) / df["ap_base"] * 100.0,
        np.nan
    )
    return df



def _resolve_method_name(method_or_alias: str | None) -> str | None:
    """Resolve a method name or its display alias to the canonical method key."""
    if method_or_alias is None or str(method_or_alias).strip() == "":
        return None
    q = str(method_or_alias).strip()
    for method, alias in METHODS:
        if q == method or q == alias:
            return method
    return q


def _nanmean_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def build_utility_sensitivity_summary(
    df: pd.DataFrame,
    split: str = "mean",
    reference_method: str | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a compact utility-sensitivity summary for RGB corruption experiments.

    Definitions, using AP in [0, 1]:
      - clean_ap_pct: AP under clean RGB, reported as percentage points.
      - corrupted_ap_pct: mean AP over selected corruptions and severities, reported as percentage points.
      - sensitivity_pp: mean AP drop from clean AP under corruptions, in percentage points.
      - sensitivity_pct: relative AP drop, normalized by clean AP.
      - retention_pct: corrupted_ap / clean_ap.
      - worst_drop_pp: largest AP drop among selected corruption/severity cases.
      - utility_vs_ref_pp: optional clean-AP gain over a reference method, in percentage points.

    If no reference_method is supplied, clean_ap_pct itself should be interpreted as
    the clean-setting utility of each fusion strategy.
    """
    if corruptions is None:
        corruptions = CORRUPTIONS
    if severities is None:
        severities = [1, 2, 3, 4, 5]

    ref_method = _resolve_method_name(reference_method)
    ref_clean = np.nan
    if ref_method is not None:
        ref_clean_vals = df[
            (df["method"] == ref_method) &
            (df["split"] == split) &
            (df["severity"] == 0)
        ]["ap"].values
        ref_clean = _nanmean_or_nan(ref_clean_vals)
        if not np.isfinite(ref_clean):
            print(f"[WARN] reference_method={reference_method!r} resolved to {ref_method!r}, but no clean AP was found for split={split!r}.")

    rows = []
    for method, alias in METHODS:
        clean_vals = df[
            (df["method"] == method) &
            (df["split"] == split) &
            (df["severity"] == 0)
        ]["ap"].values
        clean_ap = _nanmean_or_nan(clean_vals)

        corr_rows = df[
            (df["method"] == method) &
            (df["split"] == split) &
            (df["corruption"].isin(corruptions)) &
            (df["severity"].isin(severities))
        ].copy()

        corrupted_ap = _nanmean_or_nan(corr_rows["ap"].values)
        sensitivity_pp = (clean_ap - corrupted_ap) * 100.0 if np.isfinite(clean_ap) and np.isfinite(corrupted_ap) else np.nan
        sensitivity_pct = (clean_ap - corrupted_ap) / clean_ap * 100.0 if np.isfinite(clean_ap) and clean_ap > 1e-12 and np.isfinite(corrupted_ap) else np.nan
        retention_pct = corrupted_ap / clean_ap * 100.0 if np.isfinite(clean_ap) and clean_ap > 1e-12 and np.isfinite(corrupted_ap) else np.nan

        clean_utility = (
            clean_ap - ref_clean
            if np.isfinite(clean_ap) and np.isfinite(ref_clean)
            else np.nan
        )

        corrupted_utility = (
            corrupted_ap - ref_clean
            if np.isfinite(corrupted_ap) and np.isfinite(ref_clean)
            else np.nan
        )

        retained_utility_pct = (
            corrupted_utility / clean_utility * 100.0
            if (
                np.isfinite(clean_utility)
                and abs(clean_utility) > 1e-12
                and np.isfinite(corrupted_utility)
            )
            else np.nan
        )

        # Worst case among selected corruption/severity settings.
        if corr_rows["drop_pp"].notna().any():
            worst_idx = corr_rows["drop_pp"].idxmax()
            worst = corr_rows.loc[worst_idx]
            worst_drop_pp = float(worst["drop_pp"])
            worst_ap_pct = float(worst["ap"] * 100.0) if np.isfinite(worst["ap"]) else np.nan
            worst_case = f"{worst['corruption']}.s{int(worst['severity'])}"
        else:
            worst_drop_pp = np.nan
            worst_ap_pct = np.nan
            worst_case = "NA"

        row = {
            "method": method,
            "alias": alias,
            "clean_ap_pct": clean_ap * 100.0 if np.isfinite(clean_ap) else np.nan,
            "utility_vs_ref_pp": (
                clean_utility * 100.0
                if np.isfinite(clean_utility)
                else np.nan
            ),
            "corrupted_ap_pct": corrupted_ap * 100.0 if np.isfinite(corrupted_ap) else np.nan,
            "sensitivity_pp": sensitivity_pp,
            "sensitivity_pct": sensitivity_pct,
            "retention_pct": retention_pct,
            "worst_drop_pp": worst_drop_pp,
            "worst_ap_pct": worst_ap_pct,
            "worst_case": worst_case,
            "corrupted_gain_vs_ref_pp": (
                corrupted_utility * 100.0
                if np.isfinite(corrupted_utility)
                else np.nan
            ),
            "retained_utility_pct": retained_utility_pct,

        }

        # Per-corruption mean AP drop, useful for identifying which corruption drives sensitivity.
        for corr in corruptions:
            csub = corr_rows[corr_rows["corruption"] == corr]
            row[f"{corr}_drop_pp"] = _nanmean_or_nan(csub["drop_pp"].values)
            row[f"{corr}_ap_pct"] = _nanmean_or_nan(csub["ap"].values) * 100.0 if csub["ap"].notna().any() else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary


def print_utility_sensitivity_summary(summary: pd.DataFrame, sort_by: str = "corrupted_ap_pct"):
    """Pretty-print the utility-sensitivity summary table."""
    if summary.empty:
        print("[SUMMARY] empty utility-sensitivity summary.")
        return

    if sort_by in summary.columns:
        ascending = sort_by in {"sensitivity_pp", "sensitivity_pct", "worst_drop_pp"}
        summary = summary.sort_values(sort_by, ascending=ascending)

    display_cols = [
        "alias",
        "clean_ap_pct",
        "utility_vs_ref_pp",
        "corrupted_ap_pct",
        "corrupted_gain_vs_ref_pp",
        "sensitivity_pp",
        "sensitivity_pct",
        "retention_pct",
        "retained_utility_pct",
        "worst_drop_pp",
        "worst_case",
    ]
    display_cols += [f"{c}_drop_pp" for c in CORRUPTIONS if f"{c}_drop_pp" in summary.columns]
    display_cols = [c for c in display_cols if c in summary.columns]

    printable = summary[display_cols].copy()
    if "utility_vs_ref_pp" in printable.columns and printable["utility_vs_ref_pp"].isna().all():
        printable = printable.drop(columns=["utility_vs_ref_pp"])

    print("\n========== Utility-Sensitivity Summary ==========")
    print("AP columns are reported in percentage points. Sensitivity is the mean AP drop over selected RGB corruptions/severities.")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(printable.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=================================================\n")

def plot_one(df: pd.DataFrame, corruption: str, split: str, out_path: str):
    sub = df[(df["corruption"] == corruption) & (df["split"] == split)].copy()
    if sub["ap"].notna().sum() == 0:
        print(f"[SKIP] no valid data for corruption={corruption}, split={split}")
        return

    plt.figure(figsize=(7.5, 5.5))
    for method, method_alias in METHODS:
        s = sub[sub["method"] == method].sort_values("severity")
        if s["ap"].notna().sum() == 0:
            continue
        plt.plot(s["severity"].values, s["ap"].values, marker="o", label=method_alias)

    plt.xlabel("RGB severity", fontsize=12, fontweight="bold")
    plt.ylabel("Grasp AP", fontsize=12, fontweight="bold")
    plt.title(f"{corruption} | {split}")
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_drop_bar(df: pd.DataFrame, corruption: str, split: str, out_path: str, mode: str = "pp"):
    """
    Grouped bar chart:
      x-axis: severity s1..s5
      bars: different methods
      y-axis: AP drop relative to baseline (s0)
    mode:
      - 'pp'  : (AP_base - AP) * 100
      - 'pct' : (AP_base - AP) / AP_base * 100
    """
    col = "drop_pp" if mode == "pp" else "drop_pct"
    sub = df[(df["corruption"] == corruption) & (df["split"] == split) & (df["severity"].isin([1,2,3,4,5]))].copy()
    if sub[col].notna().sum() == 0:
        print(f"[SKIP] no valid drop data for corruption={corruption}, split={split}, mode={mode}")
        return

    severities = [1, 2, 3, 4, 5]
    method_aliases = [alias for _, alias in METHODS]

    # Build matrix: shape (n_methods, n_severities)
    mat = np.full((len(METHODS), len(severities)), np.nan, dtype=np.float32)
    for mi, (method, alias) in enumerate(METHODS):
        for si, sev in enumerate(severities):
            v = sub[(sub["method"] == method) & (sub["severity"] == sev)][col].values
            if len(v) > 0:
                mat[mi, si] = v[0]

    x = np.arange(len(severities))
    n_methods = len(METHODS)
    width = 0.8 / n_methods

    plt.figure(figsize=(9.5, 5.5))
    for mi, alias in enumerate(method_aliases):
        plt.bar(x + (mi - (n_methods - 1) / 2) * width, mat[mi], width=width, label=alias)

    plt.xticks(x, [f"s{sv}" for sv in severities], fontsize=11, fontweight="bold")
    plt.xlabel("RGB severity", fontsize=12, fontweight="bold")

    if mode == "pp":
        plt.ylabel("AP drop (percentage points)", fontsize=12, fontweight="bold")
        title = f"{corruption} | {split} | AP drop (pp)"
    else:
        plt.ylabel("AP drop (%)", fontsize=12, fontweight="bold")
        title = f"{corruption} | {split} | AP drop (%)"

    plt.title(title)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_drop_bar_all_corruptions_mean(df: pd.DataFrame, out_path: str, mode: str = "pp"):
    """
    One figure for all 4 corruptions (cutout/blur/brightness/contrast) with split='mean'.
    2x2 subplots. Each subplot is grouped bars over severities s1..s5, colored by method.

    mode:
      - 'pp'  : (AP_base - AP) * 100
      - 'pct' : (AP_base - AP) / AP_base * 100
    """
    col = "drop_pp" if mode == "pp" else "drop_pct"
    severities = [1, 2, 3, 4, 5]
    n_methods = len(METHODS)
    method_aliases = [alias for _, alias in METHODS]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for ax, corr in zip(axes, CORRUPTIONS):
        sub = df[
            (df["corruption"] == corr) &
            (df["split"] == "mean") &
            (df["severity"].isin(severities))
        ].copy()

        # Build matrix: (n_methods, 5)
        mat = np.full((n_methods, len(severities)), np.nan, dtype=np.float32)
        for mi, (method, alias) in enumerate(METHODS):
            for si, sev in enumerate(severities):
                v = sub[(sub["method"] == method) & (sub["severity"] == sev)][col].values
                if len(v) > 0:
                    mat[mi, si] = v[0]

        x = np.arange(len(severities))
        width = 0.8 / n_methods

        for mi, alias in enumerate(method_aliases):
            ax.bar(x + (mi - (n_methods - 1) / 2) * width, mat[mi], width=width, label=alias)

        ax.set_title(f"{corr} (mean)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{sv}" for sv in severities], fontsize=12, fontweight="bold")
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", alpha=0.3)

    # Y label (shared)
    if mode == "pp":
        fig.supylabel("AP drop (percentage points)", fontsize=14, fontweight="bold")
        fig.suptitle("RGB Corruptions (mean) — AP drop (pp)", fontsize=16, fontweight="bold")
    else:
        fig.supylabel("AP drop (%)", fontsize=14, fontweight="bold")
        fig.suptitle("RGB Corruptions (mean) — AP drop (%)", fontsize=16, fontweight="bold")

    # One legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=False, fontsize=12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=str, default="/data2/robotarm/result/grasp/mmgnet/experiment")
    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="vis/plots_rgb")
    parser.add_argument("--only_mean", action="store_true", help="Only plot mean curve.")
    parser.add_argument("--export_csv", action="store_true", help="Export a csv with all parsed AP values.")

    parser.add_argument("--no_summary", action="store_true", help="Disable printing the utility-sensitivity summary table.")
    parser.add_argument("--summary_split", type=str, default="mean", choices=["seen", "similar", "novel", "mean"],
                        help="Split used for the utility-sensitivity summary table.")
    parser.add_argument("--summary_reference_method", type=str, default='point-only',
                        help="Optional reference method name or alias for utility_vs_ref_pp, e.g. a geometry-only baseline if it is included in METHODS.")
    parser.add_argument("--summary_sort_by", type=str, default="corrupted_ap_pct",
                        choices=["clean_ap_pct", "corrupted_ap_pct", "sensitivity_pp", "sensitivity_pct", "retention_pct", "worst_drop_pp"],
                        help="Column used to sort the printed summary.")
    parser.add_argument("--export_summary_csv", action="store_true", help="Export the utility-sensitivity summary csv.")

    # NEW
    parser.add_argument("--plot_drop_bar", action="store_true", help="Also plot AP drop as grouped bar charts.")
    parser.add_argument("--drop_mode", type=str, default="pp", choices=["pp", "pct"],
                        help="AP drop mode: pp=percentage points, pct=relative percent.")
    parser.add_argument("--plot_drop_bar_all_mean", action="store_true",
                        help="Plot one figure: 4 corruptions (mean split) AP drop grouped bars in 2x2 subplots.")

    args = parser.parse_args()

    df = build_table(args.experiment_root, args.camera, topk=args.topk)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.export_csv:
        csv_path = os.path.join(args.out_dir, "rgb_corruption_ap_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"[CSV] {csv_path}")


    summary_df = build_utility_sensitivity_summary(
        df,
        split=args.summary_split,
        reference_method=args.summary_reference_method,
    )
    if not args.no_summary:
        print_utility_sensitivity_summary(summary_df, sort_by=args.summary_sort_by)
    if args.export_summary_csv:
        summary_csv_path = os.path.join(args.out_dir, f"rgb_utility_sensitivity_summary.{args.summary_split}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[CSV] {summary_csv_path}")

    splits_to_plot = ["mean"] if args.only_mean else ["seen", "similar", "novel", "mean"]

    for corr in CORRUPTIONS:
        for sp in splits_to_plot:
            # AP curve
            out_path = os.path.join(args.out_dir, f"ap_curve.{corr}.{sp}.png")
            plot_one(df, corr, sp, out_path)

            # Drop bar
            if args.plot_drop_bar:
                out_drop = os.path.join(args.out_dir, f"ap_drop_bar.{corr}.{sp}.{args.drop_mode}.png")
                plot_drop_bar(df, corr, sp, out_drop, mode=args.drop_mode)

    if args.plot_drop_bar_all_mean:
        out_all = os.path.join(args.out_dir, f"ap_drop_bar.ALL4.mean.{args.drop_mode}.png")
        plot_drop_bar_all_corruptions_mean(df, out_all, mode=args.drop_mode)



if __name__ == "__main__":
    main()
