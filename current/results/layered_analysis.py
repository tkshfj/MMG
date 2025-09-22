#!/usr/bin/env python3
"""
layered_analysis.py — Analyze a sweep CSV layered by configurable groups, summarize metrics,
and produce plots. Matplotlib-only, seaborn-free, one chart per figure.
"""

from __future__ import annotations
import argparse
import ast
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Defaults --------
NAME_COL = "Name"
DEFAULT_GROUPS = ["architecture", "lr_scheduler", "calibration_method", "dropout_rate"]
DEFAULT_METRICS = [
    "val/acc", "val/auc", "val/cls_confmat_00", "val/cls_confmat_01",
    "val/cls_confmat_10", "val/cls_confmat_11", "val/dice", "val/iou", "val/multi", "val/loss",
]
DEFAULT_FACTORS = ["lr_scheduler", "lr", "calibration_method", "dropout_rate"]
PRIMARY_DEFAULT = "val/auc"


# -------- I/O --------
def _maybe_parse(x):
    if isinstance(x, str):
        s = x.strip()
        try:
            return ast.literal_eval(s)
        except Exception:
            return x
    return x


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize a couple common columns
    if "parameters.lr.max" in df.columns and "parameters.lr.min" in df.columns:
        df = df.rename(columns={"parameters.lr.max": "params/lr_max", "parameters.lr.min": "params/lr_min"})
    if "Created" in df.columns:
        df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
    for c in ["input_shape", "parameters.input_shape"]:
        if c in df.columns:
            df[c] = df[c].apply(_maybe_parse)

    # canonicalize lr_scheduler WITHOUT killing NaNs; keep 'lr' separate
    src = next((c for c in ["lr_scheduler", "lr_schedular", "parameters.lr_scheduler",
                            "parameters.lr_schedulers", "lr_strategy", "method"] if c in df.columns), None)
    if src is not None:
        s = df[src].apply(_maybe_parse)

        def to_scalar(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            if isinstance(v, (list, tuple)):
                for u in v:
                    if u is None or (isinstance(u, float) and np.isnan(u)):
                        continue
                    t = str(u).strip()
                    if t and t.lower() not in ("nan", "null", "none"):
                        return t
                return np.nan
            if isinstance(v, dict):
                for k in ("name", "scheduler", "type", "kind"):
                    if k in v and v[k]:
                        return str(v[k]).strip()
                for u in v.values():
                    if u:
                        return str(u).strip()
                return np.nan
            t = str(v).strip()
            return np.nan if t == "" else t

        df["lr_scheduler"] = s.map(to_scalar)

        def canon(t):
            if not isinstance(t, str):
                return t
            z = "".join(ch for ch in t.lower().strip() if ch.isalpha())
            if z in ("cosineannealingwarmrestarts",) or ("cosine" in z and "warm" in z):
                return "warmcos"
            if "cosine" in z:
                return "cosine"
            if "plateau" in z or "reducelronplateau" in z:
                return "plateau"
            if z in ("none", "fixed", "constant", "noscheduler"):
                return "none"
            return t
        df["lr_scheduler"] = df["lr_scheduler"].apply(canon)
    else:
        df["lr_scheduler"] = np.nan

    # cast common categoricals
    for c in ["architecture", "calibration_method", NAME_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -------- Aggregations --------
def layered_summary(df: pd.DataFrame, groups: List[str], metrics: List[str]) -> pd.DataFrame:
    groups = [c for c in groups if c in df.columns]
    metrics = [c for c in metrics if c in df.columns]
    if not groups:
        raise ValueError("No valid grouping columns present in the CSV.")
    if not metrics:
        raise ValueError("None of the requested metric columns are present in the CSV.")
    agg = df.groupby(groups, dropna=False)[metrics].agg(["count", "median", "mean", "std", "min", "max"])
    agg.columns = [f"{m}/{s}" for m, s in agg.columns]
    return agg.reset_index()


def pick_winners(df: pd.DataFrame, groups: List[str], score_col: str) -> pd.DataFrame:
    groups = [c for c in groups if c in df.columns]
    if not groups:
        raise ValueError("No valid grouping columns present for winners.")
    sort_cols = [c for c in [score_col, "val/acc", "val/dice", "val/iou", "val/multi", "val/loss"] if c in df.columns]
    if not sort_cols:
        raise ValueError("No scoring columns found to select winners.")
    ascending = [False if c != "val/loss" else True for c in sort_cols]
    return (df.sort_values(sort_cols, ascending=ascending)
              .groupby(groups, dropna=False, as_index=False)
              .head(1)
              .reset_index(drop=True))


def add_effective_lr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["params/lr_max", "lr", "opt/lr", "train/lr"]:
        if c in out.columns:
            v = pd.to_numeric(out[c], errors="coerce")
            if v.notna().any():
                out["effective_lr"] = v
                break
    if "effective_lr" not in out.columns:
        out["effective_lr"] = np.nan
    return out


# -------- Plots --------
def plot_heatmap(df: pd.DataFrame, score_col: str, outdir: Path):
    if score_col not in df.columns or "architecture" not in df.columns or "lr_scheduler" not in df.columns:
        return
    cal_methods = sorted(df["calibration_method"].dropna().astype(str).unique()) if "calibration_method" in df.columns else ["ALL"]
    for cm in cal_methods:
        dfx = df if cm == "ALL" else df[df["calibration_method"].astype(str) == cm]
        if dfx.empty:
            continue
        pivot = (dfx.groupby(["architecture", "lr_scheduler"])[score_col].median().unstack("lr_scheduler").sort_index())
        if pivot.empty:
            continue
        fig = plt.figure(figsize=(max(6, 1.2 * pivot.shape[1]), max(4, 0.8 * pivot.shape[0])))
        ax = plt.gca()
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Median {score_col} by Architecture x Scheduler" + ("" if cm == "ALL" else f"  (calibration_method={cm})"))
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(outdir / f"heatmap_{score_col.replace('/','_')}_arch_vs_sched{'' if cm=='ALL' else f'_cal_{cm}'}.png", dpi=160)
        plt.close(fig)


def plot_box_by_scheduler(df: pd.DataFrame, metric: str, outdir: Path):
    if metric not in df.columns or "lr_scheduler" not in df.columns or "architecture" not in df.columns:
        return
    for arch, dfa in df.groupby("architecture"):
        scheds = sorted(dfa["lr_scheduler"].fillna("NA").astype(str).unique(), key=lambda s: (s == "NA", s))
        data = [dfa.loc[dfa["lr_scheduler"].fillna("NA").astype(str) == s, metric].dropna().values for s in scheds]
        if not any(len(x) for x in data):
            continue
        fig = plt.figure(figsize=(max(6, 1.0 * len(scheds)), 4))
        ax = plt.gca()
        # version-safe labels
        try:
            import matplotlib as mpl
            major, minor, *_ = (int(x) for x in mpl.__version__.split("."))
            kw = {"tick_labels": scheds} if (major > 3 or (major == 3 and minor >= 9)) else {"labels": scheds}
        except Exception:
            kw = {"labels": scheds}
        ax.boxplot(data, showmeans=True, **kw)
        ax.set_title(f"{arch}: {metric} by lr_scheduler")
        ax.set_ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"box_{metric.replace('/','_')}_by_scheduler_{arch}.png", dpi=160)
        plt.close(fig)


def _plot_metric_vs_factor_by_arch(df: pd.DataFrame, metric: str, factor: str, outdir: Path):
    if metric not in df.columns or "architecture" not in df.columns or factor not in df.columns:
        return
    dfx = df.dropna(subset=[metric, factor, "architecture"])
    if dfx.empty:
        return
    fig = plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    if pd.api.types.is_numeric_dtype(dfx[factor]):
        for arch, dfa in dfx.groupby("architecture"):
            ax.scatter(dfa[factor].values, dfa[metric].values, alpha=0.8, label=str(arch))
        ax.set_xlabel(factor)
    else:
        cats = sorted(dfx[factor].astype(str).unique())
        code_map = {c: i for i, c in enumerate(cats)}
        rng = np.random.default_rng(42)
        for arch, dfa in dfx.groupby("architecture"):
            x = dfa[factor].astype(str).map(code_map).to_numpy(dtype=float)
            x += (rng.random(len(x)) - 0.5) * 0.3  # jitter width=0.15*2
            ax.scatter(x, dfa[metric].values, alpha=0.8, label=str(arch))
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_xlabel(factor + " (categorical)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {factor}")
    ax.legend(title="architecture", loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{metric.replace('/','_')}_vs_{factor}.png", dpi=160)
    plt.close(fig)


def plot_auc_acc_vs_factors_with_arch_colors(df: pd.DataFrame, factors: List[str], outdir: Path):
    metrics = [m for m in ["val/auc", "val/acc"] if m in df.columns]
    factors = [f for f in factors if f in df.columns]
    if not metrics or not factors or "architecture" not in df.columns:
        return
    for m in metrics:
        for f in factors:
            _plot_metric_vs_factor_by_arch(df, m, f, outdir)


def _plot_metric_vs_lr_colored_by_sched_for_arch(dfa: pd.DataFrame, arch: str, metric: str, lr_col: str, outdir: Path):
    dfx = dfa.dropna(subset=[metric, lr_col, "lr_scheduler"])
    if dfx.empty:
        return
    fig = plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    for sched, dfs in dfx.groupby("lr_scheduler"):
        ax.scatter(dfs[lr_col].values, dfs[metric].values, alpha=0.85, label=str(sched))
    if (dfx[lr_col] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel(lr_col)
    ax.set_ylabel(metric)
    ax.set_title(f"{arch}: {metric} vs {lr_col} (color = lr_scheduler)")
    ax.legend(title="lr_scheduler", loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{metric.replace('/', '_')}_vs_lr__arch={arch}.png", dpi=160)
    plt.close(fig)


def plot_auc_acc_vs_lr_colored_by_scheduler(df: pd.DataFrame, outdir: Path, lr_col: str):
    metrics = [m for m in ["val/auc", "val/acc"] if m in df.columns]
    if not metrics or "architecture" not in df.columns or "lr_scheduler" not in df.columns or lr_col not in df.columns:
        return
    for arch, dfa in df.groupby("architecture"):
        for m in metrics:
            _plot_metric_vs_lr_colored_by_sched_for_arch(dfa, arch, m, lr_col, outdir)


def plot_core_boxes_by_scheduler(df: pd.DataFrame, outdir: Path):
    metrics = [m for m in ["val/auc", "val/acc"] if m in df.columns]
    if not metrics or "architecture" not in df.columns or "lr_scheduler" not in df.columns:
        return
    sched_order = sorted(df["lr_scheduler"].fillna("NA").astype(str).unique(), key=lambda s: (s == "NA", s))
    for arch, dfa in df.groupby("architecture"):
        for metric in metrics:
            series_list = [dfa.loc[dfa["lr_scheduler"].fillna("NA").astype(str) == s, metric].dropna().values for s in sched_order]
            if not any(len(x) for x in series_list):
                continue
            fig = plt.figure(figsize=(max(6, 1.0 * len(sched_order)), 4))
            ax = plt.gca()
            try:
                import matplotlib as mpl
                major, minor, *_ = (int(x) for x in mpl.__version__.split("."))
                kw = {"tick_labels": sched_order} if (major > 3 or (major == 3 and minor >= 9)) else {"labels": sched_order}
            except Exception:
                kw = {"labels": sched_order}
            ax.boxplot(series_list, showmeans=True, **kw)
            ax.set_title(f"{arch}: {metric} by lr_scheduler")
            ax.set_ylabel(metric)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(outdir / f"box_{metric.replace('/', '_')}_by_scheduler_{arch}.png", dpi=160)
            plt.close(fig)


# -------- Orchestrator --------
def run(csv: str, outdir: str, score_col: str, groups: List[str], metrics: List[str], factors: List[str], lr_col_cli: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_csv(csv)
    if df.empty:
        raise SystemExit("CSV has no rows.")

    lr_candidates = ["params/lr_max", "lr", "opt/lr", "train/lr"]
    keep = list({NAME_COL, *groups, *metrics, *factors, score_col, *lr_candidates})
    keep = [c for c in keep if c in df.columns]
    work = df[keep].copy()

    # only coerce numeric for non-categoricals
    categorical = {"lr_scheduler", "calibration_method", "architecture", NAME_COL}
    to_numeric = [c for c in (metrics + factors + lr_candidates) if c in work.columns and c not in categorical]
    work = coerce_numeric(work, to_numeric)

    work = add_effective_lr(work)
    lr_col = lr_col_cli if lr_col_cli in work.columns else "effective_lr"

    work.to_csv(out / "subset_raw.csv", index=False)

    # sanity print for scheduler buckets
    if "lr_scheduler" in work.columns:
        print("lr_scheduler counts:")
        print(work["lr_scheduler"].fillna("NA").value_counts())

    layered_summary(work, groups, metrics).to_csv(out / "layered_summary.csv", index=False)
    winners = pick_winners(work, groups, score_col)
    ordered = [c for c in [NAME_COL, *groups, *metrics, score_col] if c in winners.columns]
    winners[ordered].to_csv(out / "layered.csv", index=False)

    # plots
    plot_heatmap(work, score_col, out)
    for met in ["val/dice", "val/iou", "val/multi"]:
        if met in work.columns:
            plot_box_by_scheduler(work, met, out)
    plot_auc_acc_vs_factors_with_arch_colors(work, factors, out)
    plot_auc_acc_vs_lr_colored_by_scheduler(work, out, lr_col)
    plot_core_boxes_by_scheduler(work, out)

    print("Saved layered analysis in:", out.resolve())
    print(f"LR column used for LR plots: {lr_col}")
    print("Key files: layered_summary.csv, layered.csv (w/ Name), subset_raw.csv, and plots.")


def main():
    p = argparse.ArgumentParser(description="Layered analysis for sweep CSV.")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="sweep_layered")
    p.add_argument("--score", default=PRIMARY_DEFAULT)
    p.add_argument("--groups", default=", ".join(DEFAULT_GROUPS))
    p.add_argument("--metrics", default=", ".join(DEFAULT_METRICS))
    p.add_argument("--factors", default=", ".join(DEFAULT_FACTORS))
    p.add_argument("--lr_col", default="lr", help="Column to use for LR in LR plots (default: lr).")
    a = p.parse_args()

    def parse(s):
        if not s:
            return []
        return [x.strip() for x in s.split(", ") if x.strip()]

    run(a.csv, a.out, a.score, parse(a.groups), parse(a.metrics), parse(a.factors), a.lr_col)


if __name__ == "__main__":
    main()
