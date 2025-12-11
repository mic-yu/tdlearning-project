"""
Evaluate a value function on the CoT-as-MRP test set with state-level and trajectory metrics,
and save visualizations.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from train_value_function import ValueModel, load_config_file


def load_episodes(path: str, use_parsed_only: bool = True) -> List[dict]:
    df = pd.read_csv(path)
    if use_parsed_only and "steps_parsed" in df.columns:
        df = df[df["steps_parsed"]]
    episodes = []
    for _, row in df.iterrows():
        states = json.loads(row["reasoning_steps_json"])
        reward = float(row.get("label", 0))
        episodes.append(
            {
                "problem_id": row.get("problem_id", None),
                "states": states,
                "label": reward,
            }
        )
    return episodes


def expand_states(episodes: Sequence[dict]) -> pd.DataFrame:
    records = []
    for ep_idx, ep in enumerate(episodes):
        for step_idx, state in enumerate(ep["states"]):
            records.append(
                {
                    "episode_idx": ep_idx,
                    "problem_id": ep.get("problem_id"),
                    "step_idx": step_idx + 1,
                    "state": state,
                    "label": ep["label"],
                }
            )
    return pd.DataFrame(records)


def batch_predict(model: ValueModel, texts: Sequence[str], device: torch.device, batch_size: int = 2):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            ids, attn = model.encode(chunk, device)
            logits = model(ids, attn)
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().tolist())
    return preds


def state_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df["label"].astype(int).values
    p = np.clip(df["pred"].values, 1e-6, 1 - 1e-6)
    metrics = {
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else math.nan,
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
    }
    # per-depth bins
    bins = {
        "step1": df[df.step_idx == 1],
        "step2": df[df.step_idx == 2],
        "step3_5": df[(df.step_idx >= 3) & (df.step_idx <= 5)],
        "step6p": df[df.step_idx >= 6],
    }
    for name, sub in bins.items():
        if len(sub) == 0 or len(sub.label.unique()) < 2:
            continue
        yb = sub.label.values
        pb = np.clip(sub.pred.values, 1e-6, 1 - 1e-6)
        metrics[f"auroc_{name}"] = roc_auc_score(yb, pb)
        metrics[f"auprc_{name}"] = average_precision_score(yb, pb)
        metrics[f"brier_{name}"] = brier_score_loss(yb, pb)
        metrics[f"logloss_{name}"] = log_loss(yb, pb)
    return metrics


def episode_metrics(df: pd.DataFrame, tau_high: float, tau_low: float) -> Dict[str, float]:
    by_ep: Dict[int, pd.DataFrame] = dict(tuple(df.groupby("episode_idx")))
    hit_wrong = []
    false_alarm_correct = []
    time_to_safety_correct = []
    wrong_low = []
    max_scores = []
    mean_scores = []
    labels = []
    for ep_idx, ep_df in by_ep.items():
        label = ep_df["label"].iloc[0]
        preds = ep_df.sort_values("step_idx")["pred"].values
        labels.append(label)
        # hitting time to high risk
        high_indices = np.where(preds > tau_high)[0]
        hit = high_indices[0] + 1 if len(high_indices) else None
        # time to safety low
        low_indices = np.where(preds < tau_low)[0]
        low_hit = low_indices[0] + 1 if len(low_indices) else None
        if label == 1:
            if hit:
                hit_wrong.append(hit)
            wrong_low.append(low_hit is not None)
        else:
            if hit:
                false_alarm_correct.append(hit)
            if low_hit:
                time_to_safety_correct.append(low_hit)
        max_scores.append(preds.max())
        mean_scores.append(preds.mean())
    labels_np = np.array(labels)
    metrics = {}
    if hit_wrong:
        metrics["median_hit_step_incorrect"] = float(np.median(hit_wrong))
    metrics["false_alarm_rate"] = len(false_alarm_correct) / max(1, sum(labels_np == 0))
    if time_to_safety_correct:
        metrics["median_time_to_safety_correct"] = float(np.median(time_to_safety_correct))
    metrics["wrong_lowrisk_rate"] = sum(wrong_low) / max(1, sum(labels_np == 1))
    metrics["max_auroc"] = roc_auc_score(labels_np, max_scores) if len(np.unique(labels_np)) > 1 else math.nan
    metrics["max_auprc"] = average_precision_score(labels_np, max_scores)
    metrics["mean_auroc"] = roc_auc_score(labels_np, mean_scores) if len(np.unique(labels_np)) > 1 else math.nan
    metrics["mean_auprc"] = average_precision_score(labels_np, mean_scores)
    return metrics


def plot_reliability(df: pd.DataFrame, out_path: Path, bins: int = 10):
    y = df["label"].astype(int).values
    p = np.clip(df["pred"].values, 1e-6, 1 - 1e-6)
    bin_ids = np.clip((p * bins).astype(int), 0, bins - 1)
    acc = []
    conf = []
    for b in range(bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc.append(y[mask].mean())
        conf.append(p[mask].mean())
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    plt.plot(conf, acc, marker="o")
    plt.xlabel("Predicted error probability")
    plt.ylabel("Empirical error rate")
    plt.title("Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hit_hist(values: List[int], title: str, out_path: Path):
    if not values:
        return
    plt.figure()
    plt.hist(values, bins=range(1, max(values) + 2), align="left", rwidth=0.8)
    plt.xlabel("Step index")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, prefix: Path):
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC")
        plt.tight_layout()
        plt.savefig(prefix.with_suffix(".roc.png"), dpi=200)
        plt.close()
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR")
    plt.tight_layout()
    plt.savefig(prefix.with_suffix(".pr.png"), dpi=200)
    plt.close()


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", help="JSON/YAML config; CLI overrides.")
    config_args, remaining = base_parser.parse_known_args()
    cfg = load_config_file(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(parents=[base_parser], description="Evaluate value function.")
    parser.add_argument("--test-path", default=os.path.join("data", "qwen-test.csv"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default=None, help="Path to model state_dict.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--tau-high", type=float, default=0.5)
    parser.add_argument("--tau-low", type=float, default=0.2)
    parser.add_argument("--out-dir", default="eval_outputs")
    if cfg:
        parser.set_defaults(**cfg)
    return parser.parse_args(remaining)


def main():
    args = parse_args()
    device = torch.device(args.device)

    episodes = load_episodes(args.test_path, use_parsed_only=True)
    df_states = expand_states(episodes)

    model = ValueModel(
        args.model_name,
        freeze_base=args.freeze_base,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        attn_heads=args.attn_heads,
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()

    df_states["pred"] = batch_predict(
        model, df_states["state"].tolist(), device=device, batch_size=args.batch_size
    )

    metrics_state = state_metrics(df_states)
    metrics_ep = episode_metrics(df_states, tau_high=args.tau_high, tau_low=args.tau_low)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "state_metrics.json").write_text(json.dumps(metrics_state, indent=2))
    (out_dir / "episode_metrics.json").write_text(json.dumps(metrics_ep, indent=2))

    plot_reliability(df_states, out_dir / "reliability.png")
    hit_wrong = []
    false_alarm = []
    by_ep = dict(tuple(df_states.groupby("episode_idx")))
    for ep_idx, ep_df in by_ep.items():
        label = ep_df.label.iloc[0]
        preds = ep_df.sort_values("step_idx").pred.values
        high_indices = np.where(preds > args.tau_high)[0]
        hit = high_indices[0] + 1 if len(high_indices) else None
        if label == 1 and hit:
            hit_wrong.append(hit)
        if label == 0 and hit:
            false_alarm.append(hit)
    plot_hit_hist(hit_wrong, "First high-risk step (incorrect episodes)", out_dir / "hit_incorrect.png")
    plot_hit_hist(false_alarm, "False alarm steps (correct episodes)", out_dir / "false_alarm.png")

    y_true = df_states["label"].astype(int).values
    y_score = np.clip(df_states["pred"].values, 1e-6, 1 - 1e-6)
    plot_roc_pr(y_true, y_score, out_dir / "state")

    print("State metrics:", json.dumps(metrics_state, indent=2))
    print("Episode metrics:", json.dumps(metrics_ep, indent=2))
    print(f"Plots and metrics saved to {out_dir}")


if __name__ == "__main__":
    main()
