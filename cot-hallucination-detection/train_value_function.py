"""
Train value functions for the CoT-as-MRP dataset using TD(0), TD(lambda), or Monte Carlo.
States come from the cumulative reasoning steps extracted by data/process_qwen_responses.py.

Assumptions:
- The processed CSV lives in data/qwen-3b-instruct-gsm8k-responses-processed.csv (or override via --train-path/--test-path).
- Terminal reward = 1 if the final answer is incorrect (label==1), else 0.
- Intermediate rewards are 0.
- A policy is fixed (the original Qwen model that generated the trajectories); we only evaluate its value function.

This script initializes the value function from a Qwen model (encoder) with a small value head on top.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None


# -----------------------------
# Data utils
# -----------------------------


def load_episodes(path: str, use_parsed_only: bool = True) -> List[dict]:
    """Load episodes from CSV into list of dicts with states and terminal reward."""
    import pandas as pd

    df = pd.read_csv(path)
    if use_parsed_only and "steps_parsed" in df.columns:
        df = df[df["steps_parsed"]]
    episodes = []
    for _, row in df.iterrows():
        states = json.loads(row["reasoning_steps_json"])
        reward = float(row.get("label", 0))  # 1 if incorrect, else 0
        episodes.append(
            {
                "problem_id": row.get("problem_id", None),
                "states": states,
                "reward": reward,
            }
        )
    return episodes


def build_td0_transitions(episodes: Sequence[dict], gamma: float):
    """Build one-step transitions with terminal reward on last step."""
    transitions = []
    for ep in episodes:
        states = ep["states"]
        R = ep["reward"]
        for i, state in enumerate(states):
            next_state = states[i + 1] if i + 1 < len(states) else None
            r = 0.0 if next_state is not None else R
            transitions.append((state, next_state, r))
    return transitions


def build_mc_targets(episodes: Sequence[dict], gamma: float):
    """Build Monte Carlo targets (only terminal reward discounted back)."""
    pairs = []
    for ep in episodes:
        states = ep["states"]
        R = ep["reward"]
        T = len(states)
        for t, state in enumerate(states):
            # only terminal reward => discounted terminal
            target = (gamma ** (T - t - 1)) * R
            pairs.append((state, target))
    return pairs


# -----------------------------
# Model
# -----------------------------


class ValueModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        freeze_base: bool = True,
        head_hidden: int = 512,
        head_dropout: float = 0.1,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base = AutoModel.from_pretrained(base_model_name)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        hidden = self.base.config.hidden_size
        # Learnable query to attend over token embeddings.
        self.query = nn.Parameter(torch.randn(1, 1, hidden))
        self.attn = nn.MultiheadAttention(hidden, attn_heads, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text and predict scalar value for each sequence."""
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)
        # Attend over tokens with a learnable query; mask pads.
        batch_size = hidden.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, H)
        key_padding_mask = attention_mask == 0  # True at pads
        attn_out, _ = self.attn(query, hidden, hidden, key_padding_mask=key_padding_mask)
        pooled = attn_out.squeeze(1)  # (B, H)
        logits = self.value_head(pooled).squeeze(-1)  # (B,)
        return logits

    def encode(self, texts: List[str], device: torch.device):
        """Tokenize a batch of strings to tensors on the target device."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


# -----------------------------
# Datasets
# -----------------------------


@dataclass
class Transition:
    state: str
    next_state: str
    reward: float


class TransitionDataset(Dataset):
    def __init__(self, transitions: List[tuple]):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]


def collate_transitions(batch):
    """Keep None for terminal next_state; return tuple of lists."""
    states, next_states, rewards = zip(*batch)
    return list(states), list(next_states), list(rewards)


class MCDataset(Dataset):
    def __init__(self, pairs: List[tuple]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# -----------------------------
# Training loops
# -----------------------------


def train_td0(model: ValueModel, transitions: List[tuple], args, device):
    """One-step TD policy evaluation."""
    ds = TransitionDataset(transitions)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_transitions)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            states, next_states, rewards = batch
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            input_ids, attn = model.encode(list(states), device)
            logits = model(input_ids, attn)

            next_probs = torch.zeros_like(logits)
            mask = torch.zeros_like(logits)
            if any(next_states):
                valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
                if valid_indices:
                    ns_texts = [next_states[i] for i in valid_indices]
                    ns_ids, ns_attn = model.encode(ns_texts, device)
                    ns_logits = model(ns_ids, ns_attn).detach()
                    ns_probs = torch.sigmoid(ns_logits)
                    for idx, v in zip(valid_indices, ns_probs):
                        next_probs[idx] = v
                        mask[idx] = 1.0

            targets = rewards + args.gamma * next_probs * mask
            loss = bce(logits, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if args.wandb_project and wandb is not None:
                wandb.log({"loss_td0": loss.item(), "epoch": epoch, "step": step})
            step += 1
        print(f"[TD0] epoch {epoch} loss {loss.item():.4f}")


def eval_mc(model: ValueModel, pairs: List[tuple], args, device, split_name: str):
    """Evaluate MC loss on a split without updating weights."""
    if not pairs:
        return None
    ds = MCDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            states, targets = batch
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            input_ids, attn = model.encode(list(states), device)
            logits = model(input_ids, attn)
            loss = bce(logits, targets)
            total_loss += loss.item() * len(states)
            total_count += len(states)
    avg_loss = total_loss / max(total_count, 1)
    if args.wandb_project and wandb is not None:
        wandb.log({f"loss_mc_{split_name}": avg_loss})
    print(f"[MC eval:{split_name}] loss {avg_loss:.4f}")
    return avg_loss


def train_mc(model: ValueModel, pairs: List[tuple], args, device):
    """Monte Carlo policy evaluation."""
    ds = MCDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            states, targets = batch
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            input_ids, attn = model.encode(list(states), device)
            logits = model(input_ids, attn)
            loss = bce(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.wandb_project and wandb is not None:
                wandb.log({"loss_mc": loss.item(), "epoch": epoch, "step": step})
            step += 1
        print(f"[MC] epoch {epoch} loss {loss.item():.4f}")


def train_td_lambda(model: ValueModel, episodes: Sequence[dict], args, device):
    """Forward-view TD(lambda) with bootstrap on next-state value."""
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    step = 0
    for epoch in range(args.epochs):
        for ep in episodes:
            states = ep["states"]
            R = ep["reward"]
            rewards = [0.0] * (len(states) - 1) + [R]

            # Predict current values
            ids, attn = model.encode(states, device)
            logits = model(ids, attn)
            probs_detached = torch.sigmoid(logits.detach())

            # Compute lambda-returns (forward view with bootstrap on next value)
            T = len(states)
            lambda_returns = [0.0] * T
            next_return = 0.0
            for t in reversed(range(T)):
                v_next = probs_detached[t + 1] if t + 1 < T else torch.tensor(0.0, device=device)
                next_return = rewards[t] + args.gamma * ((1 - args.lmbda) * v_next + args.lmbda * next_return)
                lambda_returns[t] = next_return
            targets = torch.stack(lambda_returns)

            loss = bce(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if args.wandb_project and wandb is not None:
                wandb.log({"loss_td_lambda": loss.item(), "epoch": epoch, "step": step})
            step += 1
        print(f"[TD-lambda] epoch {epoch} loss {loss.item():.4f}")


# -----------------------------
# CLI
# -----------------------------


def load_config_file(path: str) -> Dict[str, Any]:
    """Load JSON or YAML config dict."""
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = cfg_path.read_text()
    # Try YAML first if available, else JSON
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None and cfg_path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def parse_args():
    # First parse config path
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        help="Path to JSON/YAML config; values set defaults; CLI flags override.",
    )
    config_args, remaining = config_parser.parse_known_args()
    config_defaults = load_config_file(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        parents=[config_parser], description="Policy evaluation for CoT MRP."
    )
    parser.add_argument(
        "--train-path",
        default=os.path.join("data", "qwen-train.csv"),
        help="Path to training CSV (processed).",
    )
    parser.add_argument(
        "--test-path",
        default=os.path.join("data", "qwen-test.csv"),
        help="Path to test CSV (processed).",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model name for initializing the value encoder.",
    )
    parser.add_argument("--algo", choices=["td0", "tdlambda", "mc", "all"], default="all")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze transformer weights.")
    parser.add_argument("--head-hidden", type=int, default=512, help="Hidden width of value head MLP.")
    parser.add_argument("--head-dropout", type=float, default=0.1, help="Dropout rate in value head MLP.")
    parser.add_argument("--attn-heads", type=int, default=4, help="Attention heads for token pooling.")
    parser.add_argument("--wandb-project", default=None, help="If set, log metrics to this W&B project.")

    # Apply config defaults (CLI overrides)
    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args(remaining)


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.wandb_project and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    train_episodes = load_episodes(args.train_path, use_parsed_only=True)
    test_episodes: List[dict] = []
    if os.path.exists(args.test_path):
        test_episodes = load_episodes(args.test_path, use_parsed_only=True)

    model = ValueModel(
        args.model_name,
        freeze_base=args.freeze_base,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        attn_heads=args.attn_heads,
    ).to(device)

    if args.algo in ("td0", "all"):
        transitions = build_td0_transitions(train_episodes, args.gamma)
        train_td0(model, transitions, args, device)
        if test_episodes:
            mc_pairs_test = build_mc_targets(test_episodes, args.gamma)
            eval_mc(model, mc_pairs_test, args, device, split_name="test")

    if args.algo in ("mc", "all"):
        mc_pairs = build_mc_targets(train_episodes, args.gamma)
        train_mc(model, mc_pairs, args, device)
        if test_episodes:
            mc_pairs_test = build_mc_targets(test_episodes, args.gamma)
            eval_mc(model, mc_pairs_test, args, device, split_name="test")

    if args.algo in ("tdlambda", "all"):
        train_td_lambda(model, train_episodes, args, device)
        if test_episodes:
            mc_pairs_test = build_mc_targets(test_episodes, args.gamma)
            eval_mc(model, mc_pairs_test, args, device, split_name="test")

    if args.wandb_project and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
