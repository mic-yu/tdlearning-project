"""
Train value functions for the CoT-as-MRP dataset using TD(0), TD(lambda), or Monte Carlo.
States come from the cumulative reasoning steps extracted by data/process_qwen_responses.py.

Assumptions:
- The processed CSV lives in data/qwen-3b-instruct-gsm8k-responses-processed.csv (or override via --train-path/--test-path).
- Terminal reward = 1 if the final answer is incorrect (label==1), else 0.
- Intermediate rewards are 0.
- A policy is fixed (the original Qwen model that generated the trajectories); we only evaluate its value function.

The value function V(s) estimates P(incorrect | state s), i.e., the probability that the 
reasoning trajectory will lead to an incorrect final answer given the current state.

This script initializes the value function from a Qwen model with a linear value head on top.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    import wandb
except ImportError:
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


def build_td0_transitions(episodes: Sequence[dict]) -> List[tuple]:
    """
    Build one-step transitions (s, s', r) for TD(0).
    
    For each state in an episode:
    - next_state is the subsequent state (or None if terminal)
    - reward is 0 for all non-terminal transitions, R at the terminal state
    """
    transitions = []
    for ep in episodes:
        states = ep["states"]
        R = ep["reward"]
        T = len(states)
        for t, state in enumerate(states):
            is_terminal = (t == T - 1)
            next_state = None if is_terminal else states[t + 1]
            reward = R if is_terminal else 0.0
            transitions.append((state, next_state, reward))
    return transitions


def build_mc_targets(episodes: Sequence[dict], gamma: float) -> List[tuple]:
    """
    Build Monte Carlo targets (state, discounted_return) pairs.
    
    Since only terminal states have non-zero reward:
    G_t = gamma^(T-1-t) * R
    
    where T is the episode length and R is the terminal reward.
    """
    pairs = []
    for ep in episodes:
        states = ep["states"]
        R = ep["reward"]
        T = len(states)
        for t, state in enumerate(states):
            discounted_return = (gamma ** (T - 1 - t)) * R
            pairs.append((state, discounted_return))
    return pairs


# -----------------------------
# Model
# -----------------------------


class ValueModel(nn.Module):
    """
    Value function V(s) that estimates P(incorrect | state s).
    
    Uses a Qwen model as encoder with mean pooling over tokens,
    followed by a single linear layer to produce a scalar value.
    
    Output is a logit; apply sigmoid to get probability.
    """
    
    def __init__(self, base_model_name: str, freeze_base: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base = AutoModel.from_pretrained(base_model_name)
        
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        
        hidden_size = self.base.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text and predict scalar value (logit) for each sequence.
        
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask
            
        Returns:
            logits: (B,) scalar logits for each sequence
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)
        
        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_hidden = (hidden * mask_expanded).sum(dim=1)  # (B, H)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = sum_hidden / lengths  # (B, H)
        
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

    def predict_values(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Convenience method: encode texts and return value predictions (probabilities)."""
        input_ids, attention_mask = self.encode(texts, device)
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)


# -----------------------------
# Datasets
# -----------------------------


class TransitionDataset(Dataset):
    """Dataset of (state, next_state, reward) transitions for TD learning."""
    
    def __init__(self, transitions: List[tuple]):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]


def collate_transitions(batch):
    """Collate transitions, keeping None for terminal next_states."""
    states, next_states, rewards = zip(*batch)
    return list(states), list(next_states), list(rewards)


class MCDataset(Dataset):
    """Dataset of (state, target_return) pairs for Monte Carlo."""
    
    def __init__(self, pairs: List[tuple]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_mc(batch):
    """Collate MC pairs."""
    states, targets = zip(*batch)
    return list(states), list(targets)


# -----------------------------
# Training loops
# -----------------------------


def train_td0(model: ValueModel, transitions: List[tuple], args, device):
    """
    TD(0) policy evaluation.
    
    Update rule (in expectation):
        V(s) <- V(s) + α * [r + γ * V(s') - V(s)]
    
    For terminal states (s' = None), V(s') = 0.
    
    We use MSE loss: L = (V(s) - [r + γ * V(s')])^2
    where V(s) is the sigmoid of our logits (to stay in [0,1] for probabilities).
    """
    ds = TransitionDataset(transitions)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_transitions)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for states, next_states, rewards in loader:
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # Compute V(s) for current states
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits)  # V(s) in [0, 1]
            
            # Compute V(s') for next states (0 for terminal states)
            next_values = torch.zeros_like(values)
            non_terminal_indices = [i for i, ns in enumerate(next_states) if ns is not None]
            
            if non_terminal_indices:
                non_terminal_states = [next_states[i] for i in non_terminal_indices]
                ns_ids, ns_attn = model.encode(non_terminal_states, device)
                with torch.no_grad():  # Don't backprop through bootstrap target
                    ns_logits = model(ns_ids, ns_attn)
                    ns_values = torch.sigmoid(ns_logits)
                for idx, val in zip(non_terminal_indices, ns_values):
                    next_values[idx] = val
            
            # TD target: r + γ * V(s')
            targets = rewards + args.gamma * next_values
            
            # MSE loss between V(s) and TD target
            loss = nn.functional.mse_loss(values, targets.detach())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if args.wandb_project and wandb is not None:
                wandb.log({"td0/loss": loss.item(), "td0/step": global_step})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[TD0] Epoch {epoch}: avg_loss = {avg_loss:.4f}")


def train_mc(model: ValueModel, pairs: List[tuple], args, device):
    """
    Monte Carlo policy evaluation.
    
    For each (state, G) pair where G is the discounted return:
        V(s) <- V(s) + α * [G - V(s)]
    
    We use MSE loss: L = (V(s) - G)^2
    """
    ds = MCDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mc)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for states, targets in loader:
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits)  # V(s) in [0, 1]
            
            loss = nn.functional.mse_loss(values, targets)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if args.wandb_project and wandb is not None:
                wandb.log({"mc/loss": loss.item(), "mc/step": global_step})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[MC] Epoch {epoch}: avg_loss = {avg_loss:.4f}")


def train_td_lambda(model: ValueModel, episodes: Sequence[dict], args, device):
    """
    TD(λ) policy evaluation using forward view (λ-returns).
    
    The λ-return is defined as:
        G_t^λ = (1-λ) * Σ_{n=1}^{T-t-1} λ^{n-1} * G_t^{(n)} + λ^{T-t-1} * G_t
    
    where G_t^{(n)} is the n-step return and G_t is the full Monte Carlo return.
    
    For our MRP with only terminal rewards:
        G_t^{(n)} = γ^n * V(s_{t+n})  for n < T-t
        G_t = γ^{T-t-1} * R           (Monte Carlo return)
    
    Computed efficiently via backward recursion:
        G_t^λ = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]
    
    with G_T^λ = 0 (value after terminal state).
    """
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_episodes = 0
        
        for ep in episodes:
            states = ep["states"]
            R = ep["reward"]
            T = len(states)
            
            # rewards: 0 for t < T-1, R for t = T-1
            rewards = [0.0] * (T - 1) + [R]
            
            # Get current value estimates for all states
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits)  # (T,)
            
            # Compute λ-returns via backward recursion (detached values for bootstrap)
            values_detached = values.detach()
            lambda_returns = torch.zeros(T, device=device)
            
            # G_T^λ = 0 (after terminal state)
            next_lambda_return = 0.0
            
            for t in reversed(range(T)):
                r_t = rewards[t]
                if t == T - 1:
                    # Terminal state: G_{T-1}^λ = r_{T-1} = R
                    lambda_returns[t] = r_t
                else:
                    # G_t^λ = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]
                    # Since r_t = 0 for non-terminal:
                    v_next = values_detached[t + 1]
                    lambda_returns[t] = args.gamma * (
                        (1 - args.lmbda) * v_next + args.lmbda * next_lambda_return
                    )
                next_lambda_return = lambda_returns[t]
            
            # MSE loss between V(s) and λ-returns
            loss = nn.functional.mse_loss(values, lambda_returns)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_episodes += 1
            global_step += 1
            
            if args.wandb_project and wandb is not None:
                wandb.log({"td_lambda/loss": loss.item(), "td_lambda/step": global_step})
        
        avg_loss = epoch_loss / max(num_episodes, 1)
        print(f"[TD(λ)] Epoch {epoch}: avg_loss = {avg_loss:.4f}")


def evaluate(model: ValueModel, episodes: List[dict], args, device, split_name: str = "test"):
    """
    Evaluate the value function using Monte Carlo returns as ground truth.
    
    Reports MSE between predicted values and actual discounted returns.
    """
    pairs = build_mc_targets(episodes, args.gamma)
    if not pairs:
        return None
    
    ds = MCDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mc)
    
    total_loss = 0.0
    total_count = 0
    
    model.eval()
    with torch.no_grad():
        for states, targets in loader:
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits)
            
            loss = nn.functional.mse_loss(values, targets, reduction='sum')
            total_loss += loss.item()
            total_count += len(states)
    
    model.train()
    
    avg_loss = total_loss / max(total_count, 1)
    print(f"[Eval:{split_name}] MSE = {avg_loss:.4f}")
    
    if args.wandb_project and wandb is not None:
        wandb.log({f"eval/{split_name}_mse": avg_loss})
    
    return avg_loss


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
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None and cfg_path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def parse_args():
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
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--lmbda", type=float, default=0.9, help="Lambda for TD(λ).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze transformer weights.")
    parser.add_argument("--wandb-project", default=None, help="If set, log metrics to this W&B project.")
    parser.add_argument("--save-path", default=None, help="Path to save trained model.")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args(remaining)


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.wandb_project and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    print(f"Loading training data from {args.train_path}")
    train_episodes = load_episodes(args.train_path, use_parsed_only=True)
    print(f"Loaded {len(train_episodes)} training episodes")
    
    test_episodes: List[dict] = []
    if os.path.exists(args.test_path):
        print(f"Loading test data from {args.test_path}")
        test_episodes = load_episodes(args.test_path, use_parsed_only=True)
        print(f"Loaded {len(test_episodes)} test episodes")

    print(f"Initializing value model from {args.model_name}")
    model = ValueModel(args.model_name, freeze_base=args.freeze_base).to(device)

    if args.algo in ("td0", "all"):
        print("\n=== Training with TD(0) ===")
        transitions = build_td0_transitions(train_episodes)
        train_td0(model, transitions, args, device)
        if test_episodes:
            evaluate(model, test_episodes, args, device, split_name="test_td0")

    if args.algo in ("mc", "all"):
        print("\n=== Training with Monte Carlo ===")
        pairs = build_mc_targets(train_episodes, args.gamma)
        train_mc(model, pairs, args, device)
        if test_episodes:
            evaluate(model, test_episodes, args, device, split_name="test_mc")

    if args.algo in ("tdlambda", "all"):
        print(f"\n=== Training with TD(λ={args.lmbda}) ===")
        train_td_lambda(model, train_episodes, args, device)
        if test_episodes:
            evaluate(model, test_episodes, args, device, split_name="test_tdlambda")

    if args.save_path:
        print(f"\nSaving model to {args.save_path}")
        torch.save(model.state_dict(), args.save_path)

    if args.wandb_project and wandb is not None:
        wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()