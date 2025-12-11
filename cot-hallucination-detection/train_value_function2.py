"""
Train value functions for the CoT-as-MRP dataset using TD(0), TD(lambda), or Monte Carlo.
States come from the cumulative reasoning steps extracted by data/process_qwen_responses.py.

This corrected version includes:
- Improved ValueModel architecture with better pooling and selective unfreezing
- Stable training with target networks for TD methods
- MSE loss instead of BCE for continuous value regression
- Better hyperparameter defaults
- Proper initialization and gradient clipping
"""

import argparse
import json
import os
from copy import deepcopy
from dataclasses import dataclass
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
# Model Architecture
# -----------------------------


class ResidualBlock(nn.Module):
    """Residual block for value head."""
    
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ValueModel(nn.Module):
    """Value function estimator for reasoning steps."""
    
    def __init__(
        self,
        base_model_name: str,
        freeze_base: bool = True,
        unfreeze_last_n_layers: int = 2,
        pooling_strategy: str = "mean",
        head_hidden: int = 512,
        head_layers: int = 3,
        head_dropout: float = 0.1,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base = AutoModel.from_pretrained(base_model_name)
        self.pooling_strategy = pooling_strategy
        
        # Selective freezing: unfreeze last few layers for better adaptation
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
            
            # Unfreeze last N transformer layers
            if unfreeze_last_n_layers > 0:
                if hasattr(self.base, 'encoder') and hasattr(self.base.encoder, 'layer'):
                    layers = self.base.encoder.layer
                elif hasattr(self.base, 'layers'):
                    layers = self.base.layers
                else:
                    layers = []
                
                for layer in layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                print(f"Unfroze last {unfreeze_last_n_layers} transformer layers")
        
        hidden = self.base.config.hidden_size
        
        # Pooling components
        if pooling_strategy == "attention":
            self.query = nn.Parameter(torch.zeros(1, 1, hidden))
            nn.init.normal_(self.query, mean=0.0, std=0.02)
            self.attn = nn.MultiheadAttention(
                hidden, attn_heads, batch_first=True, dropout=head_dropout
            )
            self.layer_norm = nn.LayerNorm(hidden)
        
        # Build value head
        self.value_head = self._build_value_head(
            hidden, head_hidden, head_layers, head_dropout
        )
        
        # Initialize value head to output near zero initially
        self._init_value_head()
    
    def _build_value_head(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> nn.Module:
        """Build a deeper MLP value head."""
        if num_layers == 1:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _init_value_head(self):
        """Initialize value head to output small values."""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pool_embeddings(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings into a single vector per sequence."""
        if self.pooling_strategy == "last_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_indices, seq_lengths]
        
        elif self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size())
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        
        elif self.pooling_strategy == "attention":
            batch_size = hidden.size(0)
            query = self.query.expand(batch_size, -1, -1)
            key_padding_mask = attention_mask == 0
            attn_out, _ = self.attn(
                query, hidden, hidden, key_padding_mask=key_padding_mask
            )
            pooled = self.layer_norm(attn_out.squeeze(1))
        
        elif self.pooling_strategy == "cls":
            pooled = hidden[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode text and predict scalar value for each sequence."""
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self._pool_embeddings(hidden, attention_mask)
        values = self.value_head(pooled).squeeze(-1)
        return values
    
    def encode(self, texts: List[str], device: torch.device):
        """Tokenize a batch of strings to tensors on the target device."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


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
        reward = float(row.get("label", 0))
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
            target = (gamma ** (T - t - 1)) * R
            pairs.append((state, target))
    return pairs


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
# Training Utils
# -----------------------------


def create_target_network(model):
    """Create a copy of the model for stable target computation."""
    target_model = deepcopy(model)
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    return target_model


def update_target_network(model, target_model, tau=1.0):
    """Update target network via Polyak averaging or hard update."""
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# -----------------------------
# Training loops
# -----------------------------


def train_td0(model: ValueModel, transitions: List[tuple], args, device):
    """Stable one-step TD with target network."""
    target_model = create_target_network(model).to(device)
    
    ds = TransitionDataset(transitions)
    loader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_transitions
    )
    
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader)
    )
    
    mse_loss = nn.MSELoss()
    
    step = 0
    update_target_every = max(1, len(loader) // 4)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_td_errors = []
        
        for batch_idx, batch in enumerate(loader):
            states, next_states, rewards = batch
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # Get current state values
            input_ids, attn = model.encode(list(states), device)
            values = model(input_ids, attn)
            
            # Get next state values from TARGET network (stable)
            with torch.no_grad():
                next_values = torch.zeros_like(values)
                valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
                
                if valid_indices:
                    ns_texts = [next_states[i] for i in valid_indices]
                    ns_ids, ns_attn = target_model.encode(ns_texts, device)
                    ns_values = target_model(ns_ids, ns_attn)
                    
                    for idx, v in zip(valid_indices, ns_values):
                        next_values[idx] = v
            
            # TD target: r + gamma * V_target(s')
            targets = rewards + args.gamma * next_values
            
            # Compute TD error for monitoring
            td_error = (values - targets).abs().mean()
            epoch_td_errors.append(td_error.item())
            
            # MSE loss
            loss = mse_loss(values, targets)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            
            # Periodic target network update
            if (batch_idx + 1) % update_target_every == 0:
                update_target_network(model, target_model, tau=0.005)
            
            if args.wandb_project and wandb is not None:
                wandb.log({
                    "loss_td0": loss.item(),
                    "td_error": td_error.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": step
                })
            step += 1
        
        # Hard update target network at end of epoch
        update_target_network(model, target_model, tau=1.0)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_td_error = sum(epoch_td_errors) / len(epoch_td_errors)
        print(f"[TD0] epoch {epoch} avg_loss={avg_loss:.4f} avg_td_error={avg_td_error:.4f}")
        
        # Log sample predictions
        if epoch % max(1, args.epochs // 5) == 0:
            with torch.no_grad():
                sample_vals = values[:min(5, len(values))].cpu().numpy()
                sample_targets = targets[:min(5, len(targets))].cpu().numpy()
                print(f"  Sample values: {sample_vals}")
                print(f"  Sample targets: {sample_targets}")


def train_mc(model: ValueModel, pairs: List[tuple], args, device):
    """Stable Monte Carlo training."""
    ds = MCDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader)
    )
    
    mse_loss = nn.MSELoss()
    
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        for batch in loader:
            states, targets = batch
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            
            input_ids, attn = model.encode(list(states), device)
            values = model(input_ids, attn)
            
            loss = mse_loss(values, targets)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            
            if args.wandb_project and wandb is not None:
                wandb.log({
                    "loss_mc": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": step
                })
            step += 1
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[MC] epoch {epoch} avg_loss={avg_loss:.4f}")
        
        # Log sample predictions periodically
        if epoch % max(1, args.epochs // 5) == 0:
            with torch.no_grad():
                sample_vals = values[:min(5, len(values))].cpu().numpy()
                sample_targets = targets[:min(5, len(targets))].cpu().numpy()
                print(f"  Sample values: {sample_vals}")
                print(f"  Sample targets: {sample_targets}")


def train_td_lambda(model: ValueModel, episodes: Sequence[dict], args, device):
    """Stable TD(lambda) training with target network."""
    target_model = create_target_network(model).to(device)
    
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=1e-5
    )
    
    mse_loss = nn.MSELoss()
    
    step = 0
    update_target_every = max(1, len(episodes) // 4)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        for ep_idx, ep in enumerate(episodes):
            states = ep["states"]
            R = ep["reward"]
            rewards = [0.0] * (len(states) - 1) + [R]

            # Predict current values
            ids, attn = model.encode(states, device)
            values = model(ids, attn)
            
            # Use target network for bootstrap
            with torch.no_grad():
                values_target = target_model(ids, attn)

            # Compute lambda-returns with target network
            T = len(states)
            lambda_returns = []
            next_return = 0.0
            
            for t in reversed(range(T)):
                v_next = values_target[t + 1] if t + 1 < T else 0.0
                next_return = rewards[t] + args.gamma * (
                    (1 - args.lmbda) * v_next + args.lmbda * next_return
                )
                lambda_returns.insert(0, next_return)
            
            targets = torch.tensor(lambda_returns, dtype=torch.float32, device=device)
            
            loss = mse_loss(values, targets)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            epoch_losses.append(loss.item())
            
            # Periodic target update
            if (ep_idx + 1) % update_target_every == 0:
                update_target_network(model, target_model, tau=0.005)
            
            if args.wandb_project and wandb is not None:
                wandb.log({
                    "loss_td_lambda": loss.item(),
                    "epoch": epoch,
                    "step": step
                })
            step += 1
        
        # Hard update at epoch end
        update_target_network(model, target_model, tau=1.0)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[TD-lambda] epoch {epoch} avg_loss={avg_loss:.4f}")


def eval_mc(model: ValueModel, pairs: List[tuple], args, device, split_name: str):
    """Evaluate MC loss on a split without updating weights."""
    if not pairs:
        return None
    
    model.eval()
    ds = MCDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    mse_loss = nn.MSELoss(reduction="sum")
    
    total_loss = 0.0
    total_count = 0
    all_values = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            states, targets = batch
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            
            input_ids, attn = model.encode(list(states), device)
            values = model(input_ids, attn)
            
            loss = mse_loss(values, targets)
            total_loss += loss.item()
            total_count += len(states)
            
            all_values.extend(values.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    
    avg_loss = total_loss / max(total_count, 1)
    
    # Log statistics
    import numpy as np
    print(f"[MC eval:{split_name}] loss={avg_loss:.4f}")
    print(f"  Value stats: mean={np.mean(all_values):.3f} std={np.std(all_values):.3f}")
    print(f"  Target stats: mean={np.mean(all_targets):.3f} std={np.std(all_targets):.3f}")
    
    if args.wandb_project and wandb is not None:
        wandb.log({
            f"loss_mc_{split_name}": avg_loss,
            f"value_mean_{split_name}": np.mean(all_values),
            f"value_std_{split_name}": np.std(all_values),
        })
    
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
    except Exception:
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
    parser.add_argument("--algo", choices=["td0", "tdlambda", "mc", "all"], default="mc")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze transformer weights.")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=2, 
                        help="Number of last transformer layers to unfreeze.")
    parser.add_argument("--pooling-strategy", default="mean", 
                        choices=["last_token", "mean", "attention", "cls"],
                        help="Token pooling strategy.")
    parser.add_argument("--head-hidden", type=int, default=512, help="Hidden width of value head MLP.")
    parser.add_argument("--head-layers", type=int, default=3, help="Number of layers in value head.")
    parser.add_argument("--head-dropout", type=float, default=0.1, help="Dropout rate in value head MLP.")
    parser.add_argument("--attn-heads", type=int, default=4, help="Attention heads for token pooling.")
    parser.add_argument("--wandb-project", default=None, help="If set, log metrics to this W&B project.")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args(remaining)


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.wandb_project and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Load data
    print(f"Loading training data from {args.train_path}")
    train_episodes = load_episodes(args.train_path, use_parsed_only=True)
    print(f"Loaded {len(train_episodes)} train episodes")
    
    # Data statistics
    rewards = [ep['reward'] for ep in train_episodes]
    print(f"Reward statistics:")
    print(f"  Mean reward: {sum(rewards)/len(rewards):.3f}")
    print(f"  Num incorrect (reward=1): {sum(rewards)}")
    print(f"  Num correct (reward=0): {len(rewards) - sum(rewards)}")
    
    state_lengths = [len(ep['states']) for ep in train_episodes]
    print(f"Episode length statistics:")
    print(f"  Min: {min(state_lengths)}, Max: {max(state_lengths)}, Mean: {sum(state_lengths)/len(state_lengths):.1f}")
    
    test_episodes: List[dict] = []
    if os.path.exists(args.test_path):
        print(f"Loading test data from {args.test_path}")
        test_episodes = load_episodes(args.test_path, use_parsed_only=True)
        print(f"Loaded {len(test_episodes)} test episodes")

    # Initialize model
    print(f"\nInitializing ValueModel:")
    print(f"  Base model: {args.model_name}")
    print(f"  Freeze base: {args.freeze_base}")
    print(f"  Unfreeze last N layers: {args.unfreeze_last_n_layers}")
    print(f"  Pooling strategy: {args.pooling_strategy}")
    print(f"  Value head layers: {args.head_layers}")
    
    model = ValueModel(
        args.model_name,
        freeze_base=args.freeze_base,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        pooling_strategy=args.pooling_strategy,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        attn_heads=args.attn_heads,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Training
    print(f"\nStarting training with algorithm: {args.algo}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    if args.algo in ("tdlambda", "all"):
        print(f"  Lambda: {args.lmbda}")
    
    if args.algo in ("td0", "all"):
        print("\n" + "="*50)
        print("Training TD(0)")
        print("="*50)
        transitions = build_td0_transitions(train_episodes, args.gamma)
        print(f"Built {len(transitions)} transitions")
        train_td0(model, transitions, args, device)
        if test_episodes:
            mc_pairs_test = build_mc_targets(test_episodes, args.gamma)
            eval_mc(model, mc_pairs_test, args, device, split_name="test")

    if args.algo in ("mc", "all"):
        print("\n" + "="*50)
        print("Training Monte Carlo")
        print("="*50)
        mc_pairs = build_mc_targets(train_episodes, args.gamma)
        print(f"Built {len(mc_pairs)} state-target pairs")
        train_mc(model, mc_pairs, args, device)
        if test_episodes:
            mc_pairs_test = build_mc_targets(test_episodes, args.gamma)
            eval_mc(model, mc_pairs_test, args, device, split_name="test")

    if args.algo in ("tdlambda", "all"):
        print("\n" + "="*50)
        print("Training TD(lambda)")
        print("="*50)
        train_td_lambda(model, train_episodes, args, device)
        if test_episodes:
            mc_pairs_test = build_mc_targets(test_episodes, args.gamma)
            eval_mc(model, mc_pairs_test, args, device, split_name="test")

    if args.wandb_project and wandb is not None:
        wandb.finish()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()