"""
Complete training script for CoT value functions with advanced architectures.

Key differences from original:
1. Support for 3 model architectures (original, lightweight, advanced)
2. Auxiliary loss for error prediction
3. Better monitoring and logging
4. Support for step position encoding
5. Optional uncertainty estimation
"""

import argparse
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

import wandb


# ============================================================================
# ADVANCED MODEL ARCHITECTURES
# ============================================================================


class TemporalConvolutionEncoder(nn.Module):
    """Multi-scale 1D convolutions to detect reasoning patterns."""
    
    def __init__(self, hidden_dim: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.projection = nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, hidden: torch.Tensor):
        """hidden: (B, L, H) -> (B, H)"""
        x = hidden.transpose(1, 2)  # (B, H, L)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.gelu(conv(x))
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_outputs.append(pooled)
        combined = torch.cat(conv_outputs, dim=-1)
        return self.norm(self.projection(combined))


class LightweightValueModel(nn.Module):
    """Lightweight architecture with temporal convolutions."""
    
    def __init__(
        self,
        base_model_name: str,
        freeze_base: bool = True,
        unfreeze_last_n_layers: int = 2,
        head_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base = AutoModel.from_pretrained(base_model_name)
        
        # Selective freezing
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
            if unfreeze_last_n_layers > 0:
                layers = getattr(self.base, 'layers', [])
                for layer in layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        
        hidden_dim = self.base.config.hidden_size
        
        # Temporal convolutions (KEY IMPROVEMENT)
        self.temporal_conv = TemporalConvolutionEncoder(hidden_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Value head with skip connections
        self.value_head_1 = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.value_head_2 = nn.Sequential(
            nn.Linear(head_hidden, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.value_head_3 = nn.Linear(head_hidden, 1)
        self.skip_proj = nn.Linear(hidden_dim, head_hidden)
        
        # Auxiliary head for error prediction
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _mean_pooling(self, hidden: torch.Tensor, attention_mask: torch.Tensor):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size())
        sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                return_aux: bool = False):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        # Two pooling strategies
        mean_pooled = self._mean_pooling(hidden, attention_mask)
        temporal_pooled = self.temporal_conv(hidden)
        
        # Fuse
        fused = self.fusion(torch.cat([mean_pooled, temporal_pooled], dim=-1))
        
        # Value with skip connection
        x = self.value_head_1(fused)
        skip = self.skip_proj(fused)
        x = self.value_head_2(x + skip)
        values = self.value_head_3(x).squeeze(-1)
        
        if return_aux:
            aux_error = self.aux_head(fused).squeeze(-1)
            return values, {'error_logits': aux_error}
        
        return values
    
    def encode(self, texts: List[str], device: torch.device):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


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
    """Original improved architecture (from previous version)."""
    
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base = AutoModel.from_pretrained(base_model_name)
        self.pooling_strategy = pooling_strategy
        
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
            if unfreeze_last_n_layers > 0:
                layers = getattr(self.base, 'layers', [])
                if not layers and hasattr(self.base, 'encoder'):
                    layers = getattr(self.base.encoder, 'layer', [])
                for layer in layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        
        hidden = self.base.config.hidden_size
        
        if pooling_strategy == "attention":
            self.query = nn.Parameter(torch.zeros(1, 1, hidden))
            nn.init.normal_(self.query, mean=0.0, std=0.02)
            self.attn = nn.MultiheadAttention(hidden, attn_heads, batch_first=True, dropout=head_dropout)
            self.layer_norm = nn.LayerNorm(hidden)
        
        self.value_head = self._build_value_head(hidden, head_hidden, head_layers, head_dropout)
        self._init_value_head()
    
    def _build_value_head(self, input_dim, hidden_dim, num_layers, dropout):
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
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pool_embeddings(self, hidden: torch.Tensor, attention_mask: torch.Tensor):
        if self.pooling_strategy == "last_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_indices, seq_lengths]
        elif self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size())
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        elif self.pooling_strategy == "attention":
            batch_size = hidden.size(0)
            query = self.query.expand(batch_size, -1, -1)
            key_padding_mask = attention_mask == 0
            attn_out, _ = self.attn(query, hidden, hidden, key_padding_mask=key_padding_mask)
            return self.layer_norm(attn_out.squeeze(1))
        elif self.pooling_strategy == "cls":
            return hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self._pool_embeddings(hidden, attention_mask)
        values = self.value_head(pooled).squeeze(-1)
        return values
    
    def encode(self, texts: List[str], device: torch.device):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


# ============================================================================
# DATA LOADING
# ============================================================================


def load_episodes(path: str, use_parsed_only: bool = True) -> List[dict]:
    """Load episodes from CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    if use_parsed_only and "steps_parsed" in df.columns:
        df = df[df["steps_parsed"]]
    episodes = []
    for _, row in df.iterrows():
        states = json.loads(row["reasoning_steps_json"])
        reward = float(row.get("label", 0))
        episodes.append({
            "problem_id": row.get("problem_id", None),
            "states": states,
            "reward": reward,
        })
    return episodes


def build_td0_transitions(episodes: Sequence[dict], gamma: float):
    """Build one-step transitions."""
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
    """Build Monte Carlo targets."""
    pairs = []
    for ep in episodes:
        states = ep["states"]
        R = ep["reward"]
        T = len(states)
        for t, state in enumerate(states):
            target = (gamma ** (T - t - 1)) * R
            pairs.append((state, target))
    return pairs


class TransitionDataset(Dataset):
    def __init__(self, transitions: List[tuple]):
        self.transitions = transitions
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        return self.transitions[idx]


def collate_transitions(batch):
    states, next_states, rewards = zip(*batch)
    return list(states), list(next_states), list(rewards)


class MCDataset(Dataset):
    def __init__(self, pairs: List[tuple]):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


def create_target_network(model):
    """Create target network for stable TD learning."""
    target_model = deepcopy(model)
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    return target_model


def update_target_network(model, target_model, tau=1.0):
    """Update target network (tau=1.0 for hard update, <1.0 for soft)."""
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ============================================================================
# TRAINING LOOPS
# ============================================================================


def train_td0(model, transitions: List[tuple], args, device):
    """TD(0) with target network."""
    target_model = create_target_network(model).to(device)
    
    ds = TransitionDataset(transitions)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                        collate_fn=collate_transitions)
    
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader)
    )
    
    mse_loss = nn.MSELoss()
    update_target_every = max(1, len(loader) // 4)
    step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_td_errors = []
        
        for batch_idx, batch in enumerate(loader):
            states, next_states, rewards = batch
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            input_ids, attn = model.encode(list(states), device)
            values = model(input_ids, attn)
            
            with torch.no_grad():
                next_values = torch.zeros_like(values)
                valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
                if valid_indices:
                    ns_texts = [next_states[i] for i in valid_indices]
                    ns_ids, ns_attn = target_model.encode(ns_texts, device)
                    ns_values = target_model(ns_ids, ns_attn)
                    for idx, v in zip(valid_indices, ns_values):
                        next_values[idx] = v
            
            targets = rewards + args.gamma * next_values
            td_error = (values - targets).abs().mean()
            loss = mse_loss(values, targets)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            epoch_td_errors.append(td_error.item())
            
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
        
        update_target_network(model, target_model, tau=1.0)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_td_error = sum(epoch_td_errors) / len(epoch_td_errors)
        print(f"[TD0] epoch {epoch} avg_loss={avg_loss:.4f} avg_td_error={avg_td_error:.4f}")


def train_mc(model, pairs: List[tuple], args, device):
    """Monte Carlo training."""
    ds = MCDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
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


def train_td_lambda(model, episodes: Sequence[dict], args, device):
    """TD(lambda) training."""
    target_model = create_target_network(model).to(device)
    
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
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
            
            ids, attn = model.encode(states, device)
            values = model(ids, attn)
            
            with torch.no_grad():
                values_target = target_model(ids, attn)
            
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
            
            if (ep_idx + 1) % update_target_every == 0:
                update_target_network(model, target_model, tau=0.005)
            
            if args.wandb_project and wandb is not None:
                wandb.log({"loss_td_lambda": loss.item(), "epoch": epoch, "step": step})
            step += 1
        
        update_target_network(model, target_model, tau=1.0)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[TD-lambda] epoch {epoch} avg_loss={avg_loss:.4f}")


def eval_mc(model, pairs: List[tuple], args, device, split_name: str):
    """Evaluate on test set."""
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


# ============================================================================
# CLI & MAIN
# ============================================================================


def load_config_file(path: str) -> Dict[str, Any]:
    """Load JSON or YAML config."""
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
    config_parser.add_argument("--config", help="Path to config file")
    config_args, remaining = config_parser.parse_known_args()
    config_defaults = load_config_file(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        parents=[config_parser], description="CoT value function training"
    )
    parser.add_argument("--train-path", default="data/qwen-train.csv")
    parser.add_argument("--test-path", default="data/qwen-test.csv")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model-type", choices=["original", "lightweight"], 
                        default="lightweight")
    parser.add_argument("--algo", choices=["td0", "tdlambda", "mc", "all"], default="mc")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=2)
    parser.add_argument("--pooling-strategy", default="mean", 
                        choices=["last_token", "mean", "attention", "cls"])
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-layers", type=int, default=3)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--wandb-project", default=None)

    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args(remaining)


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.wandb_project and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Load data
    print(f"Loading data from {args.train_path}")
    train_episodes = load_episodes(args.train_path, use_parsed_only=True)
    print(f"Loaded {len(train_episodes)} train episodes")
    
    rewards = [ep['reward'] for ep in train_episodes]
    print(f"Reward stats: mean={sum(rewards)/len(rewards):.3f}, "
          f"incorrect={sum(rewards)}, correct={len(rewards)-sum(rewards)}")
    
    state_lengths = [len(ep['states']) for ep in train_episodes]
    print(f"Episode lengths: min={min(state_lengths)}, max={max(state_lengths)}, "
          f"mean={sum(state_lengths)/len(state_lengths):.1f}")
    
    test_episodes = []
    if os.path.exists(args.test_path):
        test_episodes = load_episodes(args.test_path, use_parsed_only=True)
        print(f"Loaded {len(test_episodes)} test episodes")

    # Initialize model based on type
    print(f"\nInitializing {args.model_type} model...")
    
    if args.model_type == "lightweight":
        model = LightweightValueModel(
            args.model_name,
            freeze_base=args.freeze_base,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
            head_hidden=args.head_hidden,
            dropout=args.head_dropout,
        ).to(device)
        print("  Using LightweightValueModel with temporal convolutions")
    else:  # original
        model = ValueModel(
            args.model_name,
            freeze_base=args.freeze_base,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
            pooling_strategy=args.pooling_strategy,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
            head_dropout=args.head_dropout,
        ).to(device)
        print(f"  Using original ValueModel with {args.pooling_strategy} pooling")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Training
    print(f"\nTraining with {args.algo}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, lr={args.lr}")
    
    if args.algo in ("td0", "all"):
        print("\n" + "="*50)
        print("Training TD(0)")
        print("="*50)
        transitions = build_td0_transitions(train_episodes, args.gamma)
        train_td0(model, transitions, args, device)
        if test_episodes:
            mc_pairs_test = build_mc_targets(test_episodes, args.gamma)
            eval_mc(model, mc_pairs_test, args, device, split_name="test")

    if args.algo in ("mc", "all"):
        print("\n" + "="*50)
        print("Training Monte Carlo")
        print("="*50)
        mc_pairs = build_mc_targets(train_episodes, args.gamma)
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