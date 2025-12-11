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
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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
    
    Uses a Qwen model as encoder with configurable pooling strategy,
    followed by an MLP value head.
    
    Output is a logit; apply sigmoid to get probability.
    """
    
    def __init__(
        self,
        base_model_name: str,
        freeze_base: bool = True,
        use_bf16: bool = True,
        max_length: int = 1024,
        pooling: str = "last",  # "last", "mean", or "max"
        head_hidden_dim: int = 256,
        head_num_layers: int = 2,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.freeze_base = freeze_base
        self.max_length = max_length
        self.pooling = pooling
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model in bfloat16 to save memory
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        
        # Try Flash Attention 2, fall back to SDPA, then eager
        attn_impl = None
        for impl in ["flash_attention_2", "sdpa", "eager"]:
            try:
                self.base = AutoModel.from_pretrained(
                    base_model_name,
                    torch_dtype=dtype,
                    attn_implementation=impl,
                )
                attn_impl = impl
                break
            except (ImportError, ValueError) as e:
                if impl == "eager":
                    raise  # Last resort failed
                continue
        
        print(f"Using attention implementation: {attn_impl}")
        
        if freeze_base:
            self.base.eval()  # Permanently in eval mode
            for p in self.base.parameters():
                p.requires_grad = False
        
        hidden_size = self.base.config.hidden_size
        
        # MLP value head for more capacity
        layers = []
        in_dim = hidden_size
        for i in range(head_num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, head_hidden_dim),
                nn.LayerNorm(head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
            ])
            in_dim = head_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.value_head = nn.Sequential(*layers)
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool sequence of hidden states to single vector."""
        if self.pooling == "last":
            # Get last non-padded token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, seq_lengths.long()]  # (B, H)
        elif self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            lengths = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / lengths
        elif self.pooling == "max":
            # Mask padding with -inf before max
            mask_expanded = attention_mask.unsqueeze(-1)
            hidden_masked = hidden.masked_fill(~mask_expanded.bool(), float('-inf'))
            pooled = hidden_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text and predict scalar value (logit) for each sequence.
        """
        # Use no_grad for frozen base to avoid storing activations
        context = torch.no_grad() if self.freeze_base else nullcontext()
        
        with context:
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, L, H)
            pooled = self._pool(hidden, attention_mask)  # (B, H)
        
        # Detach if frozen so gradient only flows through value_head
        if self.freeze_base:
            pooled = pooled.detach().float()  # Convert to float32 for value head
        
        logits = self.value_head(pooled).squeeze(-1)  # (B,)
        return logits
        
        logits = self.value_head(pooled).squeeze(-1)  # (B,)
        return logits

    def encode(self, texts: List[str], device: torch.device):
        """Tokenize a batch of strings to tensors on the target device."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
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
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_transitions,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    accum_steps = args.gradient_accumulation_steps
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        opt.zero_grad()
        
        for batch_idx, (states, next_states, rewards) in enumerate(loader):
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
                del ns_ids, ns_attn, ns_logits, ns_values
            
            # TD target: r + γ * V(s')
            targets = rewards + args.gamma * next_values
            
            # MSE loss between V(s) and TD target
            loss = nn.functional.mse_loss(values, targets.detach())
            loss = loss / accum_steps
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                global_step += 1
                
                if args.wandb_project and wandb is not None:
                    wandb.log({"td0/loss": loss.item() * accum_steps, "td0/step": global_step})
            
            epoch_loss += loss.item() * accum_steps
            num_batches += 1
            
            del input_ids, attn, logits, values, loss
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        if num_batches % accum_steps != 0:
            opt.step()
            opt.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[TD0] Epoch {epoch}: avg_loss = {avg_loss:.4f}")


def train_mc(model: ValueModel, pairs: List[tuple], args, device):
    """
    Monte Carlo policy evaluation.
    
    For each (state, G) pair where G is the discounted return:
        V(s) <- V(s) + α * [G - V(s)]
    
    We use MSE loss: L = (V(s) - G)^2
    
    Uses gradient accumulation for memory efficiency.
    """
    ds = MCDataset(pairs)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mc,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Gradient accumulation to simulate larger batches
    accum_steps = args.gradient_accumulation_steps
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        opt.zero_grad()
        
        for batch_idx, (states, targets) in enumerate(loader):
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits)  # V(s) in [0, 1]
            
            loss = nn.functional.mse_loss(values, targets)
            loss = loss / accum_steps  # Scale for accumulation
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                global_step += 1
                
                if args.wandb_project and wandb is not None:
                    wandb.log({"mc/loss": loss.item() * accum_steps, "mc/step": global_step})
            
            epoch_loss += loss.item() * accum_steps
            num_batches += 1
            
            # Free memory
            del input_ids, attn, logits, values, loss
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if num_batches % accum_steps != 0:
            opt.step()
            opt.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[MC] Epoch {epoch}: avg_loss = {avg_loss:.4f}")


def compute_lambda_returns(model: ValueModel, episodes: Sequence[dict], args, device) -> List[tuple]:
    """
    Pre-compute λ-returns for all states in all episodes.
    
    Returns list of (state, λ-return) pairs that can be batched like MC.
    """
    pairs = []
    
    model.eval()
    with torch.no_grad():
        for ep in episodes:
            states = ep["states"]
            R = ep["reward"]
            T = len(states)
            
            # Get value estimates for all states in this episode
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits).cpu()  # (T,)
            
            # Compute λ-returns via backward recursion
            lambda_returns = [0.0] * T
            next_lambda_return = 0.0
            
            for t in reversed(range(T)):
                if t == T - 1:
                    # Terminal state: G_{T-1}^λ = R
                    lambda_returns[t] = R
                else:
                    # G_t^λ = γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]
                    v_next = values[t + 1].item()
                    lambda_returns[t] = args.gamma * (
                        (1 - args.lmbda) * v_next + args.lmbda * next_lambda_return
                    )
                next_lambda_return = lambda_returns[t]
            
            # Add (state, λ-return) pairs
            for state, lr in zip(states, lambda_returns):
                pairs.append((state, lr))
            
            del input_ids, attn, logits, values
    
    model.train()
    torch.cuda.empty_cache()
    
    return pairs


def train_td_lambda(model: ValueModel, episodes: Sequence[dict], args, device):
    """
    TD(λ) policy evaluation using forward view (λ-returns).
    
    Strategy: 
    1. Pre-compute λ-returns using current value estimates (like fitted value iteration)
    2. Train on (state, λ-return) pairs with batching (like MC)
    3. Repeat for each epoch (recompute λ-returns with updated values)
    
    This allows batching while still using bootstrapped λ-returns.
    """
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    accum_steps = args.gradient_accumulation_steps
    
    global_step = 0
    for epoch in range(args.epochs):
        # Recompute λ-returns with current value function
        print(f"[TD(λ)] Epoch {epoch}: computing λ-returns...")
        pairs = compute_lambda_returns(model, episodes, args, device)
        
        # Train on pairs like MC
        ds = MCDataset(pairs)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_mc,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        epoch_loss = 0.0
        num_batches = 0
        opt.zero_grad()
        
        for batch_idx, (states, targets) in enumerate(loader):
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits)
            
            loss = nn.functional.mse_loss(values, targets)
            loss = loss / accum_steps
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                global_step += 1
                
                if args.wandb_project and wandb is not None:
                    wandb.log({"td_lambda/loss": loss.item() * accum_steps, "td_lambda/step": global_step})
            
            epoch_loss += loss.item() * accum_steps
            num_batches += 1
            
            del input_ids, attn, logits, values, loss
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        if num_batches % accum_steps != 0:
            opt.step()
            opt.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
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
    loader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_mc,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    total_loss = 0.0
    total_count = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (states, targets) in enumerate(loader):
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = torch.sigmoid(logits)
            
            loss = nn.functional.mse_loss(values, targets, reduction='sum')
            total_loss += loss.item()
            total_count += len(states)
            
            del input_ids, attn, logits, values
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
    
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
    """Load JSON or YAML config dict, converting kebab-case keys to snake_case."""
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
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    
    # Convert kebab-case to snake_case for argparse compatibility
    return {k.replace("-", "_"): v for k, v in raw.items()}


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
    
    # Memory optimization arguments
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, 
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length for tokenization.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers.")
    parser.add_argument("--use-bf16", action="store_true", default=True,
                        help="Use bfloat16 for base model.")
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16",
                        help="Disable bfloat16.")
    
    # Model architecture arguments
    parser.add_argument("--pooling", choices=["last", "mean", "max"], default="last",
                        help="Pooling strategy for sequence embeddings.")
    parser.add_argument("--head-hidden-dim", type=int, default=256,
                        help="Hidden dimension of value head MLP.")
    parser.add_argument("--head-num-layers", type=int, default=2,
                        help="Number of layers in value head MLP.")
    parser.add_argument("--head-dropout", type=float, default=0.1,
                        help="Dropout rate in value head.")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args(remaining)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Debug: print config
    print("=" * 50)
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 50)

    # Initialize wandb
    if args.wandb_project:
        if wandb is None:
            print("WARNING: wandb not installed, skipping logging. Install with: pip install wandb")
        else:
            try:
                run = wandb.init(project=args.wandb_project, config=vars(args))
                print(f"Wandb initialized: {run.url}")
            except Exception as e:
                print(f"WARNING: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")

    print(f"\nLoading training data from {args.train_path}")
    train_episodes = load_episodes(args.train_path, use_parsed_only=True)
    print(f"Loaded {len(train_episodes)} training episodes")
    
    test_episodes: List[dict] = []
    if os.path.exists(args.test_path):
        print(f"Loading test data from {args.test_path}")
        test_episodes = load_episodes(args.test_path, use_parsed_only=True)
        print(f"Loaded {len(test_episodes)} test episodes")

    print(f"Initializing value model from {args.model_name}")
    model = ValueModel(
        args.model_name, 
        freeze_base=args.freeze_base,
        use_bf16=args.use_bf16,
        max_length=args.max_length,
        pooling=args.pooling,
        head_hidden_dim=args.head_hidden_dim,
        head_num_layers=args.head_num_layers,
        head_dropout=args.head_dropout,
    ).to(device)

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