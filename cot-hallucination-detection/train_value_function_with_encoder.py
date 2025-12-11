"""
Train value functions for the CoT-as-MRP dataset using TD(0), TD(lambda), or Monte Carlo.

The value function V(s) estimates P(incorrect | state s), i.e., the probability that the 
reasoning trajectory will lead to an incorrect final answer given the current state.

Uses encoder models (e.g., BGE, E5) for embeddings with an MLP value head.
"""

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    For an episode with states [s_0, s_1, ..., s_{T-1}] and terminal reward R:
    
    Non-terminal transitions (t < T-1):
        (s_t, s_{t+1}, 0)  - reward is 0
    
    Terminal transition (t = T-1):
        (s_{T-1}, None, R)  - reward is R, next_state is None
    
    TD(0) update: V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
    For terminal: V(s') = 0, so target = r = R
    """
    transitions = []
    for ep in episodes:
        states = ep["states"]
        R = ep["reward"]  # 1 if incorrect, 0 if correct
        T = len(states)
        
        for t in range(T):
            is_terminal = (t == T - 1)
            next_state = None if is_terminal else states[t + 1]
            reward = R if is_terminal else 0.0
            transitions.append((states[t], next_state, reward))
    
    return transitions


def build_mc_targets(episodes: Sequence[dict], gamma: float) -> List[tuple]:
    """
    Build Monte Carlo targets (state, discounted_return) pairs.
    
    For an episode with states [s_0, s_1, ..., s_{T-1}] and terminal reward R:
    
    The return from state s_t is the discounted sum of future rewards.
    Since only the terminal state has reward R:
        G_t = γ^{T-1-t} * R
    
    Examples (γ=1.0, T=3, R=1):
        G_0 = 1.0^2 * 1 = 1
        G_1 = 1.0^1 * 1 = 1  
        G_2 = 1.0^0 * 1 = 1
    
    Examples (γ=0.9, T=3, R=1):
        G_0 = 0.9^2 * 1 = 0.81
        G_1 = 0.9^1 * 1 = 0.9
        G_2 = 0.9^0 * 1 = 1.0
    """
    pairs = []
    for ep in episodes:
        states = ep["states"]
        R = ep["reward"]  # 1 if incorrect, 0 if correct
        T = len(states)
        
        for t in range(T):
            # Number of steps to terminal = T - 1 - t
            discounted_return = (gamma ** (T - 1 - t)) * R
            pairs.append((states[t], discounted_return))
    
    return pairs


def verify_algorithms():
    """
    Verify TD algorithms are correct with a simple test case.
    
    Test episode: 3 states, incorrect answer (R=1)
    """
    print("=" * 60)
    print("VERIFYING TD ALGORITHMS")
    print("=" * 60)
    
    # Test episode
    test_ep = {
        "states": ["step0", "step1", "step2"],
        "reward": 1.0,  # Incorrect answer
    }
    gamma = 0.9
    lmbda = 0.8
    
    # Simulate value estimates: V(s0)=0.3, V(s1)=0.5, V(s2)=0.7
    mock_values = [0.3, 0.5, 0.7]
    
    print(f"\nTest episode: {len(test_ep['states'])} states, R={test_ep['reward']}")
    print(f"Mock values: V(s0)={mock_values[0]}, V(s1)={mock_values[1]}, V(s2)={mock_values[2]}")
    print(f"gamma={gamma}, lambda={lmbda}")
    
    # 1. Test MC targets
    print("\n--- Monte Carlo ---")
    mc_pairs = build_mc_targets([test_ep], gamma)
    for i, (state, target) in enumerate(mc_pairs):
        expected = (gamma ** (2 - i)) * 1.0
        status = "✓" if abs(target - expected) < 1e-6 else "✗"
        print(f"  G_{i} = {target:.4f} (expected: {expected:.4f}) {status}")
    
    # 2. Test TD(0) transitions
    print("\n--- TD(0) ---")
    transitions = build_td0_transitions([test_ep])
    for i, (state, next_state, reward) in enumerate(transitions):
        is_terminal = next_state is None
        if is_terminal:
            # TD target = r + γ * 0 = R
            expected_target = reward
            print(f"  s_{i}: r={reward}, s'=None (terminal)")
            print(f"       TD target = {reward} + {gamma}*0 = {expected_target:.4f} ✓")
        else:
            # TD target = r + γ * V(s')
            v_next = mock_values[i + 1]
            expected_target = reward + gamma * v_next
            print(f"  s_{i}: r={reward}, s'=s_{i+1}")
            print(f"       TD target = {reward} + {gamma}*{v_next} = {expected_target:.4f} ✓")
    
    # 3. Test λ-returns manually
    print("\n--- TD(λ) λ-returns ---")
    # Compute λ-returns by hand
    # G_2^λ = R = 1.0 (terminal)
    G2 = 1.0
    # G_1^λ = r_1 + γ * [(1-λ)*V(s2) + λ*G_2^λ]
    #       = 0 + 0.9 * [(1-0.8)*0.7 + 0.8*1.0]
    #       = 0.9 * [0.2*0.7 + 0.8*1.0]
    #       = 0.9 * [0.14 + 0.8]
    #       = 0.9 * 0.94 = 0.846
    G1 = 0 + gamma * ((1 - lmbda) * mock_values[2] + lmbda * G2)
    # G_0^λ = r_0 + γ * [(1-λ)*V(s1) + λ*G_1^λ]
    #       = 0 + 0.9 * [(1-0.8)*0.5 + 0.8*0.846]
    #       = 0.9 * [0.1 + 0.6768]
    #       = 0.9 * 0.7768 = 0.69912
    G0 = 0 + gamma * ((1 - lmbda) * mock_values[1] + lmbda * G1)
    
    print(f"  G_2^λ = {G2:.4f} (terminal, = R)")
    print(f"  G_1^λ = 0 + {gamma}*[(1-{lmbda})*{mock_values[2]} + {lmbda}*{G2:.4f}] = {G1:.4f}")
    print(f"  G_0^λ = 0 + {gamma}*[(1-{lmbda})*{mock_values[1]} + {lmbda}*{G1:.4f}] = {G0:.4f}")
    
    # Verify: as λ→1, G_t^λ → MC return
    # as λ→0, G_t^λ → TD(0) target
    print("\n  Sanity checks:")
    mc_G0 = gamma ** 2 * 1.0  # = 0.81
    td0_G0 = 0 + gamma * mock_values[1]  # = 0.9 * 0.5 = 0.45
    print(f"    MC return G_0 = {mc_G0:.4f}")
    print(f"    TD(0) target for s_0 = {td0_G0:.4f}")
    print(f"    λ-return G_0^λ = {G0:.4f} (should be between TD(0) and MC) ", end="")
    if td0_G0 <= G0 <= mc_G0 or mc_G0 <= G0 <= td0_G0:
        print("✓")
    else:
        print("✗")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


# -----------------------------
# Model
# -----------------------------


class ValueModel(nn.Module):
    """
    Value function V(s) that estimates P(incorrect | state s).
    
    Uses an encoder model (BGE, E5, etc.) for embeddings with an MLP value head.
    Output is a logit; apply sigmoid to get probability.
    """
    
    def __init__(
        self,
        base_model_name: str = "BAAI/bge-large-en-v1.5",
        freeze_base: bool = True,
        use_bf16: bool = True,
        max_length: int = 512,
        pooling: str = "cls",  # "cls", "mean", or "last"
        head_hidden_dim: int = 256,
        head_num_layers: int = 2,
        head_dropout: float = 0.1,
        normalize_embeddings: bool = False,  # Usually False for classification
    ):
        super().__init__()
        self.freeze_base = freeze_base
        self.max_length = max_length
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        
        # Try different attention implementations
        self.base = None
        for impl in ["flash_attention_2", "sdpa", "eager", None]:
            try:
                kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
                if impl is not None:
                    kwargs["attn_implementation"] = impl
                self.base = AutoModel.from_pretrained(base_model_name, **kwargs)
                print(f"Loaded model with attention: {impl or 'default'}")
                break
            except Exception as e:
                if impl is None:
                    raise RuntimeError(f"Failed to load model: {e}")
                continue
        
        print(f"Model type: {type(self.base).__name__}")
        print(f"Hidden size: {self.base.config.hidden_size}")
        
        # Freeze base if requested
        if freeze_base:
            self.base.eval()
            for p in self.base.parameters():
                p.requires_grad = False
        
        hidden_size = self.base.config.hidden_size
        
        # MLP value head
        # Architecture: Linear → GELU → LayerNorm → Dropout → ... → Linear(1)
        if head_num_layers < 1:
            raise ValueError("head_num_layers must be >= 1")
        
        if head_num_layers == 1:
            # Single linear layer
            self.value_head = nn.Linear(hidden_size, 1)
        else:
            layers = []
            in_dim = hidden_size
            
            for i in range(head_num_layers - 1):
                layers.append(nn.Linear(in_dim, head_hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(head_hidden_dim))
                layers.append(nn.Dropout(head_dropout))
                in_dim = head_hidden_dim
            
            # Final projection to scalar
            layers.append(nn.Linear(in_dim, 1))
            self.value_head = nn.Sequential(*layers)
        
        # Initialize weights properly
        self._init_weights()
        
        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")
    
    def _init_weights(self, prior_prob: float = 0.29):
        """
        Initialize value head weights.
        
        Args:
            prior_prob: Prior probability of positive class (incorrect answers).
                       Used to initialize output bias for faster convergence.
        """
        for name, module in self.value_head.named_modules():
            if isinstance(module, nn.Linear):
                # Kaiming init for GELU (approximates ReLU)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                
                if module.bias is not None:
                    # Check if this is the final layer (outputs 1 value)
                    if module.out_features == 1:
                        # Initialize bias so sigmoid(bias) ≈ prior_prob
                        # This helps with class imbalance
                        initial_bias = math.log(prior_prob / (1 - prior_prob))
                        nn.init.constant_(module.bias, initial_bias)
                    else:
                        nn.init.zeros_(module.bias)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool sequence of hidden states to single vector."""
        if self.pooling == "cls":
            # [CLS] token is the first token for encoder models like BERT, BGE
            pooled = hidden[:, 0, :]
        elif self.pooling == "last":
            # Last non-padded token
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, seq_lengths.long(), :]
        elif self.pooling == "mean":
            # Mean of non-padded tokens
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
            sum_hidden = (hidden * mask).sum(dim=1)  # (B, H)
            lengths = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            pooled = sum_hidden / lengths
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Normalize embeddings (recommended for BGE, E5)
        if self.normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text and predict scalar value (logit) for each sequence."""
        # Use no_grad for frozen base
        context = torch.no_grad() if self.freeze_base else nullcontext()
        
        with context:
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, L, H)
            pooled = self._pool(hidden, attention_mask)  # (B, H)
        
        # Detach and convert to float32 for value head
        if self.freeze_base:
            pooled = pooled.detach().float()
        
        logits = self.value_head(pooled).squeeze(-1)  # (B,)
        return logits

    def encode(self, texts: List[str], device: torch.device):
        """Tokenize a batch of strings."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    def predict_probs(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Return predicted probabilities for a batch of texts."""
        input_ids, attention_mask = self.encode(texts, device)
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)


# -----------------------------
# Datasets
# -----------------------------


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


def collate_mc(batch):
    states, targets = zip(*batch)
    return list(states), list(targets)


def compute_value_loss(logits: torch.Tensor, targets: torch.Tensor, args) -> torch.Tensor:
    """
    Compute loss for value function training.
    
    Args:
        logits: Raw model outputs (B,)
        targets: Target values in [0, 1] (B,)
        args: Must have args.loss and args.output_activation
    
    Returns:
        Scalar loss tensor
    """
    if args.output_activation == "sigmoid":
        values = torch.sigmoid(logits)
    else:
        values = logits
    
    if args.loss == "mse":
        return F.mse_loss(values, targets)
    elif args.loss == "bce":
        if args.output_activation == "sigmoid":
            # Use BCE with logits for numerical stability
            return F.binary_cross_entropy_with_logits(logits, targets)
        else:
            # Clamp values to valid range for BCE
            values = values.clamp(1e-7, 1 - 1e-7)
            return F.binary_cross_entropy(values, targets)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def get_values(logits: torch.Tensor, args) -> torch.Tensor:
    """Apply output activation to get value predictions."""
    if args.output_activation == "sigmoid":
        return torch.sigmoid(logits)
    else:
        return logits


# -----------------------------
# Checkpointing
# -----------------------------


class CheckpointManager:
    """Manages model checkpointing during training."""
    
    def __init__(self, args, model, algo_name: str):
        self.args = args
        self.model = model
        self.algo_name = algo_name
        self.best_loss = float('inf')
        self.checkpoint_dir = args.checkpoint_dir
        
        if args.save_every > 0 or args.save_best:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save(self, path: str, epoch: int = None, loss: float = None):
        """Save complete checkpoint for later evaluation."""
        checkpoint = {
            # Model weights
            'model_state_dict': self.model.state_dict(),
            
            # Model architecture (needed to reconstruct model)
            'model_config': {
                'base_model_name': self.args.model_name,
                'freeze_base': self.args.freeze_base,
                'max_length': self.args.max_length,
                'pooling': self.args.pooling,
                'head_hidden_dim': self.args.head_hidden_dim,
                'head_num_layers': self.args.head_num_layers,
                'head_dropout': self.args.head_dropout,
                'normalize_embeddings': self.args.normalize_embeddings,
            },
            
            # Inference config (needed for prediction)
            'inference_config': {
                'output_activation': self.args.output_activation,
                'gamma': self.args.gamma,
            },
            
            # Training metadata
            'training_info': {
                'algo': self.algo_name,
                'epoch': epoch,
                'loss': loss,
                'lr': self.args.lr,
                'batch_size': self.args.batch_size,
            },
        }
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def on_epoch_end(self, epoch: int, train_loss: float, eval_loss: float = None):
        """Called at the end of each epoch to handle checkpointing."""
        
        # Save every N epochs
        if self.args.save_every > 0 and (epoch + 1) % self.args.save_every == 0:
            path = os.path.join(
                self.checkpoint_dir, 
                f"{self.algo_name}_epoch{epoch+1}.pt"
            )
            self.save(path, epoch=epoch, loss=train_loss)
        
        # Save best model based on eval loss (or train loss if no eval)
        if self.args.save_best:
            current_loss = eval_loss if eval_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                path = os.path.join(self.checkpoint_dir, f"{self.algo_name}_best.pt")
                self.save(path, epoch=epoch, loss=current_loss)
                print(f"  New best loss: {current_loss:.4f}")


def load_checkpoint(path: str, device: str = "cuda") -> tuple:
    """
    Load a saved checkpoint for evaluation.
    
    Args:
        path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        (model, inference_config, training_info)
    
    Example usage:
        model, inf_cfg, train_info = load_checkpoint("checkpoints/mc_best.pt")
        model.eval()
        
        # Get predictions
        texts = ["reasoning step 1", "reasoning step 2"]
        input_ids, attn = model.encode(texts, device)
        logits = model(input_ids, attn)
        
        # Apply activation
        if inf_cfg['output_activation'] == 'sigmoid':
            values = torch.sigmoid(logits)
        else:
            values = logits
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Reconstruct model from saved config
    model_cfg = checkpoint['model_config']
    model = ValueModel(
        base_model_name=model_cfg['base_model_name'],
        freeze_base=model_cfg['freeze_base'],
        max_length=model_cfg['max_length'],
        pooling=model_cfg['pooling'],
        head_hidden_dim=model_cfg['head_hidden_dim'],
        head_num_layers=model_cfg['head_num_layers'],
        head_dropout=model_cfg['head_dropout'],
        normalize_embeddings=model_cfg['normalize_embeddings'],
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['inference_config'], checkpoint['training_info']


# -----------------------------
# Training loops
# -----------------------------


def train_td0(model: ValueModel, transitions: List[tuple], args, device, eval_episodes: List[dict] = None):
    """TD(0) policy evaluation."""
    ds = TransitionDataset(transitions)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_transitions,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Checkpointing
    ckpt = CheckpointManager(args, model, "td0")
    
    accum_steps = args.gradient_accumulation_steps
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        opt.zero_grad()
        
        for batch_idx, (states, next_states, rewards) in enumerate(loader):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # V(s)
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = get_values(logits, args)
            
            # V(s') for non-terminal states
            with torch.no_grad():
                next_values = torch.zeros_like(values)
                non_terminal_idx = [i for i, ns in enumerate(next_states) if ns is not None]
                
                if non_terminal_idx:
                    ns_texts = [next_states[i] for i in non_terminal_idx]
                    ns_ids, ns_attn = model.encode(ns_texts, device)
                    ns_logits = model(ns_ids, ns_attn)
                    ns_values = get_values(ns_logits, args)
                    for i, val in zip(non_terminal_idx, ns_values):
                        next_values[i] = val
            
            # TD target: r + γ * V(s')
            targets = rewards + args.gamma * next_values
            
            loss = compute_value_loss(logits, targets.detach(), args)
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
        
        # Handle remaining gradients
        if num_batches % accum_steps != 0:
            opt.step()
            opt.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[TD0] Epoch {epoch}: loss = {avg_loss:.4f}")
        
        # Evaluate and checkpoint
        eval_loss = None
        if eval_episodes:
            eval_loss = evaluate(model, eval_episodes, args, device, f"epoch{epoch}")
        
        ckpt.on_epoch_end(epoch, avg_loss, eval_loss)


def train_mc(model: ValueModel, pairs: List[tuple], args, device, eval_episodes: List[dict] = None):
    """Monte Carlo policy evaluation."""
    ds = MCDataset(pairs)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mc,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Checkpointing
    ckpt = CheckpointManager(args, model, "mc")
    
    accum_steps = args.gradient_accumulation_steps
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        opt.zero_grad()
        
        for batch_idx, (states, targets) in enumerate(loader):
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            
            loss = compute_value_loss(logits, targets, args)
            loss = loss / accum_steps
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                global_step += 1
                
                if args.wandb_project and wandb is not None:
                    wandb.log({"mc/loss": loss.item() * accum_steps, "mc/step": global_step})
            
            epoch_loss += loss.item() * accum_steps
            num_batches += 1
        
        if num_batches % accum_steps != 0:
            opt.step()
            opt.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[MC] Epoch {epoch}: loss = {avg_loss:.4f}")
        
        # Evaluate and checkpoint
        eval_loss = None
        if eval_episodes:
            eval_loss = evaluate(model, eval_episodes, args, device, f"epoch{epoch}")
        
        ckpt.on_epoch_end(epoch, avg_loss, eval_loss)


def compute_lambda_returns(model: ValueModel, episodes: Sequence[dict], args, device) -> List[tuple]:
    """
    Pre-compute λ-returns for all states using backward recursion.
    
    The λ-return is defined as:
        G_t^λ = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]
    
    For terminal state s_{T-1}:
        G_{T-1}^λ = r_{T-1} + γ * 0 = R  (no next state)
    
    For non-terminal states (where r_t = 0 in our MRP):
        G_t^λ = 0 + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]
    
    Note: G_T^λ = 0 (return after episode ends is 0)
    """
    pairs = []
    
    model.eval()
    with torch.no_grad():
        for ep in episodes:
            states = ep["states"]
            R = ep["reward"]  # Terminal reward (1 if incorrect, 0 if correct)
            T = len(states)
            
            if T == 0:
                continue
            
            # Build reward vector: [0, 0, ..., 0, R]
            rewards = [0.0] * T
            rewards[T - 1] = R
            
            # Get value estimates V(s_t) for all states
            input_ids, attn = model.encode(states, device)
            logits = model(input_ids, attn)
            values = get_values(logits, args).cpu().tolist()
            
            # Backward recursion for λ-returns
            # G_T^λ = 0 (after terminal state)
            lambda_returns = [0.0] * T
            G_next = 0.0  # G_{t+1}^λ, initialized to G_T^λ = 0
            
            for t in reversed(range(T)):
                r_t = rewards[t]
                
                if t == T - 1:
                    # Terminal state: no next state, so V(s_T) = 0
                    # G_{T-1}^λ = r_{T-1} + γ * 0 = R
                    lambda_returns[t] = r_t
                else:
                    # Non-terminal: G_t^λ = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]
                    V_next = values[t + 1]
                    lambda_returns[t] = r_t + args.gamma * (
                        (1 - args.lmbda) * V_next + args.lmbda * G_next
                    )
                
                G_next = lambda_returns[t]
            
            for state, lr in zip(states, lambda_returns):
                pairs.append((state, lr))
    
    model.train()
    return pairs


def train_td_lambda(model: ValueModel, episodes: Sequence[dict], args, device, eval_episodes: List[dict] = None):
    """TD(λ) policy evaluation."""
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Checkpointing
    ckpt = CheckpointManager(args, model, "td_lambda")
    
    accum_steps = args.gradient_accumulation_steps
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"[TD(λ)] Epoch {epoch}: computing λ-returns...")
        pairs = compute_lambda_returns(model, episodes, args, device)
        
        model.train()
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
            
            loss = compute_value_loss(logits, targets, args)
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
        
        if num_batches % accum_steps != 0:
            opt.step()
            opt.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[TD(λ)] Epoch {epoch}: loss = {avg_loss:.4f}")
        
        # Evaluate and checkpoint
        eval_loss = None
        if eval_episodes:
            eval_loss = evaluate(model, eval_episodes, args, device, f"epoch{epoch}")
        
        ckpt.on_epoch_end(epoch, avg_loss, eval_loss)


def evaluate(model: ValueModel, episodes: List[dict], args, device, split_name: str = "test"):
    """Evaluate using MC returns as ground truth."""
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
            
            # Always use MSE for evaluation (comparable across configs)
            values = get_values(logits, args)
            loss = F.mse_loss(values, targets, reduction='sum')
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
    """Load JSON or YAML config, converting kebab-case to snake_case."""
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = cfg_path.read_text()
    
    try:
        import yaml
        raw = yaml.safe_load(text) if cfg_path.suffix in {".yml", ".yaml"} else json.loads(text)
    except ImportError:
        raw = json.loads(text)
    
    return {k.replace("-", "_"): v for k, v in (raw or {}).items()}


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Path to config file")
    config_args, remaining = config_parser.parse_known_args()
    config_defaults = load_config_file(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(parents=[config_parser])
    
    # Data
    parser.add_argument("--train-path", default="data/qwen-train.csv")
    parser.add_argument("--test-path", default="data/qwen-test.csv")
    
    # Model
    parser.add_argument("--model-name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--freeze-base", action="store_true", default=True)
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base")
    parser.add_argument("--pooling", choices=["cls", "mean", "last"], default="cls")
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-num-layers", type=int, default=2)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--normalize-embeddings", action="store_true", default=False)
    parser.add_argument("--no-normalize-embeddings", action="store_false", dest="normalize_embeddings")
    
    # Training
    parser.add_argument("--algo", choices=["td0", "tdlambda", "mc", "all"], default="mc")
    parser.add_argument("--verify", action="store_true", help="Run algorithm verification tests")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--loss", choices=["mse", "bce"], default="mse",
                        help="Loss function: mse (standard RL) or bce (better gradients for sigmoid)")
    parser.add_argument("--output-activation", choices=["sigmoid", "none"], default="sigmoid",
                        help="Output activation: sigmoid bounds to [0,1], none for raw logits")
    
    # System
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default=None)
    
    # Checkpointing
    parser.add_argument("--save-path", default=None, help="Path to save final model")
    parser.add_argument("--save-every", type=int, default=0, 
                        help="Save checkpoint every N epochs (0 = disabled)")
    parser.add_argument("--save-best", action="store_true", default=True,
                        help="Save best model based on eval loss")
    parser.add_argument("--no-save-best", action="store_false", dest="save_best")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for checkpoints")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args(remaining)


def main():
    args = parse_args()
    
    # Run verification if requested
    if args.verify:
        verify_algorithms()
        return
    
    device = torch.device(args.device)

    print("=" * 60)
    print("Configuration:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    # Wandb
    if args.wandb_project:
        if wandb is None:
            print("WARNING: wandb not installed")
        else:
            try:
                run = wandb.init(project=args.wandb_project, config=vars(args))
                print(f"Wandb: {run.url}")
            except Exception as e:
                print(f"WARNING: wandb init failed: {e}")

    # Data
    print(f"\nLoading training data: {args.train_path}")
    train_episodes = load_episodes(args.train_path)
    print(f"Loaded {len(train_episodes)} episodes")
    
    test_episodes = []
    if os.path.exists(args.test_path):
        print(f"Loading test data: {args.test_path}")
        test_episodes = load_episodes(args.test_path)
        print(f"Loaded {len(test_episodes)} test episodes")

    # Model
    print(f"\nInitializing model: {args.model_name}")
    model = ValueModel(
        base_model_name=args.model_name,
        freeze_base=args.freeze_base,
        use_bf16=args.use_bf16,
        max_length=args.max_length,
        pooling=args.pooling,
        head_hidden_dim=args.head_hidden_dim,
        head_num_layers=args.head_num_layers,
        head_dropout=args.head_dropout,
        normalize_embeddings=args.normalize_embeddings,
    ).to(device)

    # Training
    if args.algo in ("td0", "all"):
        print("\n=== TD(0) ===")
        transitions = build_td0_transitions(train_episodes)
        train_td0(model, transitions, args, device, eval_episodes=test_episodes or None)

    if args.algo in ("mc", "all"):
        print("\n=== Monte Carlo ===")
        pairs = build_mc_targets(train_episodes, args.gamma)
        train_mc(model, pairs, args, device, eval_episodes=test_episodes or None)

    if args.algo in ("tdlambda", "all"):
        print(f"\n=== TD(λ={args.lmbda}) ===")
        train_td_lambda(model, train_episodes, args, device, eval_episodes=test_episodes or None)

    # Save final model
    if args.save_path:
        print(f"\nSaving final model to {args.save_path}")
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        
        checkpoint = {
            # Model weights
            'model_state_dict': model.state_dict(),
            
            # Model architecture (needed to reconstruct model)
            'model_config': {
                'base_model_name': args.model_name,
                'freeze_base': args.freeze_base,
                'max_length': args.max_length,
                'pooling': args.pooling,
                'head_hidden_dim': args.head_hidden_dim,
                'head_num_layers': args.head_num_layers,
                'head_dropout': args.head_dropout,
                'normalize_embeddings': args.normalize_embeddings,
            },
            
            # Inference config (needed for prediction)
            'inference_config': {
                'output_activation': args.output_activation,
                'gamma': args.gamma,
            },
            
            # Training metadata
            'training_info': {
                'algo': args.algo,
                'epochs': args.epochs,
                'lr': args.lr,
                'batch_size': args.batch_size,
            },
        }
        torch.save(checkpoint, args.save_path)

    if args.wandb_project and wandb is not None:
        wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()