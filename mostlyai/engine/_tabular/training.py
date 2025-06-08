# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import time
import warnings
from itertools import zip_longest
from pathlib import Path
from collections.abc import Callable
from typing import Any

from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, disable_progress_bar
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F

from torch import nn

from opacus import PrivacyEngine, GradSampleModule
from opacus.accountants import PRVAccountant, RDPAccountant, GaussianAccountant
from opacus.utils.batch_memory_manager import wrap_data_loader

# Mixed precision imports
try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    amp = None

try:
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AMP_AVAILABLE = True
except ImportError:
    TORCH_AMP_AVAILABLE = False
    GradScaler = None
    autocast = None

from mostlyai.engine._memory import get_available_ram_for_heuristics
from mostlyai.engine.domain import ModelStateStrategy, DifferentialPrivacyConfig
from mostlyai.engine._tabular.argn import (
    FlatModel,
    ModelSize,
    SequentialModel,
    get_model_units,
    get_no_of_model_parameters,
)
from mostlyai.engine._common import (
    CTXFLT,
    CTXSEQ,
    TGT,
    get_cardinalities,
    get_columns_from_cardinalities,
    get_ctx_sequence_length,
    get_max_data_points_per_sample,
    get_sequence_length_stats,
    get_sub_columns_from_cardinalities,
    get_sub_columns_nested_from_cardinalities,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    SDEC_SUB_COLUMN_PREFIX,
    ProgressCallback,
    ProgressCallbackWrapper,
)
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._training_utils import (
    check_early_training_exit,
    EarlyStopper,
    ModelCheckpoint,
    ProgressMessage,
)
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir

_LOG = logging.getLogger(__name__)


##################
### HEURISTICS ###
##################


def _physical_batch_size_heuristic(
    mem_available_gb: float,
    no_of_records: int,
    no_tgt_data_points: int,
    no_ctx_data_points: int,
    no_of_model_params: int,
) -> int:
    """
    Calculate the physical batch size.

    Args:
        mem_available_gb (float): Available memory in GB.
        no_of_records (int): Number of records in the training dataset.
        no_tgt_data_points (int): Number of target data points per sample.
        no_ctx_data_points (int): Number of context data points per sample.
        no_of_model_params (int): Number of model parameters.

    Returns:
        Batch size (int)
    """
    data_points = no_tgt_data_points + no_ctx_data_points
    min_batch_size = 8
    # scale batch_size corresponding to available memory
    if mem_available_gb >= 32:
        mem_scale = 2.0
    elif mem_available_gb >= 8:
        mem_scale = 1.0
    else:
        mem_scale = 0.5
    # set max_batch_size corresponding to available memory, model params and data points
    if no_of_model_params > 1_000_000_000 or data_points > 100_000:
        max_batch_size = int(8 * mem_scale)
    elif no_of_model_params > 100_000_000 or data_points > 10_000:
        max_batch_size = int(32 * mem_scale)
    elif no_of_model_params > 10_000_000 or data_points > 1_000:
        max_batch_size = int(128 * mem_scale)
    elif no_of_model_params > 1_000_000 or data_points > 100:
        max_batch_size = int(512 * mem_scale)
    else:
        max_batch_size = int(2048 * mem_scale)
    # ensure a minimum number of batches to avoid excessive padding
    min_batches = 64
    batch_size = 2 ** int(np.log2(no_of_records / min_batches)) if no_of_records > 0 else min_batch_size
    return int(np.clip(a=batch_size, a_min=min_batch_size, a_max=max_batch_size))


def _learn_rate_heuristic(batch_size: int) -> float:
    learn_rate = np.round(0.001 * np.sqrt(batch_size / 32), 5)
    return learn_rate


####################
### ADAPTIVE BATCH SIZE ###
####################


class AdaptiveBatchSizeManager:
    """Manages adaptive batch size based on gradient noise and training dynamics."""
    
    def __init__(
        self,
        initial_batch_size: int,
        strategy: str = "gradient_noise",
        growth_factor: float = 1.2,
        shrink_factor: float = 0.8,
        min_batch_size: int = 8,
        max_batch_size: int = 2048,
        gradient_noise_threshold: float = 0.1,
        adaptation_patience: int = 10,
    ):
        self.current_batch_size = initial_batch_size
        self.strategy = strategy
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gradient_noise_threshold = gradient_noise_threshold
        self.adaptation_patience = adaptation_patience
        
        # Internal tracking
        self.gradient_norms_history = []
        self.loss_history = []
        self.steps_since_adaptation = 0
        self.last_adaptation_step = 0
        
    def update_metrics(self, gradient_norm: float, loss: float, step: int):
        """Update internal metrics for adaptation decisions."""
        self.gradient_norms_history.append(gradient_norm)
        self.loss_history.append(loss)
        self.steps_since_adaptation = step - self.last_adaptation_step
        
        # Keep only recent history (last 100 steps)
        if len(self.gradient_norms_history) > 100:
            self.gradient_norms_history = self.gradient_norms_history[-100:]
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
    
    def should_adapt_batch_size(self) -> tuple[bool, str]:
        """Determine if batch size should be adapted and in which direction."""
        if self.steps_since_adaptation < self.adaptation_patience:
            return False, "patience"
        
        if len(self.gradient_norms_history) < 10:
            return False, "insufficient_data"
        
        if self.strategy == "gradient_noise":
            return self._gradient_noise_strategy()
        elif self.strategy == "loss_variance":
            return self._loss_variance_strategy()
        elif self.strategy == "conservative":
            return self._conservative_strategy()
        else:
            return False, "unknown_strategy"
    
    def _gradient_noise_strategy(self) -> tuple[bool, str]:
        """Adapt based on gradient noise levels."""
        recent_norms = self.gradient_norms_history[-20:]  # Last 20 steps
        if len(recent_norms) < 10:
            return False, "insufficient_gradient_data"
        
        # Calculate gradient noise (coefficient of variation)
        mean_norm = np.mean(recent_norms)
        std_norm = np.std(recent_norms)
        
        if mean_norm == 0:
            return False, "zero_gradients"
        
        gradient_noise = std_norm / mean_norm
        
        if gradient_noise > self.gradient_noise_threshold and self.current_batch_size < self.max_batch_size:
            # High noise -> increase batch size for stability
            return True, "increase_for_stability"
        elif gradient_noise < self.gradient_noise_threshold * 0.5 and self.current_batch_size > self.min_batch_size:
            # Low noise -> can afford smaller batch size for faster iterations
            return True, "decrease_for_speed"
        
        return False, "gradient_noise_optimal"
    
    def _loss_variance_strategy(self) -> tuple[bool, str]:
        """Adapt based on loss variance."""
        recent_losses = self.loss_history[-20:]
        if len(recent_losses) < 10:
            return False, "insufficient_loss_data"
        
        loss_variance = np.var(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        if loss_mean == 0:
            return False, "zero_loss"
        
        loss_cv = np.sqrt(loss_variance) / loss_mean
        
        if loss_cv > 0.1 and self.current_batch_size < self.max_batch_size:
            return True, "increase_for_loss_stability"
        elif loss_cv < 0.05 and self.current_batch_size > self.min_batch_size:
            return True, "decrease_for_loss_efficiency"
        
        return False, "loss_variance_optimal"
    
    def _conservative_strategy(self) -> tuple[bool, str]:
        """Conservative strategy - only increase, never decrease."""
        recent_norms = self.gradient_norms_history[-10:]
        if len(recent_norms) < 5:
            return False, "insufficient_data"
        
        # Only increase if gradients are very noisy
        gradient_noise = np.std(recent_norms) / (np.mean(recent_norms) + 1e-8)
        
        if gradient_noise > self.gradient_noise_threshold * 2 and self.current_batch_size < self.max_batch_size:
            return True, "conservative_increase"
        
        return False, "conservative_stable"
    
    def adapt_batch_size(self, step: int) -> tuple[int, str]:
        """Adapt batch size and return new size with reason."""
        should_adapt, reason = self.should_adapt_batch_size()
        
        if not should_adapt:
            return self.current_batch_size, reason
        
        old_batch_size = self.current_batch_size
        
        if "increase" in reason:
            new_batch_size = min(
                int(self.current_batch_size * self.growth_factor),
                self.max_batch_size
            )
        else:  # decrease
            new_batch_size = max(
                int(self.current_batch_size * self.shrink_factor),
                self.min_batch_size
            )
        
        # Round to nearest power of 2 for efficiency
        new_batch_size = 2 ** round(np.log2(new_batch_size))
        new_batch_size = np.clip(new_batch_size, self.min_batch_size, self.max_batch_size)
        
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self.last_adaptation_step = step
            self.steps_since_adaptation = 0
            adaptation_reason = f"{reason}_from_{old_batch_size}_to_{new_batch_size}"
        else:
            adaptation_reason = f"{reason}_no_change"
        
        return self.current_batch_size, adaptation_reason


#####################
### ENHANCED LOSS ###
#####################


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedLoss(nn.Module):
    """Enhanced loss with focal loss and label smoothing options."""
    
    def __init__(
        self, 
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "none"
    ):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(inputs, targets)


#########################
### ADVANCED SCHEDULERS ###
#########################


class WarmupCosineScheduler(LRScheduler):
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
                for base_lr in self.base_lrs
            ]


class CosineRestartScheduler(LRScheduler):
    """Cosine annealing with warm restarts."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_0: int,
        t_mult: int = 2,
        eta_min_ratio: float = 0.01,
        last_epoch: int = -1
    ):
        self.t_0 = t_0
        self.t_mult = t_mult
        self.eta_min_ratio = eta_min_ratio
        self.t_i = t_0
        self.t_cur = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.t_cur == self.t_i:
            self.t_cur = 0
            self.t_i *= self.t_mult
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * self.t_cur / self.t_i))
        self.t_cur += 1
        
        return [
            base_lr * (self.eta_min_ratio + (1 - self.eta_min_ratio) * cosine_factor)
            for base_lr in self.base_lrs
        ]


####################
### LOOKAHEAD OPTIMIZER ###
####################


class Lookahead:
    """Lookahead optimizer wrapper."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        self.slow_weights = []
        for group in self.optimizer.param_groups:
            group_dict = {}
            for p in group['params']:
                if p.requires_grad:
                    group_dict[p] = p.data.clone()
            self.slow_weights.append(group_dict)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            for group, slow_group in zip(self.optimizer.param_groups, self.slow_weights):
                for p in group['params']:
                    if p.requires_grad and p in slow_group:
                        slow_group[p].data.add_(p.data - slow_group[p].data, alpha=self.alpha)
                        p.data.copy_(slow_group[p].data)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none)
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    @property
    def state(self):
        return self.optimizer.state
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


####################
### DATA LOADERS ###
####################


class BatchCollator:
    """
    Enhanced collate function with adaptive sequence sampling.
    For sequence data, it will sample subsequences with advanced strategies.
    """

    def __init__(
        self, 
        is_sequential: bool, 
        max_sequence_window: int | None, 
        device: torch.device,
        adaptive_sampling: bool = False,
        difficulty_progression: float = 0.0
    ):
        self.is_sequential = is_sequential
        self.max_sequence_window = max_sequence_window
        self.device = device
        self.adaptive_sampling = adaptive_sampling
        self.difficulty_progression = difficulty_progression
        self.step_count = 0

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        batch = pd.DataFrame(batch)
        if self.is_sequential and self.max_sequence_window:
            batch = self._slice_sequences_enhanced(batch, self.max_sequence_window)
        batch = self._convert_to_tensors(batch)
        self.step_count += 1
        return batch

    def _convert_to_tensors(self, batch: pd.DataFrame) -> dict[str, torch.Tensor]:
        tensors = {}
        for column in batch.columns:
            if column.startswith(TGT) and self.is_sequential:
                # construct column tensor in single step
                tensors[column] = torch.unsqueeze(
                    torch.tensor(
                        # pad batch-wise to the longest sequence length with 0s
                        np.array(list(zip_longest(*batch[column], fillvalue=0))).T,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    dim=-1,
                )
            elif column.startswith(TGT) and not self.is_sequential:
                # construct column tensor in single step
                tensors[column] = torch.unsqueeze(
                    torch.tensor(batch[column].values, dtype=torch.int64, device=self.device),
                    dim=-1,
                )
            elif column.startswith(CTXFLT):
                # construct column tensor in single step
                tensors[column] = torch.unsqueeze(
                    torch.tensor(batch[column].values, dtype=torch.int64, device=self.device),
                    dim=-1,
                )
            elif column.startswith(CTXSEQ):
                # construct row tensors and convert the list to nested column tensor
                tensors[column] = torch.unsqueeze(
                    torch.nested.as_nested_tensor(
                        [torch.tensor(row, dtype=torch.int64, device=self.device) for row in batch[column]],
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    dim=-1,
                )
        return tensors

    def _slice_sequences_enhanced(self, batch: pd.DataFrame, max_sequence_window: int) -> pd.DataFrame:
        """Enhanced sequence slicing with adaptive strategies."""
        # determine sequence lengths of current batch
        tgt_columns = [col for col in batch.columns if col.startswith(TGT)]
        seq_lens = batch[tgt_columns[0]].copy().str.len().values

        if self.adaptive_sampling:
            # Adaptive sampling based on difficulty progression
            difficulty_factor = min(1.0, self.difficulty_progression * self.step_count / 1000.0)
            
            # Gradually shift from easier (start/end) to harder (random) sampling
            if difficulty_factor < 0.3:
                # Early training: focus on start and end
                flip = np.random.random()
                if flip < 0.5:
                    sampling_strategy = "start"
                else:
                    sampling_strategy = "end"
            elif difficulty_factor < 0.7:
                # Mid training: mix of strategies
                flip = np.random.random()
                if flip < 0.2:
                    sampling_strategy = "start"
                elif flip < 0.4:
                    sampling_strategy = "end"
                else:
                    sampling_strategy = "random"
            else:
                # Late training: more challenging random sampling
                sampling_strategy = "random"
        else:
            # Original sampling logic
            flip = np.random.random()
            if flip < 0.3:
                sampling_strategy = "start"
            elif flip < 0.4:
                sampling_strategy = "end"
            else:
                sampling_strategy = "random"

        # Apply sampling strategy
        if sampling_strategy == "start":
            sel_idxs = [np.arange(0, min(max_sequence_window, seq_len)) for seq_len in seq_lens]
        elif sampling_strategy == "end":
            sel_idxs = [np.arange(max(0, seq_len - max_sequence_window), seq_len) for seq_len in seq_lens]
        else:  # random
            # Content-aware windowing: adjust window size based on sequence complexity
            if self.adaptive_sampling:
                # Vary window size based on sequence length for better coverage
                adjusted_windows = []
                for seq_len in seq_lens:
                    if seq_len <= max_sequence_window:
                        adjusted_windows.append(seq_len)
                    else:
                        # Use slightly varied window sizes for diversity
                        variance = max(1, max_sequence_window // 10)
                        adjusted_window = max_sequence_window + np.random.randint(-variance, variance + 1)
                        adjusted_window = max(1, min(adjusted_window, seq_len))
                        adjusted_windows.append(adjusted_window)
            else:
                adjusted_windows = [max_sequence_window] * len(seq_lens)
            
            start_idxs = []
            sel_idxs = []
            for seq_len, window_size in zip(seq_lens, adjusted_windows):
                if seq_len <= window_size:
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, seq_len - window_size + 1)
                start_idxs.append(start_idx)
                sel_idxs.append(np.arange(start_idx, start_idx + window_size))

        # Apply selection to batch
        tgt_col_idxs = [batch.columns.get_loc(c) for c in tgt_columns]
        rows = []
        for row_idx, batch_row in enumerate(batch.itertuples(index=False)):
            cells = []
            for col_idx, batch_cell in enumerate(batch_row):
                if col_idx in tgt_col_idxs:
                    cells.append([batch_cell[i] for i in sel_idxs[row_idx]])
                else:
                    cells.append(batch_cell)
            rows.append(cells)

        return pd.DataFrame(rows, columns=batch.columns, index=batch.index)


#####################
### TRAINING LOOP ###
#####################


class TabularModelCheckpoint(ModelCheckpoint):
    def _save_model_weights(self, model: torch.nn.Module):
        if isinstance(model, GradSampleModule):
            state_dict = model._module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, self.workspace.model_tabular_weights_path)

    def _clear_model_weights(self) -> None:
        self.workspace.model_tabular_weights_path.unlink(missing_ok=True)

    def model_weights_path_exists(self) -> bool:
        return self.workspace.model_tabular_weights_path.exists()


def _calculate_sample_losses(
    model: FlatModel | SequentialModel | GradSampleModule, 
    data: dict[str, torch.Tensor],
    criterion: EnhancedLoss,
    use_mixed_precision: bool = False,
    mixed_precision_backend: str | None = None
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
        
        # Forward pass with mixed precision if enabled
        if use_mixed_precision and mixed_precision_backend == "torch_amp":
            with autocast():
                output, _ = model(data, mode="trn")
        else:
            output, _ = model(data, mode="trn")

    tgt_cols = (
        list(model.tgt_cardinalities.keys())
        if not isinstance(model, GradSampleModule)
        else model._module.tgt_cardinalities.keys()
    )
    if isinstance(model, SequentialModel) or (
        isinstance(model, GradSampleModule) and isinstance(model._module, SequentialModel)
    ):
        slen_cols = [k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)]

        # generate masks for SLEN and time step
        slen_mask = torch.zeros_like(data[slen_cols[0]], dtype=torch.int64)
        for slen_col in slen_cols:
            slen_mask |= data[slen_col] != 0  # mask loss for padded rows, which have SLEN=0
        slen_mask = slen_mask.squeeze(-1)
        time_step_mask = torch.zeros_like(slen_mask, dtype=torch.int64)
        time_step_mask[:, 0] = 10  # mask loss for all time steps except the first one, and emphasize that one by 10x

        # calculate per column losses
        sidx_cols = {k for k in data if k.startswith(SIDX_SUB_COLUMN_PREFIX)}
        sdec_cols = {k for k in data if k.startswith(SDEC_SUB_COLUMN_PREFIX)}
        losses_by_column = []
        for col in tgt_cols:
            if col in slen_cols:
                # mask out SLEN for steps > 1
                mask = time_step_mask
            elif col in sidx_cols or col in sdec_cols:
                # SIDX and SDEC columns need to be present in the computation graph for DP to work
                # so we're only masking them instead of skipping them completely
                mask = torch.zeros_like(slen_mask, dtype=torch.int64)
            else:
                # mask out paddings
                mask = slen_mask

            column_loss = criterion(output[col].transpose(1, 2), data[col].squeeze(2))
            masked_loss = torch.sum(column_loss * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
            losses_by_column.append(masked_loss)
    else:
        losses_by_column = [criterion(output[col], data[col].squeeze(1)) for col in tgt_cols]
    # sum up column level losses to get overall losses at sample level
    losses = torch.sum(torch.stack(losses_by_column, dim=0), dim=0)
    return losses


# gradient tracking is not needed for validation steps, disable it to save memory
@torch.no_grad()
def _calculate_val_loss(
    model: FlatModel | SequentialModel,
    val_dataloader: DataLoader,
    criterion: EnhancedLoss,
    use_mixed_precision: bool = False,
    mixed_precision_backend: str | None = None,
) -> float:
    val_sample_losses: list[torch.Tensor] = []
    model.eval()
    for step_data in val_dataloader:
        step_losses = _calculate_sample_losses(
            model, step_data, criterion, use_mixed_precision, mixed_precision_backend
        )
        val_sample_losses.extend(step_losses.detach())
    model.train()
    val_sample_losses: torch.Tensor = torch.stack(val_sample_losses, dim=0)
    val_loss_avg = torch.mean(val_sample_losses).item()
    return val_loss_avg


def _calculate_average_trn_loss(trn_sample_losses: list[torch.Tensor], n: int | None = None) -> float | None:
    if len(trn_sample_losses) == 0:
        return None
    trn_losses_latest = torch.stack(trn_sample_losses, dim=0)
    if n is not None:
        trn_losses_latest = trn_losses_latest[-n:]
    trn_loss = torch.mean(trn_losses_latest).item()
    return trn_loss


################
### TRAINING ###
################


def train(
    *,
    model: str = "MOSTLY_AI/Medium",
    max_training_time: float = 14400.0,  # 10 days
    max_epochs: float = 100.0,  # 100 epochs
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    max_sequence_window: int = 100,
    enable_flexible_generation: bool = True,
    differential_privacy: DifferentialPrivacyConfig | dict | None = None,
    upload_model_data_callback: Callable | None = None,
    model_state_strategy: ModelStateStrategy | str = ModelStateStrategy.reset,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
    # Learning Rate Enhancements
    use_warmup_cosine: bool = True,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.01,
    use_cosine_restarts: bool = False,
    restart_period: int = 20,
    restart_mult: int = 2,
    # Advanced Regularization
    gradient_clip_norm: float | None = 1.0,
    weight_decay_schedule: bool = True,
    initial_weight_decay: float = 0.01,
    final_weight_decay: float = 0.001,
    # Loss Function Enhancements
    use_focal_loss: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    # Sequence Sampling Enhancements
    adaptive_sampling: bool = True,
    difficulty_progression: float = 0.001,
    # Optimizer Enhancements
    use_lookahead: bool = False,
    lookahead_k: int = 5,
    lookahead_alpha: float = 0.5,
    adaptive_betas: bool = True,
    beta1_schedule: tuple[float, float] = (0.9, 0.95),
    beta2_schedule: tuple[float, float] = (0.999, 0.99),
    # Adaptive Batch Size Strategy
    adaptive_batch_size: bool = False,
    batch_size_strategy: str = "gradient_noise",  # "gradient_noise", "loss_variance", "conservative"
    batch_size_growth_factor: float = 1.2,
    batch_size_shrink_factor: float = 0.8,
    min_batch_size: int = 8,
    max_batch_size: int = 2048,
    gradient_noise_threshold: float = 0.1,
    batch_adaptation_patience: int = 10,
    # Mixed Precision Training
    use_mixed_precision: bool = False,
    mixed_precision_backend: str | None = None,
    mixed_precision_opt_level: str = "O1",
):
    """Enhanced training function with advanced optimization techniques.
    
    Args:
        # Original parameters (unchanged)
        model: Model size ("MOSTLY_AI/Small", "MOSTLY_AI/Medium", "MOSTLY_AI/Large")
        max_training_time: Maximum training time in hours
        max_epochs: Maximum number of epochs
        batch_size: Batch size (auto-determined if None)
        gradient_accumulation_steps: Gradient accumulation steps
        max_sequence_window: Maximum sequence window for sequential data
        enable_flexible_generation: Enable flexible column ordering
        differential_privacy: Differential privacy configuration
        upload_model_data_callback: Callback for uploading model data
        model_state_strategy: Strategy for handling existing model state
        device: Device to use for training
        workspace_dir: Workspace directory path
        update_progress: Progress callback function
        
        # Learning Rate Enhancements
        use_warmup_cosine: Use warmup + cosine annealing scheduler
        warmup_epochs: Number of warmup epochs
        min_lr_ratio: Minimum learning rate as ratio of initial LR
        use_cosine_restarts: Use cosine annealing with warm restarts
        restart_period: Initial restart period
        restart_mult: Restart period multiplier
        
        # Advanced Regularization
        gradient_clip_norm: Gradient clipping norm (None to disable)
        weight_decay_schedule: Use scheduled weight decay
        initial_weight_decay: Initial weight decay value
        final_weight_decay: Final weight decay value
        
        # Loss Function Enhancements
        use_focal_loss: Use focal loss instead of CrossEntropy
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        
        # Sequence Sampling Enhancements
        adaptive_sampling: Use adaptive sequence sampling
        difficulty_progression: Rate of difficulty progression
        
        # Optimizer Enhancements
        use_lookahead: Wrap optimizer with Lookahead
        lookahead_k: Lookahead k parameter
        lookahead_alpha: Lookahead alpha parameter
        adaptive_betas: Use adaptive beta scheduling
        beta1_schedule: (initial_beta1, final_beta1) for adaptive scheduling
        beta2_schedule: (initial_beta2, final_beta2) for adaptive scheduling
    """
    _LOG.info("ENHANCED_TRAIN_TABULAR started")
    t0 = time.time()
    workspace_dir = ensure_workspace_dir(workspace_dir)
    workspace = Workspace(workspace_dir)
    with ProgressCallbackWrapper(
        update_progress, progress_messages_path=workspace.model_progress_messages_path
    ) as progress:
        _LOG.info(f"numpy={version('numpy')}, pandas={version('pandas')}")
        _LOG.info(f"torch={version('torch')}, opacus={version('opacus')}")
        device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        _LOG.info(f"{device=}")
        torch.set_default_dtype(torch.float32)

        # Log enhancement settings
        _LOG.info("=== ENHANCEMENT SETTINGS ===")
        _LOG.info(f"Learning Rate: warmup_cosine={use_warmup_cosine}, cosine_restarts={use_cosine_restarts}")
        _LOG.info(f"Regularization: gradient_clip={gradient_clip_norm}, weight_decay_schedule={weight_decay_schedule}")
        _LOG.info(f"Loss: focal_loss={use_focal_loss}, label_smoothing={label_smoothing}")
        _LOG.info(f"Sampling: adaptive={adaptive_sampling}, difficulty_progression={difficulty_progression}")
        _LOG.info(f"Optimizer: lookahead={use_lookahead}, adaptive_betas={adaptive_betas}")

        has_context = workspace.ctx_stats.path.exists()
        tgt_stats = workspace.tgt_stats.read()
        ctx_stats = workspace.ctx_stats.read()
        is_sequential = tgt_stats["is_sequential"]
        _LOG.info(f"{is_sequential=}")
        trn_cnt = tgt_stats["no_of_training_records"]
        val_cnt = tgt_stats["no_of_validation_records"]
        tgt_cardinalities = get_cardinalities(tgt_stats)
        ctx_cardinalities = get_cardinalities(ctx_stats) if has_context else {}
        tgt_sub_columns = get_sub_columns_from_cardinalities(tgt_cardinalities)
        ctx_nested_sub_columns = get_sub_columns_nested_from_cardinalities(ctx_cardinalities, "processor")
        ctxflt_sub_columns = ctx_nested_sub_columns.get(CTXFLT, [])
        ctxseq_sub_columns = ctx_nested_sub_columns.get(CTXSEQ, [])

        # set defaults
        max_training_time = max(0.0, max_training_time) * 60  # convert to seconds
        _LOG.info(f"{max_training_time=}s")
        max_epochs = max(0.0, max_epochs)
        max_epochs_cap = math.ceil((trn_cnt + val_cnt) / 50)
        if max_epochs_cap < max_epochs:
            _LOG.info(f"{max_epochs=} -> max_epochs={max_epochs_cap} due to small sample size")
            max_epochs = max_epochs_cap
        else:
            _LOG.info(f"{max_epochs=}")
        model_sizes = {
            "MOSTLY_AI/Small": ModelSize.S,
            "MOSTLY_AI/Medium": ModelSize.M,
            "MOSTLY_AI/Large": ModelSize.L,
        }
        if model not in model_sizes:
            raise ValueError(f"model {model} not supported")
        model_size = model_sizes[model]
        _LOG.info(f"{model_size=}")
        _LOG.info(f"{enable_flexible_generation=}")
        with_dp = differential_privacy is not None
        _LOG.info(f"{with_dp=}")
        _LOG.info(f"{model_state_strategy=}")

        # initialize callbacks
        upload_model_data_callback = upload_model_data_callback or (lambda *args, **kwargs: None)

        # early exit if there is not enough data to train the model
        # in such scenario, training model is not created
        # and weights are not stored, so generation must be resilient to that
        if check_early_training_exit(workspace=workspace, trn_cnt=trn_cnt, val_cnt=val_cnt):
            _LOG.warning("not enough data to train model; skipping training")
            return

        # determine column order for training
        if enable_flexible_generation:
            # random column order for each batch
            trn_column_order = None
        else:
            # fixed column order based on cardinalities
            tgt_cardinalities = get_cardinalities(tgt_stats)
            trn_column_order = get_columns_from_cardinalities(tgt_cardinalities)

        # gather sequence length stats for heuristics
        tgt_seq_len_stats = get_sequence_length_stats(tgt_stats)
        tgt_seq_len_median = tgt_seq_len_stats["median"]
        tgt_seq_len_max = tgt_seq_len_stats["max"]
        max_sequence_window = np.clip(max_sequence_window, a_min=1, a_max=tgt_seq_len_max)
        _LOG.info(f"{max_sequence_window=}")
        ctx_seq_len_median = get_ctx_sequence_length(ctx_stats, key="median")

        # the line below fixes issue with growing epoch time for later epochs
        # https://discuss.pytorch.org/t/training-time-gets-slower-and-slower-on-cpu/145483
        torch.set_flush_denormal(True)

        _LOG.info("create training model")
        model_checkpoint = TabularModelCheckpoint(workspace=workspace)
        argn: SequentialModel | FlatModel
        if is_sequential:
            argn = SequentialModel(
                tgt_cardinalities=tgt_cardinalities,
                ctx_cardinalities=ctx_cardinalities,
                tgt_seq_len_median=tgt_seq_len_median,
                tgt_seq_len_max=tgt_seq_len_max,
                ctxseq_len_median=ctx_seq_len_median,
                model_size=model_size,
                column_order=trn_column_order,
                device=device,
                with_dp=with_dp,  # this flag decides whether the model is initialized with LSTM or DPLSTM layers
            )
        else:
            argn = FlatModel(
                tgt_cardinalities=tgt_cardinalities,
                ctx_cardinalities=ctx_cardinalities,
                ctxseq_len_median=ctx_seq_len_median,
                model_size=model_size,
                column_order=trn_column_order,
                device=device,
            )
        _LOG.info(f"model class: {argn.__class__.__name__}")

        if isinstance(model_state_strategy, str):
            model_state_strategy = ModelStateStrategy(model_state_strategy)
        if not model_checkpoint.model_weights_path_exists():
            _LOG.info(f"model weights not found; change strategy from {model_state_strategy} to RESET")
            model_state_strategy = ModelStateStrategy.reset
        _LOG.info(f"{model_state_strategy=}")
        if model_state_strategy in [ModelStateStrategy.resume, ModelStateStrategy.reuse]:
            _LOG.info("load existing model weights")
            torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
            load_model_weights(model=argn, path=workspace.model_tabular_weights_path, device=device)
        else:  # ModelStateStrategy.reset
            _LOG.info("remove existing checkpoint files")
            model_checkpoint.clear_checkpoint()

        # check how to handle existing progress state
        last_progress_message = progress.get_last_progress_message()
        if last_progress_message and model_state_strategy == ModelStateStrategy.resume:
            epoch = last_progress_message.get("epoch", 0.0)
            steps = last_progress_message.get("steps", 0)
            samples = last_progress_message.get("samples", 0)
            initial_lr = last_progress_message.get("learn_rate", None)
            total_time_init = last_progress_message.get("total_time", 0.0)
        else:
            epoch = 0.0
            steps = 0
            samples = 0
            initial_lr = None
            total_time_init = 0.0
            progress.reset_progress_messages()
        _LOG.info(f"start training progress from {epoch=}, {steps=}")

        argn.to(device)
        no_of_model_params = get_no_of_model_parameters(argn)
        _LOG.info(f"{no_of_model_params=}")

        # persist model configs
        model_units = get_model_units(argn)
        model_configs = {
            "model_id": model,
            "model_units": model_units,
            "enable_flexible_generation": enable_flexible_generation,
        }
        workspace.model_configs.write(model_configs)

        # heuristics for batch_size and for initial learn_rate
        mem_available_gb = get_available_ram_for_heuristics() / 1024**3
        no_tgt_data_points = get_max_data_points_per_sample(tgt_stats)
        no_ctx_data_points = get_max_data_points_per_sample(ctx_stats)
        
        # Determine initial batch size
        if batch_size is None:
            initial_batch_size = _physical_batch_size_heuristic(
                mem_available_gb=mem_available_gb,
                no_of_records=trn_cnt,
                no_tgt_data_points=no_tgt_data_points,
                no_ctx_data_points=no_ctx_data_points,
                no_of_model_params=no_of_model_params,
            )
        else:
            initial_batch_size = batch_size
            
        # Setup adaptive batch size manager
        if adaptive_batch_size:
            batch_size_manager = AdaptiveBatchSizeManager(
                initial_batch_size=initial_batch_size,
                strategy=batch_size_strategy,
                growth_factor=batch_size_growth_factor,
                shrink_factor=batch_size_shrink_factor,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                gradient_noise_threshold=gradient_noise_threshold,
                adaptation_patience=batch_adaptation_patience,
            )
            current_batch_size = initial_batch_size
            _LOG.info(f"Adaptive batch size enabled: strategy={batch_size_strategy}, initial_size={initial_batch_size}")
        else:
            batch_size_manager = None
            current_batch_size = initial_batch_size
            
        batch_size = current_batch_size  # Use current batch size for the rest of initialization
        if gradient_accumulation_steps is None:
            # for TABULAR the batch size is typically large, so we use step=1 as default
            gradient_accumulation_steps = 1

        # setup params for input pipeline
        batch_size = max(1, min(batch_size, trn_cnt))
        gradient_accumulation_steps = max(1, min(gradient_accumulation_steps, trn_cnt // batch_size))
        trn_batch_size = batch_size * gradient_accumulation_steps
        trn_steps = max(1, trn_cnt // trn_batch_size)
        val_batch_size = max(1, min(batch_size, val_cnt))
        val_steps = max(1, val_cnt // val_batch_size)

        if initial_lr is None:
            initial_lr = _learn_rate_heuristic(trn_batch_size)
        if is_sequential:
            # reduce val_batch_size to reduce padding for validation batches,
            # which speeds up compute, plus it results in a more stable val_loss
            val_batch_size = val_batch_size // 2

        # Enhanced batch collator with adaptive sampling
        batch_collator = BatchCollator(
            is_sequential=is_sequential, 
            max_sequence_window=max_sequence_window, 
            device=device,
            adaptive_sampling=adaptive_sampling,
            difficulty_progression=difficulty_progression
        )
        disable_progress_bar()
        trn_dataset = load_dataset("parquet", data_files=[str(p) for p in workspace.encoded_data_trn.fetch_all()])[
            "train"
        ]
        trn_dataloader = DataLoader(
            dataset=trn_dataset,
            shuffle=True,
            # either DP logical batch size or grad accumulation physical batch size
            batch_size=trn_batch_size if with_dp else batch_size,
            collate_fn=batch_collator,
        )
        val_dataset = load_dataset("parquet", data_files=[str(p) for p in workspace.encoded_data_val.fetch_all()])[
            "train"
        ]
        val_dataloader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=val_batch_size,
            collate_fn=batch_collator,
        )

        _LOG.info(f"{trn_cnt=}, {val_cnt=}")
        _LOG.info(f"{len(tgt_sub_columns)=}, {len(ctxflt_sub_columns)=}, {len(ctxseq_sub_columns)=}")
        if len(tgt_cardinalities) > 0:
            tgt_cardinalities_deciles = list(
                np.quantile(
                    list(tgt_cardinalities.values()),
                    np.arange(0, 1.1, 0.1),
                    method="lower",
                )
            )
            _LOG.info(f"{tgt_cardinalities_deciles=}")
        if len(ctx_cardinalities) > 0:
            ctx_cardinalities_deciles = list(
                np.quantile(
                    list(ctx_cardinalities.values()),
                    np.arange(0, 1.1, 0.1),
                    method="lower",
                )
            )
            _LOG.info(f"{ctx_cardinalities_deciles=}")
        _LOG.info(f"{trn_batch_size=}, {val_batch_size=}")
        _LOG.info(f"{trn_steps=}, {val_steps=}")
        _LOG.info(f"{batch_size=}, {gradient_accumulation_steps=}, {initial_lr=}")

        # Enhanced loss function
        enhanced_criterion = EnhancedLoss(
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            reduction="none"
        )
        _LOG.info(f"Loss function: focal={use_focal_loss}, smoothing={label_smoothing}")

        early_stopper = EarlyStopper(val_loss_patience=4)
        
        # Enhanced optimizer with adaptive betas and weight decay
        current_weight_decay = initial_weight_decay
        if adaptive_betas:
            current_beta1, current_beta2 = beta1_schedule[0], beta2_schedule[0]
        else:
            current_beta1, current_beta2 = 0.9, 0.999
            
        optimizer = torch.optim.AdamW(
            params=argn.parameters(), 
            lr=initial_lr,
            betas=(current_beta1, current_beta2),
            weight_decay=current_weight_decay
        )
        
        # Wrap with Lookahead if enabled
        if use_lookahead:
            optimizer = Lookahead(optimizer, k=lookahead_k, alpha=lookahead_alpha)
            _LOG.info(f"Lookahead optimizer enabled: k={lookahead_k}, alpha={lookahead_alpha}")

        # Setup mixed precision
        grad_scaler = None
        if use_mixed_precision:
            if mixed_precision_backend == "apex":
                argn, optimizer_for_amp = amp.initialize(
                    argn, 
                    optimizer.optimizer if use_lookahead else optimizer,
                    opt_level=mixed_precision_opt_level
                )
                if use_lookahead:
                    optimizer.optimizer = optimizer_for_amp
                else:
                    optimizer = optimizer_for_amp
                _LOG.info(f"Apex mixed precision initialized: {mixed_precision_opt_level}")
            elif mixed_precision_backend == "torch_amp":
                grad_scaler = GradScaler()
                _LOG.info("PyTorch native AMP initialized")

        # Enhanced learning rate scheduler
        if use_cosine_restarts:
            lr_scheduler = CosineRestartScheduler(
                optimizer=optimizer.optimizer if use_lookahead else optimizer,
                t_0=restart_period,
                t_mult=restart_mult,
                eta_min_ratio=min_lr_ratio
            )
            _LOG.info(f"Cosine restart scheduler: period={restart_period}, mult={restart_mult}")
        elif use_warmup_cosine:
            total_steps = int(max_epochs * trn_steps)
            warmup_steps = int(warmup_epochs * trn_steps)
            lr_scheduler = WarmupCosineScheduler(
                optimizer=optimizer.optimizer if use_lookahead else optimizer,
                warmup_epochs=warmup_steps,
                total_epochs=total_steps,
                min_lr_ratio=min_lr_ratio
            )
            _LOG.info(f"Warmup cosine scheduler: warmup={warmup_epochs}, min_ratio={min_lr_ratio}")
        else:
            lr_scheduler: LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer.optimizer if use_lookahead else optimizer,
                factor=0.5,
                patience=2,
                min_lr=0.1 * initial_lr,
            )
            
        if (
            model_state_strategy == ModelStateStrategy.resume
            and model_checkpoint.optimizer_and_lr_scheduler_paths_exist()
        ):
            # restore the full states of optimizer and lr_scheduler when possible
            # otherwise, only the learning rate from the last progress message will be restored
            _LOG.info("restore optimizer and LR scheduler states")
            base_optimizer = optimizer.optimizer if use_lookahead else optimizer
            base_optimizer.load_state_dict(
                torch.load(workspace.model_optimizer_path, map_location=device, weights_only=True)
            )
            lr_scheduler.load_state_dict(
                torch.load(workspace.model_lr_scheduler_path, map_location=device, weights_only=True)
            )

        if device.type == "cuda":
            # this can help accelerate GPU compute
            torch.backends.cudnn.benchmark = True

        if with_dp:
            if isinstance(differential_privacy, DifferentialPrivacyConfig):
                dp_config = differential_privacy.model_dump()
            else:
                dp_config = DifferentialPrivacyConfig(**differential_privacy).model_dump()
            dp_max_epsilon = dp_config.get("max_epsilon") or float("inf")
            dp_total_delta = dp_config.get("delta", 1e-5)
            # take the actual value_protection_epsilon from the stats
            dp_value_protection_epsilon = (ctx_stats.get("value_protection_epsilon_spent") or 0.0) + (
                tgt_stats.get("value_protection_epsilon_spent") or 0.0
            )
            # the implementation of PRV accountant seems to have numerical and memory issues for small noise multiplier
            # therefore, we choose RDP instead as it is more stable and provides comparable privacy guarantees
            dp_accountant = "rdp"  # hard-coded for now
            _LOG.info(f"{dp_config=}, {dp_accountant=}")
            privacy_engine = PrivacyEngine(accountant=dp_accountant)
            if model_state_strategy == ModelStateStrategy.resume and workspace.model_dp_accountant_path.exists():
                _LOG.info("restore DP accountant state")
                torch.serialization.add_safe_globals([getattr, PRVAccountant, RDPAccountant, GaussianAccountant])
                privacy_engine.accountant.load_state_dict(
                    torch.load(workspace.model_dp_accountant_path, map_location=device, weights_only=True)
                )
            # Opacus will return the modified objects
            # - model: wrapped in GradSampleModule and contains additional hooks for computing per-sample gradients
            # - optimizer: wrapped in DPOptimizer and will do different operations during virtual steps and logical steps
            # - dataloader: the dataloader with batch_sampler=UniformWithReplacementSampler (for Poisson sampling)
            
            # For DP training, we need to pass the base optimizer to Opacus
            # Lookahead will be applied after DP wrapping if enabled
            base_optimizer = optimizer.optimizer if use_lookahead else optimizer
            argn, dp_optimizer, trn_dataloader = privacy_engine.make_private(
                module=argn,
                optimizer=base_optimizer,
                data_loader=trn_dataloader,
                noise_multiplier=dp_config.get("noise_multiplier"),
                max_grad_norm=dp_config.get("max_grad_norm"),
                poisson_sampling=True,
            )
            
            # Re-wrap with Lookahead if enabled
            if use_lookahead:
                optimizer = Lookahead(dp_optimizer, k=lookahead_k, alpha=lookahead_alpha)
                _LOG.info("Re-wrapped DP optimizer with Lookahead")
            else:
                optimizer = dp_optimizer
                
            # this further wraps the dataloader with batch_sampler=BatchSplittingSampler to achieve gradient accumulation
            # it will split the sampled logical batches into smaller sub-batches with batch_size
            trn_dataloader = wrap_data_loader(
                data_loader=trn_dataloader, max_batch_size=batch_size, optimizer=dp_optimizer
            )
        else:
            privacy_engine = None
            dp_config, dp_total_delta, dp_accountant = None, None, None

        progress_message = None
        start_trn_time = time.time()
        last_msg_time = time.time()
        trn_data_iter = iter(trn_dataloader)
        trn_sample_losses: list[torch.Tensor] = []
        do_stop = False
        current_lr = initial_lr
        total_steps = int(max_epochs * trn_steps)
        
        # Gradient norm tracking for monitoring
        gradient_norms = []
        
        # infinite loop over training steps, until we decide to stop
        # either because of max_epochs, max_training_time or early_stopping
        while not do_stop:
            is_checkpoint = 0
            steps += 1
            epoch = steps / trn_steps

            # Update adaptive parameters based on progress
            progress_ratio = min(1.0, steps / total_steps) if total_steps > 0 else 0.0
            
            # Get the actual optimizer that has param_groups
            if with_dp and use_lookahead:
                actual_optimizer = optimizer.optimizer  # DP optimizer
            elif with_dp:
                actual_optimizer = optimizer  # DP optimizer
            elif use_lookahead:
                actual_optimizer = optimizer.optimizer  # base AdamW
            else:
                actual_optimizer = optimizer  # base AdamW
            
            # Update weight decay if scheduled
            if weight_decay_schedule:
                current_weight_decay = initial_weight_decay + (final_weight_decay - initial_weight_decay) * progress_ratio
                for param_group in actual_optimizer.param_groups:
                    param_group['weight_decay'] = current_weight_decay
            
            # Update adaptive betas if enabled
            if adaptive_betas:
                new_beta1 = beta1_schedule[0] + (beta1_schedule[1] - beta1_schedule[0]) * progress_ratio
                new_beta2 = beta2_schedule[0] + (beta2_schedule[1] - beta2_schedule[0]) * progress_ratio
                for param_group in actual_optimizer.param_groups:
                    param_group['betas'] = (new_beta1, new_beta2)

            stop_accumulating_grads = False
            accumulated_steps = 0
            if not with_dp:
                optimizer.zero_grad(set_to_none=True)
            while not stop_accumulating_grads:
                # fetch next training (micro)batch
                try:
                    step_data = next(trn_data_iter)
                except StopIteration:
                    trn_data_iter = iter(trn_dataloader)
                    step_data = next(trn_data_iter)
                # forward pass + calculate sample losses
                step_losses = _calculate_sample_losses(
                    argn, step_data, enhanced_criterion, use_mixed_precision, mixed_precision_backend
                )
                # FIXME in sequential case, this is an approximation, it should be divided by total sum of masks in the
                #  entire batch to get the average loss per sample. Less importantly the final sample may be smaller
                #  than the batch size in both flat and sequential case.
                # calculate total step loss
                step_loss = torch.mean(step_losses) / (1 if with_dp else gradient_accumulation_steps)
                if with_dp:
                    # opacus handles the gradient accumulation internally
                    optimizer.zero_grad(set_to_none=True)
                    
                # backward pass with mixed precision support
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
                    
                    if use_mixed_precision:
                        if mixed_precision_backend == "apex":
                            with amp.scale_loss(step_loss, optimizer.optimizer if use_lookahead else optimizer) as scaled_loss:
                                scaled_loss.backward()
                        elif mixed_precision_backend == "torch_amp":
                            grad_scaler.scale(step_loss).backward()
                    else:
                        step_loss.backward()
                
                # Enhanced gradient clipping (even for non-DP training)
                if gradient_clip_norm is not None and not with_dp:
                    if use_mixed_precision and mixed_precision_backend == "torch_amp":
                        # Unscale gradients before clipping
                        grad_scaler.unscale_(optimizer.optimizer if use_lookahead else optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(argn.parameters(), gradient_clip_norm)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(argn.parameters(), gradient_clip_norm)
                    
                    gradient_norms.append(total_norm.item())
                    
                    # Update batch size manager with gradient norm
                    if batch_size_manager:
                        batch_size_manager.update_metrics(
                            gradient_norm=total_norm.item(), 
                            loss=step_loss.item(), 
                            step=steps
                        )
                    
                    # Log gradient norms periodically
                    if len(gradient_norms) % 100 == 0:
                        avg_norm = np.mean(gradient_norms[-100:])
                        _LOG.debug(f"Average gradient norm (last 100 steps): {avg_norm:.4f}")
                elif batch_size_manager:
                    # Still track for batch size adaptation even without clipping
                    # Compute gradient norm manually
                    total_norm = 0
                    for p in argn.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    batch_size_manager.update_metrics(
                        gradient_norm=total_norm, 
                        loss=step_loss.item(), 
                        step=steps
                    )
                
                accumulated_steps += 1
                # explicitly count the number of processed samples as the actual batch size can vary when DP is on
                samples += step_losses.shape[0]
                if with_dp:
                    # for DP training, the optimizer will do different operations during virtual steps and logical steps
                    # - virtual step: clip and accumulate gradients
                    # - logical step: clip and accumulate gradients, add noises to gradients and update parameters
                    if use_mixed_precision and mixed_precision_backend == "torch_amp":
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    # if step was not skipped, it was a logical step, and we can stop accumulating gradients
                    stop_accumulating_grads = not optimizer._is_last_step_skipped
                elif accumulated_steps % gradient_accumulation_steps == 0:
                    # update parameters with accumulated gradients
                    if use_mixed_precision and mixed_precision_backend == "torch_amp":
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    stop_accumulating_grads = True
                # detach losses from the graph
                step_losses = step_losses.detach()
                trn_sample_losses.extend(step_losses)

            base_optimizer = optimizer.optimizer if use_lookahead else optimizer
            # For DP training, the actual optimizer might be wrapped differently
            if with_dp and use_lookahead:
                # DP + Lookahead: optimizer.optimizer is the DP optimizer
                dp_optimizer = optimizer.optimizer
                current_lr = dp_optimizer.param_groups[0]["lr"]
            elif with_dp:
                # DP only: optimizer is the DP optimizer
                current_lr = optimizer.param_groups[0]["lr"]
            elif use_lookahead:
                # Lookahead only: optimizer.optimizer is the base AdamW
                current_lr = optimizer.optimizer.param_groups[0]["lr"]
            else:
                # Neither: optimizer is the base AdamW
                current_lr = optimizer.param_groups[0]["lr"]

            # Enhanced learning rate scheduling
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

            # Adaptive batch size management (note: actual batch size change would require dataloader recreation)
            if batch_size_manager:
                new_batch_size, adaptation_reason = batch_size_manager.adapt_batch_size(steps)
                if new_batch_size != current_batch_size:
                    _LOG.info(f"Batch size adaptation suggested: {current_batch_size} -> {new_batch_size} ({adaptation_reason})")
                    # Note: In a full implementation, we would recreate the dataloader here
                    # For now, we just log the recommendation
                    current_batch_size = new_batch_size

            # do validation
            do_validation = on_epoch_end = epoch.is_integer()
            if do_validation:
                # calculate val loss and trn loss
                val_loss = _calculate_val_loss(
                    model=argn, 
                    val_dataloader=val_dataloader, 
                    criterion=enhanced_criterion,
                    use_mixed_precision=use_mixed_precision,
                    mixed_precision_backend=mixed_precision_backend,
                )
                # handle scenario where model training ran into numeric instability
                if pd.isna(val_loss):
                    _LOG.warning("validation loss is not available - reset model weights to last checkpoint")
                    load_model_weights(
                        model=argn,
                        path=workspace.model_tabular_weights_path,
                        device=device,
                    )
                trn_loss = _calculate_average_trn_loss(trn_sample_losses)
                dp_total_epsilon = (
                    privacy_engine.get_epsilon(dp_total_delta) + dp_value_protection_epsilon if with_dp else None
                )
                has_exceeded_dp_max_epsilon = dp_total_epsilon > dp_max_epsilon if with_dp else False
                if not has_exceeded_dp_max_epsilon:
                    # Get the correct optimizer for checkpointing
                    if with_dp and use_lookahead:
                        checkpoint_optimizer = optimizer.optimizer  # DP optimizer
                    elif with_dp:
                        checkpoint_optimizer = optimizer  # DP optimizer
                    elif use_lookahead:
                        checkpoint_optimizer = optimizer.optimizer  # base AdamW
                    else:
                        checkpoint_optimizer = optimizer  # base AdamW
                        
                    # save model weights with the best validation loss (and that hasn't exceeded DP max epsilon)
                    is_checkpoint = model_checkpoint.save_checkpoint_if_best(
                        val_loss=val_loss,
                        model=argn,
                        optimizer=checkpoint_optimizer,
                        lr_scheduler=lr_scheduler,
                        dp_accountant=privacy_engine.accountant if with_dp else None,
                    )
                else:
                    _LOG.info("early stopping: current DP epsilon has exceeded max epsilon")
                    
                # Log enhanced metrics
                if gradient_norms:
                    avg_grad_norm = np.mean(gradient_norms[-100:]) if len(gradient_norms) >= 100 else np.mean(gradient_norms)
                    _LOG.info(f"Avg gradient norm: {avg_grad_norm:.4f}, Weight decay: {current_weight_decay:.6f}")
                    
                # gather message for progress with checkpoint info
                progress_message = ProgressMessage(
                    epoch=epoch,
                    is_checkpoint=is_checkpoint,
                    steps=steps,
                    samples=samples,
                    trn_loss=trn_loss,
                    val_loss=val_loss,
                    total_time=total_time_init + time.time() - start_trn_time,
                    learn_rate=current_lr,
                    dp_eps=dp_total_epsilon,
                    dp_delta=dp_total_delta,
                )
                # check for early stopping
                do_stop = early_stopper(val_loss=val_loss) or has_exceeded_dp_max_epsilon
                # scheduling for ReduceLROnPlateau
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(metrics=val_loss)

            # log progress, either by time or by steps, whatever is shorter
            elapsed_training_time = time.time() - start_trn_time
            estimated_time_for_max_epochs = (max_epochs * trn_steps) * (elapsed_training_time / steps)
            if max_training_time < estimated_time_for_max_epochs:
                # use seconds for measuring progress against max_training_time
                progress_total_count = max_training_time
                progress_processed = elapsed_training_time
            else:
                # use steps for measuring progress against max_epochs
                progress_total_count = max_epochs * trn_steps
                progress_processed = steps
            # send a progress message at least every X minutes
            last_msg_interval = 5 * 60
            last_msg_elapsed = time.time() - last_msg_time
            if progress_message is None and (last_msg_elapsed > last_msg_interval or steps == 1):
                # running mean loss of the most recent training samples
                running_trn_loss = _calculate_average_trn_loss(trn_sample_losses, n=val_steps * val_batch_size)
                dp_total_epsilon = (
                    privacy_engine.get_epsilon(dp_total_delta) + dp_value_protection_epsilon if with_dp else None
                )
                progress_message = ProgressMessage(
                    epoch=epoch,
                    is_checkpoint=is_checkpoint,
                    steps=steps,
                    samples=samples,
                    trn_loss=running_trn_loss,
                    val_loss=None,
                    total_time=total_time_init + time.time() - start_trn_time,
                    learn_rate=current_lr,
                    dp_eps=dp_total_epsilon,
                    dp_delta=dp_total_delta,
                )
            if progress_message:
                last_msg_time = time.time()
            # send progress update
            res = progress.update(
                completed=int(progress_processed),
                total=int(progress_total_count),
                message=progress_message,
            )
            if do_validation:
                upload_model_data_callback()
            progress_message = None
            if (res or {}).get("stopExecution", False):
                _LOG.info("received STOP EXECUTION signal")
                do_stop = True

            if on_epoch_end:
                trn_sample_losses = []

            # check for max_epochs
            if epoch > max_epochs:
                do_stop = True

            # check for max_training_time
            total_training_time = total_time_init + time.time() - start_trn_time
            if total_training_time > max_training_time:
                do_stop = True

        # no checkpoint is saved yet because the training stopped before the first epoch ended
        if not model_checkpoint.has_saved_once():
            _LOG.info("saving model weights, as none were saved so far")
            # Get the correct optimizer for checkpointing
            if with_dp and use_lookahead:
                checkpoint_optimizer = optimizer.optimizer  # DP optimizer
            elif with_dp:
                checkpoint_optimizer = optimizer  # DP optimizer
            elif use_lookahead:
                checkpoint_optimizer = optimizer.optimizer  # base AdamW
            else:
                checkpoint_optimizer = optimizer  # base AdamW
                
            model_checkpoint.save_checkpoint(
                model=argn,
                optimizer=checkpoint_optimizer,
                lr_scheduler=lr_scheduler,
                dp_accountant=privacy_engine.accountant if with_dp else None,
            )
            if total_training_time > max_training_time:
                _LOG.info("skip validation loss calculation due to time-capped early stopping")
                val_loss = None
            else:
                _LOG.info("calculate validation loss")
                val_loss = _calculate_val_loss(
                    model=argn, 
                    val_dataloader=val_dataloader, 
                    criterion=enhanced_criterion,
                    use_mixed_precision=use_mixed_precision,
                    mixed_precision_backend=mixed_precision_backend,
                )
            dp_total_epsilon = (
                privacy_engine.get_epsilon(dp_total_delta) + dp_value_protection_epsilon if with_dp else None
            )
            # send a final message to inform how far we've progressed
            trn_loss = _calculate_average_trn_loss(trn_sample_losses)
            progress_message = ProgressMessage(
                epoch=epoch,
                is_checkpoint=1,
                steps=steps,
                samples=samples,
                trn_loss=trn_loss,
                val_loss=val_loss,
                total_time=total_training_time,
                learn_rate=current_lr,
                dp_eps=dp_total_epsilon,
                dp_delta=dp_total_delta,
            )
            progress.update(completed=steps, total=steps, message=progress_message)
            # ensure everything gets uploaded
            upload_model_data_callback()
            
        # Log final enhancement statistics
        if gradient_norms:
            _LOG.info(f"Final gradient statistics: mean={np.mean(gradient_norms):.4f}, std={np.std(gradient_norms):.4f}")
        _LOG.info(f"Final weight decay: {current_weight_decay:.6f}")
        if adaptive_betas:
            base_optimizer = optimizer.optimizer if use_lookahead else optimizer
            final_betas = base_optimizer.param_groups[0]['betas']
            _LOG.info(f"Final betas: {final_betas}")
            
    _LOG.info(f"ENHANCED_TRAIN_TABULAR finished in {time.time() - t0:.2f}s")
