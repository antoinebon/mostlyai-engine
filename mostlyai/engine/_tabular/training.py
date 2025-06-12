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
from dataclasses import dataclass

from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, disable_progress_bar
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast, GradScaler

from opacus import PrivacyEngine, GradSampleModule
from opacus.accountants import PRVAccountant, RDPAccountant, GaussianAccountant
from opacus.utils.batch_memory_manager import wrap_data_loader

from mostlyai.engine._memory import get_available_ram_for_heuristics
from mostlyai.engine.domain import ModelStateStrategy, DifferentialPrivacyConfig
from mostlyai.engine._tabular.enhanced_argn import EnhancedFlatModel
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
    """Calculate the physical batch size based on available resources."""
    data_points = no_tgt_data_points + no_ctx_data_points
    min_batch_size = 8
    
    # Scale batch_size corresponding to available memory
    mem_scale = 2.0 if mem_available_gb >= 32 else 1.0 if mem_available_gb >= 8 else 0.5
    
    # Set max_batch_size corresponding to available memory, model params and data points
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
    
    # Ensure a minimum number of batches to avoid excessive padding
    min_batches = 64
    batch_size = 2 ** int(np.log2(no_of_records / min_batches)) if no_of_records > 0 else min_batch_size
    return int(np.clip(a=batch_size, a_min=min_batch_size, a_max=max_batch_size))


def _learn_rate_heuristic(batch_size: int) -> float:
    """Calculate initial learning rate based on batch size."""
    return np.round(0.001 * np.sqrt(batch_size / 32), 5)


########################
### EMA MODEL WRAPPER ###
########################


class EMAModel:
    """Exponential Moving Average model wrapper for better generalization."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self._model = model
        self._decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """Update EMA weights with current model parameters."""
        for name, param in self._model.named_parameters():
            if param.requires_grad and name in self._shadow:
                self._shadow[name] = (
                    self._decay * self._shadow[name] + (1 - self._decay) * param.data
                )
    
    def apply_shadow(self) -> None:
        """Apply EMA weights to model for evaluation."""
        for name, param in self._model.named_parameters():
            if param.requires_grad and name in self._shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])
    
    def restore(self) -> None:
        """Restore original weights from backup."""
        for name, param in self._model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()




######################
### ENHANCED LOSS ###
######################


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum() if self.reduction == "sum" else focal_loss


####################
### DATA LOADERS ###
####################


class SimpleBatchCollator:
    """Simplified batch collator with optional sequence slicing."""

    def __init__(
        self, 
        is_sequential: bool, 
        max_sequence_window: int | None, 
        device: torch.device,
    ):
        self._is_sequential = is_sequential
        self._max_sequence_window = max_sequence_window
        self._device = device

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        batch = pd.DataFrame(batch)
        if self._is_sequential and self._max_sequence_window:
            batch = self._slice_sequences(batch, self._max_sequence_window)
        return self._convert_to_tensors(batch)

    def _convert_to_tensors(self, batch: pd.DataFrame) -> dict[str, torch.Tensor]:
        tensors = {}
        for column in batch.columns:
            if column.startswith(TGT) and self._is_sequential:
                tensors[column] = torch.unsqueeze(
                    torch.tensor(
                        np.array(list(zip_longest(*batch[column], fillvalue=0))).T,
                        dtype=torch.int64,
                        device=self._device,
                    ),
                    dim=-1,
                )
            elif column.startswith(TGT) and not self._is_sequential:
                tensors[column] = torch.unsqueeze(
                    torch.tensor(batch[column].values, dtype=torch.int64, device=self._device),
                    dim=-1,
                )
            elif column.startswith(CTXFLT):
                tensors[column] = torch.unsqueeze(
                    torch.tensor(batch[column].values, dtype=torch.int64, device=self._device),
                    dim=-1,
                )
            elif column.startswith(CTXSEQ):
                tensors[column] = torch.unsqueeze(
                    torch.nested.as_nested_tensor(
                        [torch.tensor(row, dtype=torch.int64, device=self._device) for row in batch[column]],
                        dtype=torch.int64,
                        device=self._device,
                    ),
                    dim=-1,
                )
        return tensors

    def _slice_sequences(self, batch: pd.DataFrame, max_sequence_window: int) -> pd.DataFrame:
        """Simple sequence slicing with random sampling."""
        tgt_columns = [col for col in batch.columns if col.startswith(TGT)]
        seq_lens = batch[tgt_columns[0]].copy().str.len().values

        # Simple random sampling strategy
        sel_idxs = []
        for seq_len in seq_lens:
            if seq_len <= max_sequence_window:
                sel_idxs.append(np.arange(seq_len))
            else:
                start_idx = np.random.randint(0, seq_len - max_sequence_window + 1)
                sel_idxs.append(np.arange(start_idx, start_idx + max_sequence_window))

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


#########################
### ENHANCED EARLY STOPPING ###
#########################


@dataclass
class EnhancedEarlyStopperConfig:
    """Configuration for enhanced early stopping."""
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    monitor_train_loss: bool = True
    divergence_threshold: float = 2.0


class EnhancedEarlyStopper:
    """Enhanced early stopping with multiple criteria."""
    
    def __init__(self, config: EnhancedEarlyStopperConfig):
        self._config = config
        self._best_val_loss = float('inf')
        self._patience_counter = 0
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        
    def __call__(self, val_loss: float, train_loss: float | None = None) -> bool:
        """Check if training should stop early."""
        # Track losses
        self._val_losses.append(val_loss)
        if train_loss is not None:
            self._train_losses.append(train_loss)
        
        # Check for improvement
        if val_loss < self._best_val_loss - self._config.min_delta:
            self._best_val_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
        
        # Check patience
        if self._patience_counter >= self._config.patience:
            _LOG.info(f"Early stopping: no improvement for {self._config.patience} epochs")
            return True
        
        # Check for training loss divergence
        if (self._config.monitor_train_loss and 
            len(self._train_losses) >= 5 and 
            train_loss is not None):
            recent_train_avg = np.mean(self._train_losses[-5:])
            if recent_train_avg > self._config.divergence_threshold * self._best_val_loss:
                _LOG.info("Early stopping: training loss divergence detected")
                return True
        
        return False
    
    @property
    def best_val_loss(self) -> float:
        """Get the best validation loss observed."""
        return self._best_val_loss


#####################
### TRAINING LOOP ###
#####################


class TabularModelCheckpoint(ModelCheckpoint):
    """Enhanced model checkpoint with EMA support."""
    
    def __init__(self, workspace: Workspace, save_ema: bool = True):
        super().__init__(workspace)
        self._save_ema = save_ema
    
    def _save_model_weights(self, model: torch.nn.Module) -> None:
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
    device: torch.device,
    use_mixed_precision: bool = True,
    use_focal_loss: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Calculate per-sample losses with enhanced loss functions."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
        
        # Forward pass with mixed precision if enabled
        if use_mixed_precision:
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type):
                output, _ = model(data, mode="trn")
        else:
            output, _ = model(data, mode="trn")

    # Get target columns
    tgt_cols = (
        list(model.tgt_cardinalities.keys())
        if not isinstance(model, GradSampleModule)
        else model._module.tgt_cardinalities.keys()
    )
    
    # Choose loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")
    
    # Handle sequential vs flat models
    if isinstance(model, SequentialModel) or (
        isinstance(model, GradSampleModule) and isinstance(model._module, SequentialModel)
    ):
        slen_cols = [k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)]
        
        # Generate masks for SLEN and time step
        slen_mask = torch.zeros_like(data[slen_cols[0]], dtype=torch.int64)
        for slen_col in slen_cols:
            slen_mask |= data[slen_col] != 0
        slen_mask = slen_mask.squeeze(-1)
        time_step_mask = torch.zeros_like(slen_mask, dtype=torch.int64)
        time_step_mask[:, 0] = 10  # Emphasize first time step
        
        # Calculate per column losses
        sidx_cols = {k for k in data if k.startswith(SIDX_SUB_COLUMN_PREFIX)}
        sdec_cols = {k for k in data if k.startswith(SDEC_SUB_COLUMN_PREFIX)}
        losses_by_column = []
        
        for col in tgt_cols:
            if col in slen_cols:
                mask = time_step_mask
            elif col in sidx_cols or col in sdec_cols:
                mask = torch.zeros_like(slen_mask, dtype=torch.int64)
            else:
                mask = slen_mask

            column_loss = criterion(output[col].transpose(1, 2), data[col].squeeze(2))
            masked_loss = torch.sum(column_loss * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
            losses_by_column.append(masked_loss)
    else:
        losses_by_column = [criterion(output[col], data[col].squeeze(1)) for col in tgt_cols]
    
    # Sum up column level losses
    losses = torch.sum(torch.stack(losses_by_column, dim=0), dim=0)
    return losses


@torch.no_grad()
def _calculate_val_loss(
    model: FlatModel | SequentialModel,
    val_dataloader: DataLoader,
    device: torch.device,
    use_mixed_precision: bool = True,
    use_focal_loss: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> float:
    """Calculate validation loss with gradient tracking disabled."""
    val_sample_losses: list[torch.Tensor] = []
    model.eval()
    
    for step_data in val_dataloader:
        step_losses = _calculate_sample_losses(
            model, step_data, device, use_mixed_precision, use_focal_loss, focal_alpha, focal_gamma, label_smoothing
        )
        val_sample_losses.extend(step_losses.detach())
    
    model.train()
    val_sample_losses_tensor = torch.stack(val_sample_losses, dim=0)
    return torch.mean(val_sample_losses_tensor).item()


def _calculate_average_trn_loss(trn_sample_losses: list[torch.Tensor], n: int | None = None) -> float | None:
    """Calculate average training loss from recent samples."""
    if len(trn_sample_losses) == 0:
        return None
    
    trn_losses_latest = torch.stack(trn_sample_losses, dim=0)
    if n is not None:
        trn_losses_latest = trn_losses_latest[-n:]
    
    return torch.mean(trn_losses_latest).item()


################
### TRAINING ###
################


def train(
    *,
    model: str = "MOSTLY_AI/Medium",
    max_training_time: float = 14400.0,
    max_epochs: float = 100.0,
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
    # Simplified enhancement parameters
    use_mixed_precision: bool = True,
    gradient_clip_norm: float = 1.0,
    learning_rate: float | None = None,
    scheduler_type: str = "warmup_cosine",  # "warmup_cosine", "one_cycle", "plateau"
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.01,
    weight_decay: float = 0.01,
    weight_decay_end: float | None = None,
    use_focal_loss: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    **kwargs,
):
    """
    Enhanced training function with simplified but effective optimizations.
    
    Args:
        # Core parameters
        model: Model size specification
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
        
        # Enhancement parameters
        use_mixed_precision: Enable PyTorch native automatic mixed precision
        gradient_clip_norm: Gradient clipping norm (1.0 recommended)
        learning_rate: Initial learning rate (auto-determined if None)
        scheduler_type: LR scheduler type ("warmup_cosine", "one_cycle", "plateau")
        warmup_epochs: Number of warmup epochs
        min_lr_ratio: Minimum LR as ratio of initial LR
        weight_decay: Weight decay coefficient
        weight_decay_end: Final weight decay (for scheduling)
        use_focal_loss: Use focal loss for class imbalance
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor (0.1 recommended)
        use_ema: Use exponential moving average of weights
        ema_decay: EMA decay factor
        early_stopping_patience: Early stopping patience
        early_stopping_min_delta: Minimum improvement threshold
    """
    _LOG.info("ENHANCED_TRAIN_TABULAR started with simplified optimizations")
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
        _LOG.info(f"Mixed Precision: {use_mixed_precision}")
        _LOG.info(f"Gradient Clipping: {gradient_clip_norm}")
        _LOG.info(f"Scheduler: {scheduler_type}")
        _LOG.info(f"Label Smoothing: {label_smoothing}")
        _LOG.info(f"EMA: {use_ema} (decay={ema_decay})")
        _LOG.info(f"Focal Loss: {use_focal_loss}")

        # Data preparation (same as original)
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

        # Set defaults
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
        
        with_dp = differential_privacy is not None
        _LOG.info(f"{with_dp=}")
        _LOG.info(f"{model_state_strategy=}")

        # Initialize callbacks
        upload_model_data_callback = upload_model_data_callback or (lambda *args, **kwargs: None)

        # Early exit check
        if check_early_training_exit(workspace=workspace, trn_cnt=trn_cnt, val_cnt=val_cnt):
            _LOG.warning("not enough data to train model; skipping training")
            return

        # Determine column order
        if enable_flexible_generation:
            trn_column_order = None
        else:
            tgt_cardinalities = get_cardinalities(tgt_stats)
            trn_column_order = get_columns_from_cardinalities(tgt_cardinalities)

        # Sequence length stats
        tgt_seq_len_stats = get_sequence_length_stats(tgt_stats)
        tgt_seq_len_median = tgt_seq_len_stats["median"]
        tgt_seq_len_max = tgt_seq_len_stats["max"]
        max_sequence_window = np.clip(max_sequence_window, a_min=1, a_max=tgt_seq_len_max)
        _LOG.info(f"{max_sequence_window=}")
        ctx_seq_len_median = get_ctx_sequence_length(ctx_stats, key="median")

        torch.set_flush_denormal(True)

        _LOG.info("create training model")
        model_checkpoint = TabularModelCheckpoint(workspace=workspace, save_ema=use_ema)
        
        # Create model
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
                with_dp=with_dp,
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
            # argn = EnhancedFlatModel(
            #     tgt_cardinalities=tgt_cardinalities,
            #     ctx_cardinalities=ctx_cardinalities,
            #     ctxseq_len_median=ctx_seq_len_median,
            #     model_size=model_size,
            #     column_order=trn_column_order,
            #     device=device,
            # )
        _LOG.info(f"model class: {argn.__class__.__name__}")

        # Handle model state strategy
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
        else:
            _LOG.info("remove existing checkpoint files")
            model_checkpoint.clear_checkpoint()

        # Handle progress state
        last_progress_message = progress.get_last_progress_message()
        if last_progress_message and model_state_strategy == ModelStateStrategy.resume:
            epoch = last_progress_message.get("epoch", 0.0)
            steps = last_progress_message.get("steps", 0)
            samples = last_progress_message.get("samples", 0)
            current_lr = last_progress_message.get("learn_rate", None)
            total_time_init = last_progress_message.get("total_time", 0.0)
        else:
            epoch = 0.0
            steps = 0
            samples = 0
            current_lr = None
            total_time_init = 0.0
            progress.reset_progress_messages()
        _LOG.info(f"start training progress from {epoch=}, {steps=}")

        argn.to(device)
        no_of_model_params = get_no_of_model_parameters(argn)
        _LOG.info(f"{no_of_model_params=}")

        # Persist model configs
        model_units = get_model_units(argn)
        model_configs = {
            "model_id": model,
            "model_units": model_units,
            "enable_flexible_generation": enable_flexible_generation,
        }
        workspace.model_configs.write(model_configs)

        # Calculate batch size and learning rate
        mem_available_gb = get_available_ram_for_heuristics() / 1024**3
        no_tgt_data_points = get_max_data_points_per_sample(tgt_stats)
        no_ctx_data_points = get_max_data_points_per_sample(ctx_stats)
        
        if batch_size is None:
            batch_size = _physical_batch_size_heuristic(
                mem_available_gb=mem_available_gb,
                no_of_records=trn_cnt,
                no_tgt_data_points=no_tgt_data_points,
                no_ctx_data_points=no_ctx_data_points,
                no_of_model_params=no_of_model_params,
            )
        
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = 1

        # Setup batch parameters
        batch_size = max(1, min(batch_size, trn_cnt))
        gradient_accumulation_steps = max(1, min(gradient_accumulation_steps, trn_cnt // batch_size))
        trn_batch_size = batch_size * gradient_accumulation_steps
        trn_steps = max(1, trn_cnt // trn_batch_size)
        val_batch_size = max(1, min(batch_size, val_cnt))
        
        if learning_rate is None:
            learning_rate = current_lr or _learn_rate_heuristic(trn_batch_size)
        
        if is_sequential:
            val_batch_size = val_batch_size // 2

        # Setup data loaders with simplified collator
        batch_collator = SimpleBatchCollator(
            is_sequential=is_sequential, 
            max_sequence_window=max_sequence_window, 
            device=device,
        )
        
        disable_progress_bar()
        trn_dataset = load_dataset("parquet", data_files=[str(p) for p in workspace.encoded_data_trn.fetch_all()])["train"]
        trn_dataloader = DataLoader(
            dataset=trn_dataset,
            shuffle=True,
            batch_size=trn_batch_size if with_dp else batch_size,
            collate_fn=batch_collator,
        )
        
        val_dataset = load_dataset("parquet", data_files=[str(p) for p in workspace.encoded_data_val.fetch_all()])["train"]
        val_dataloader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=val_batch_size,
            collate_fn=batch_collator,
        )

        _LOG.info(f"{trn_cnt=}, {val_cnt=}")
        _LOG.info(f"{len(tgt_sub_columns)=}, {len(ctxflt_sub_columns)=}, {len(ctxseq_sub_columns)=}")
        _LOG.info(f"{trn_batch_size=}, {val_batch_size=}")
        _LOG.info(f"{batch_size=}, {gradient_accumulation_steps=}, {learning_rate=}")

        # Setup EMA if enabled
        ema_model = EMAModel(argn, decay=ema_decay) if use_ema else None
        if ema_model:
            _LOG.info(f"EMA enabled with decay={ema_decay}")

        # Enhanced early stopping
        early_stopper_config = EnhancedEarlyStopperConfig(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            restore_best_weights=True,
            monitor_train_loss=True,
        )
        early_stopper = EnhancedEarlyStopper(early_stopper_config)

        # Setup optimizer with weight decay scheduling
        if weight_decay_end is None:
            weight_decay_end = weight_decay * 0.1  # Default: reduce to 10% at end
            
        optimizer = torch.optim.AdamW(
            params=argn.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8,
        )

        # Setup learning rate scheduler
        total_steps = int(max_epochs * trn_steps)
        warmup_steps = int(warmup_epochs * trn_steps)
        
        if scheduler_type == "one_cycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=learning_rate,
                steps_per_epoch=trn_steps,
                epochs=int(max_epochs),
            )
        elif scheduler_type == "warmup_cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_steps,
            )
        else:  # plateau
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=0.5,
                patience=4,
                min_lr=1e-8,
            )
        
        _LOG.info(f"Learning rate scheduler: {scheduler_type}")

        # Setup mixed precision
        grad_scaler = GradScaler() if use_mixed_precision else None
        if use_mixed_precision:
            _LOG.info("Mixed precision training enabled")

        # Restore optimizer and scheduler states if resuming
        if (model_state_strategy == ModelStateStrategy.resume and 
            model_checkpoint.optimizer_and_lr_scheduler_paths_exist()):
            _LOG.info("restore optimizer and LR scheduler states")
            optimizer.load_state_dict(
                torch.load(workspace.model_optimizer_path, map_location=device, weights_only=True)
            )
            lr_scheduler.load_state_dict(
                torch.load(workspace.model_lr_scheduler_path, map_location=device, weights_only=True)
            )

        # GPU optimizations
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Setup differential privacy
        if with_dp:
            if isinstance(differential_privacy, DifferentialPrivacyConfig):
                dp_config = differential_privacy.model_dump()
            else:
                dp_config = DifferentialPrivacyConfig(**differential_privacy).model_dump()
            dp_max_epsilon = dp_config.get("max_epsilon") or float("inf")
            dp_total_delta = dp_config.get("delta", 1e-5)
            dp_value_protection_epsilon = (ctx_stats.get("value_protection_epsilon_spent") or 0.0) + (
                tgt_stats.get("value_protection_epsilon_spent") or 0.0
            )
            dp_accountant = "rdp"
            _LOG.info(f"{dp_config=}, {dp_accountant=}")
            
            privacy_engine = PrivacyEngine(accountant=dp_accountant)
            if model_state_strategy == ModelStateStrategy.resume and workspace.model_dp_accountant_path.exists():
                _LOG.info("restore DP accountant state")
                torch.serialization.add_safe_globals([getattr, PRVAccountant, RDPAccountant, GaussianAccountant])
                privacy_engine.accountant.load_state_dict(
                    torch.load(workspace.model_dp_accountant_path, map_location=device, weights_only=True)
                )
            
            argn, optimizer, trn_dataloader = privacy_engine.make_private(
                module=argn,
                optimizer=optimizer,
                data_loader=trn_dataloader,
                noise_multiplier=dp_config.get("noise_multiplier"),
                max_grad_norm=dp_config.get("max_grad_norm"),
                poisson_sampling=True,
            )
            
            trn_dataloader = wrap_data_loader(
                data_loader=trn_dataloader, max_batch_size=batch_size, optimizer=optimizer
            )
        else:
            privacy_engine = None
            dp_config, dp_total_delta, dp_accountant = None, None, None

        # Training loop
        progress_message = None
        start_trn_time = time.time()
        last_msg_time = time.time()
        trn_data_iter = iter(trn_dataloader)
        trn_sample_losses: list[torch.Tensor] = []
        do_stop = False
        
        # Gradient norm tracking
        gradient_norms: list[float] = []
        
        _LOG.info("Starting enhanced training loop")
        
        while not do_stop:
            is_checkpoint = 0
            steps += 1
            epoch = steps / trn_steps

            # Update weight decay if scheduling
            if weight_decay_end != weight_decay:
                progress_ratio = min(1.0, steps / total_steps) if total_steps > 0 else 0.0
                current_weight_decay = weight_decay + (weight_decay_end - weight_decay) * progress_ratio
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = current_weight_decay

            # Gradient accumulation loop
            stop_accumulating_grads = False
            accumulated_steps = 0
            
            if not with_dp:
                optimizer.zero_grad(set_to_none=True)
                
            while not stop_accumulating_grads:
                # Fetch next training batch
                try:
                    step_data = next(trn_data_iter)
                except StopIteration:
                    trn_data_iter = iter(trn_dataloader)
                    step_data = next(trn_data_iter)
                
                # Forward pass with mixed precision
                step_losses = _calculate_sample_losses(
                    argn, step_data, device, use_mixed_precision, use_focal_loss, focal_alpha, focal_gamma, label_smoothing
                )
                
                step_loss = torch.mean(step_losses) / (1 if with_dp else gradient_accumulation_steps)
                
                if with_dp:
                    optimizer.zero_grad(set_to_none=True)
                
                # Backward pass with mixed precision
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
                    
                    if use_mixed_precision:
                        grad_scaler.scale(step_loss).backward()
                    else:
                        step_loss.backward()
                
                # Gradient clipping and tracking
                if gradient_clip_norm is not None and not with_dp:
                    if use_mixed_precision:
                        grad_scaler.unscale_(optimizer)
                    
                    total_norm = torch.nn.utils.clip_grad_norm_(argn.parameters(), gradient_clip_norm)
                    gradient_norms.append(total_norm.item())
                
                accumulated_steps += 1
                samples += step_losses.shape[0]
                
                # Parameter update
                if with_dp:
                    if use_mixed_precision:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    stop_accumulating_grads = not optimizer._is_last_step_skipped
                elif accumulated_steps % gradient_accumulation_steps == 0:
                    if use_mixed_precision:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    stop_accumulating_grads = True
                
                # Update EMA
                if ema_model and stop_accumulating_grads:
                    ema_model.update()
                
                step_losses = step_losses.detach()
                trn_sample_losses.extend(step_losses)

            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler_type != "plateau":
                lr_scheduler.step()

            # Validation
            do_validation = epoch.is_integer()
            if do_validation:
                # Use EMA weights for validation if enabled
                if ema_model:
                    ema_model.apply_shadow()
                
                val_loss = _calculate_val_loss(
                    model=argn,
                    val_dataloader=val_dataloader,
                    device=device,
                    use_mixed_precision=use_mixed_precision,
                    use_focal_loss=use_focal_loss,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    label_smoothing=label_smoothing,
                )
                
                # Restore original weights
                if ema_model:
                    ema_model.restore()
                
                # Handle numerical instability
                if pd.isna(val_loss):
                    _LOG.warning("validation loss is not available - reset model weights to last checkpoint")
                    load_model_weights(model=argn, path=workspace.model_tabular_weights_path, device=device)
                
                trn_loss = _calculate_average_trn_loss(trn_sample_losses)
                dp_total_epsilon = (
                    privacy_engine.get_epsilon(dp_total_delta) + dp_value_protection_epsilon if with_dp else None
                )
                has_exceeded_dp_max_epsilon = dp_total_epsilon > dp_max_epsilon if with_dp else False
                
                if not has_exceeded_dp_max_epsilon:
                    # Apply EMA weights for checkpoint saving
                    if ema_model:
                        ema_model.apply_shadow()
                    
                    is_checkpoint = model_checkpoint.save_checkpoint_if_best(
                        val_loss=val_loss,
                        model=argn,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        dp_accountant=privacy_engine.accountant if with_dp else None,
                    )
                    
                    # Restore original weights
                    if ema_model:
                        ema_model.restore()
                else:
                    _LOG.info("early stopping: current DP epsilon has exceeded max epsilon")
                
                # Enhanced early stopping
                do_stop = early_stopper(val_loss=val_loss, train_loss=trn_loss) or has_exceeded_dp_max_epsilon
                
                # Plateau scheduler step
                if scheduler_type == "plateau":
                    lr_scheduler.step(metrics=val_loss)
                
                # Log gradient statistics
                if gradient_norms:
                    recent_grad_norm = np.mean(gradient_norms[-100:]) if len(gradient_norms) >= 100 else np.mean(gradient_norms)
                    _LOG.info(f"Avg gradient norm: {recent_grad_norm:.4f}")
                
                # Progress message
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

            # Progress updates
            elapsed_training_time = time.time() - start_trn_time
            estimated_time_for_max_epochs = (max_epochs * trn_steps) * (elapsed_training_time / steps)
            
            if max_training_time < estimated_time_for_max_epochs:
                progress_total_count = max_training_time
                progress_processed = elapsed_training_time
            else:
                progress_total_count = max_epochs * trn_steps
                progress_processed = steps
            
            # Send progress message periodically
            last_msg_interval = 5 * 60
            last_msg_elapsed = time.time() - last_msg_time
            
            if progress_message is None and (last_msg_elapsed > last_msg_interval or steps == 1):
                running_trn_loss = _calculate_average_trn_loss(trn_sample_losses, n=val_batch_size * 5)
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
            
            # Send progress update
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

            if epoch.is_integer():
                trn_sample_losses = []

            # Check stopping conditions
            if epoch > max_epochs:
                do_stop = True

            total_training_time = total_time_init + time.time() - start_trn_time
            if total_training_time > max_training_time:
                do_stop = True

        # Final checkpoint if none saved
        if not model_checkpoint.has_saved_once():
            _LOG.info("saving model weights, as none were saved so far")
            
            # Apply EMA weights for final save
            if ema_model:
                ema_model.apply_shadow()
            
            model_checkpoint.save_checkpoint(
                model=argn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                dp_accountant=privacy_engine.accountant if with_dp else None,
            )
            
            # Calculate final validation loss
            if total_training_time > max_training_time:
                _LOG.info("skip validation loss calculation due to time-capped early stopping")
                val_loss = None
            else:
                _LOG.info("calculate validation loss")
                val_loss = _calculate_val_loss(
                    model=argn,
                    val_dataloader=val_dataloader,
                    device=device,
                    use_focal_loss=use_focal_loss,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    label_smoothing=label_smoothing,
                )
            
            # Restore original weights if using EMA
            if ema_model:
                ema_model.restore()
            
            dp_total_epsilon = (
                privacy_engine.get_epsilon(dp_total_delta) + dp_value_protection_epsilon if with_dp else None
            )
            
            # Final progress message
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
            upload_model_data_callback()

        # Log final statistics
        if gradient_norms:
            _LOG.info(f"Final gradient statistics: mean={np.mean(gradient_norms):.4f}, std={np.std(gradient_norms):.4f}")
        if early_stopper.best_val_loss != float('inf'):
            _LOG.info(f"Best validation loss: {early_stopper.best_val_loss:.6f}")
        
    _LOG.info(f"ENHANCED_TRAIN_TABULAR finished in {time.time() - t0:.2f}s")


__all__ = [
    "train",
    "EMAModel", 
    "FocalLoss",
    "SimpleBatchCollator",
    "EnhancedEarlyStopper",
    "EnhancedEarlyStopperConfig",
    "TabularModelCheckpoint",
]
