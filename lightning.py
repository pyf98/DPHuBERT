import math
import pathlib
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from wav2vec2.model import (
    Wav2Vec2Model,
)
from dataset.audio_dataset import (
    BucketizeBatchSampler,
    DistributedBatchSampler,
    CollateFnAudio,
    AudioDataset,
)


class LinearDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        max_updates: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [self._step_count / self.warmup_updates * base_lr for base_lr in self.base_lrs]
        elif self._step_count >= self.max_updates:
            return [0.0 for _ in self.base_lrs]
        else:
            pct_remaining = (self.max_updates - self._step_count) / (self.max_updates - self.warmup_updates)
            return [base_lr * pct_remaining for base_lr in self.base_lrs]


class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]


class DistillLoss(nn.Module):
    def __init__(self, l2_weight, l1_weight, cos_weight, cos_type):
        super().__init__()
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.cos_weight = cos_weight
        self.cos_type = cos_type
        assert cos_type in ["raw", "log_sig"], cos_type

        if l2_weight != 0:
            self.mse_loss = nn.MSELoss()
        if l1_weight != 0:
            self.l1_loss = nn.L1Loss()
        if cos_weight != 0:
            self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def __repr__(self) -> str:
        return "{}(l2={}, l1={}, {}_cos={})".format(
            self.__class__.__name__,
            self.l2_weight,
            self.l1_weight,
            self.cos_type,
            self.cos_weight,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (batch, layer, time, feature)
            target: same shape as input
        """
        loss_mse = 0
        loss_l1 = 0
        loss_cos = 0
        if self.l2_weight != 0:
            loss_mse = self.mse_loss(input, target)
        if self.l1_weight != 0:
            loss_l1 = self.l1_loss(input, target)
        if self.cos_weight != 0:    # maximize cosine similarity
            if self.cos_type == "raw":
                loss_cos = -self.cos_sim(input, target).mean()
            elif self.cos_type == "log_sig":
                loss_cos = -self.cos_sim(input, target).sigmoid().log().mean()
            else:
                raise ValueError

        loss = self.l2_weight * loss_mse + self.l1_weight * loss_l1 + self.cos_weight * loss_cos

        return loss, (loss_mse, loss_l1, loss_cos)


class DistillModule(pl.LightningModule):
    def __init__(
        self,
        *,
        teacher_model: Wav2Vec2Model,
        student_model: Wav2Vec2Model,
        distill_mode: str,  # "layer2layer", "predlayer"
        distill_layers: List[int],  # layer indices to align, from 0 to num_layers
        distill_linear_projs: nn.ModuleList, # list of linear layers which transform student to teacher
        distill_loss: DistillLoss,
        learning_rate: float,
        weight_decay: float,
        warmup_updates: int,
        max_updates: int,
        use_reg: bool,  # whether to use the L0 regularization
        reg_learning_rate: Optional[float],   # lr for loga and lambda
        target_sparsity: Optional[float],
        sparsity_warmup_updates: Optional[int],   # linearly increase the target sparsity
        tsv_dir: Union[str, pathlib.Path],
        train_subset: str,
        seconds_per_batch: float,
        num_workers: int,
    ):
        super().__init__()

        self.teacher_model = teacher_model
        self.student_model = student_model

        self.original_num_params = sum(p.numel() for p in teacher_model.parameters())

        assert distill_mode in ["layer2layer", "predlayer"], distill_mode
        assert len(distill_layers) == len(distill_linear_projs)
        self.distill_mode = distill_mode
        self.distill_layers = distill_layers
        self.distill_linear_projs = distill_linear_projs
        self.distill_loss = distill_loss

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates

        self.use_reg = use_reg
        self.reg_learning_rate = reg_learning_rate
        self.target_sparsity = target_sparsity
        self.sparsity_warmup_updates = sparsity_warmup_updates

        # lambdas for Lagrangian
        if self.use_reg:
            self.lambda1 = nn.Parameter(torch.tensor(0.0))
            self.lambda2 = nn.Parameter(torch.tensor(0.0))

        # dataset related
        self.tsv_dir = tsv_dir
        self.train_subset = train_subset
        self.seconds_per_batch = seconds_per_batch
        self.num_workers = num_workers

    def configure_optimizers(self):
        main_params = [p for n, p in self.student_model.named_parameters() if "log_alpha" not in n]
        main_params.extend(list(self.distill_linear_projs.parameters()))
        pgs = [
            {
                'params': main_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'name': 'main_params',
            },
        ]
        if self.use_reg:
            pgs.extend(
                [
                    {
                        'params': [p for n, p in self.student_model.named_parameters() if "log_alpha" in n],
                        'lr': self.reg_learning_rate,
                        'weight_decay': 0.0,
                        'name': 'log_alpha',
                    },
                    {
                        'params': [self.lambda1, self.lambda2],
                        'lr': -self.reg_learning_rate,
                        'weight_decay': 0.0,
                        'name': 'lambda',
                    },
                ]
            )
        optimizer = torch.optim.AdamW(pgs)
        lr_scheduler = LinearDecayLRScheduler(
            optimizer, warmup_updates=self.warmup_updates, max_updates=self.max_updates
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def _get_target_sparsity(self):
        if self.global_step >= self.sparsity_warmup_updates:
            return self.target_sparsity
        return self.target_sparsity * (self.global_step / self.sparsity_warmup_updates)

    def _step(self, batch, batch_idx, mode):
        waveforms, lengths = batch
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_hiddens, teacher_lengths = self.teacher_model.extract_features(waveforms, lengths)
            teacher_hiddens = torch.stack(
                [teacher_hiddens[idx] for idx in self.distill_layers], dim=1
            )   # (batch, layer, time, feature)
        
        student_hiddens, student_lengths = self.student_model.extract_features(waveforms, lengths)
        new_student_hiddens = []
        for idx, proj in zip(self.distill_layers, self.distill_linear_projs):
            if self.distill_mode == "layer2layer":
                new_student_hiddens.append(proj(student_hiddens[idx]))
            elif self.distill_mode == "predlayer":
                new_student_hiddens.append(proj(student_hiddens[-1]))
            else:
                raise ValueError(f"Invalid distill mode: {self.distill_mode}")
        student_hiddens = torch.stack(new_student_hiddens, dim=1)   # (batch, layer, time, feature)

        loss_distill, (loss_mse, loss_l1, loss_cos) = self.distill_loss(student_hiddens, teacher_hiddens)

        if self.use_reg:
            cur_target_sparsity = self._get_target_sparsity()
            cur_expected_sparsity = 1. - self.student_model.get_num_params() / self.original_num_params
            loss_reg = self.lambda1 * (cur_expected_sparsity - cur_target_sparsity) \
                + self.lambda2 * (cur_expected_sparsity - cur_target_sparsity)**2
        else:
            loss_reg = 0

        loss = loss_distill + loss_reg

        self.log_dict(
            {
                f"{mode}_loss": loss,   # total loss
                f"{mode}_loss_distill": loss_distill,   # distill total loss
                f"{mode}_loss_mse": loss_mse,
                f"{mode}_loss_l1": loss_l1,
                f"{mode}_loss_cos": loss_cos,
                f"{mode}_loss_reg": loss_reg,   # sparsity loss
            }
        )
        if mode == "train" and self.use_reg:
            self.log_dict(
                {
                    'sparsity_expected': cur_expected_sparsity,
                    'sparsity_target': cur_target_sparsity,
                    'lambda1': self.lambda1,
                    'lambda2': self.lambda2,
                },
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode="valid")
        return loss

    def train_dataloader(self):
        dataset = AudioDataset(self.tsv_dir, self.train_subset)
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=False,
        )
        sampler = DistributedBatchSampler(sampler, shuffle=True)
        sampler.set_epoch(self.current_epoch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnAudio(pad=False, rand_crop=True),   # crop to the min length in a mini-batch
            num_workers=self.num_workers,
        )
        return dataloader        

    def val_dataloader(self):
        dataset = AudioDataset(self.tsv_dir, "valid")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnAudio(pad=False, rand_crop=True),
            num_workers=self.num_workers,
        )
        return dataloader
