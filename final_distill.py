"""Perform distillation for the pruned model."""

import logging
import pathlib
from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_lite.utilities.rank_zero import _get_rank

from lightning import (
    DistillModule,
    DistillLoss,
)
from wav2vec2.model import (
    wav2vec2_model,
)

_LG = logging.getLogger(f"{__name__}:{_get_rank()}")


def run_train(args):
    pl.seed_everything(2022)

    # Callbacks
    lr_monitor = LearningRateMonitor()  # log learning rates for all param groups
    model_checkpoint = ModelCheckpoint(dirpath=args.exp_dir / "ckpts", verbose=True)   # only save the latest epoch
    callbacks = [lr_monitor, model_checkpoint]

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        callbacks=callbacks,
        max_steps=args.max_updates,
        strategy="ddp",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accumulate_grad_batches=args.accum_grad,
        replace_sampler_ddp=False,  # we use the custom distributed sampler for ddp
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=args.clip_norm,
        log_every_n_steps=args.log_interval,
        precision=args.precision,
    )

    # Create teacher model
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location='cpu')
    teacher_model = wav2vec2_model(**teacher_ckpt['config'])
    _LG.info(f"Teacher model:\n{teacher_model}")
    teacher_result = teacher_model.load_state_dict(teacher_ckpt['state_dict'], strict=False)
    _LG.info(f"Load pretrained ckpt to teacher: missing {teacher_result.missing_keys}, unexpected {teacher_result.unexpected_keys}")
    # Freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False
    _LG.info("Freeze parameters of the teacher model by setting requires_grad=False")
    teacher_model.eval()
    
    # Create student model
    student_ckpt = torch.load(args.student_ckpt, map_location='cpu')
    student_model = wav2vec2_model(**student_ckpt["config"])
    _LG.info(f"Student model:\n{student_model}")
    student_result = student_model.load_state_dict(student_ckpt["state_dict"], strict=False)
    _LG.info(f"Load pretrained ckpt to student: missing {student_result.missing_keys}, unexpected {student_result.unexpected_keys}")

    # Load weights to create linear layers which transform student hiddens to teacher hiddens
    distill_layer_groups = [[int(l) for l in g.split(",")] for g in args.distill_layers.split(".")]
    _LG.info(f"Distill transformer layers: {distill_layer_groups}")
    distill_layers = []
    for g in distill_layer_groups:
        distill_layers.extend(g)
    student_embed_dim = student_model.encoder.feature_projection.projection.out_features
    teacher_embed_dim = teacher_model.encoder.feature_projection.projection.out_features

    if args.distill_mode == "layer2layer":
        distill_linear_projs = nn.ModuleList()
        for g in distill_layer_groups:      # layers in the same group share a linear layer
            tmp_linear = nn.Linear(student_embed_dim, teacher_embed_dim)
            for _ in range(len(g)):
                distill_linear_projs.append(tmp_linear)
    elif args.distill_mode == "predlayer":      # same as DistilHuBERT
        # use independent linear layers, cannot be shared
        distill_linear_projs = nn.ModuleList(
            nn.Sequential(
                nn.Linear(student_embed_dim, teacher_embed_dim),
                nn.GELU(),
            ) for _ in range(len(distill_layers))
        )
    else:
        raise ValueError(f"Invalid distill mode: {args.distill_mode}")
    
    distill_linear_projs.load_state_dict(student_ckpt["distill_linear_projs"])

    # Create DistillLoss module
    distill_loss_criterion = DistillLoss(
        l2_weight=args.l2_weight,
        l1_weight=args.l1_weight,
        cos_weight=args.cos_weight,
        cos_type=args.cos_type,
    )
    _LG.info(f"Distill loss module:\n{distill_loss_criterion}")

    distill_module = DistillModule(
        teacher_model=teacher_model,
        student_model=student_model,
        distill_mode=args.distill_mode,
        distill_layers=distill_layers,
        distill_linear_projs=distill_linear_projs,
        distill_loss=distill_loss_criterion,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
        use_reg=False,      # no pruning, only distillation
        reg_learning_rate=None,
        target_sparsity=None,
        sparsity_warmup_updates=None,
        tsv_dir=args.tsv_dir,
        train_subset=args.train_subset,
        seconds_per_batch=args.seconds_per_batch,
        num_workers=args.num_workers,
    )

    trainer.fit(
        distill_module, 
        ckpt_path=args.resume_checkpoint,
    )


def _parse_args():
    parser = ArgumentParser(
        description="Distill the pruned model.",
    )

    # dataset and dataloader related
    parser.add_argument(
        "--tsv_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the directory containing tsv files.",
    )
    parser.add_argument(
        "--train_subset",
        default="train100",
        choices=["train100", "train960"],
        type=str,
        help="The subset name for training. (Default: 'train100')",
    )
    parser.add_argument(
        "--seconds_per_batch",
        default=87.5,
        type=float,
        help="Number of seconds of audio in a mini-batch. (Default: 87.5)",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers in DataLoader."
    )

    # general training related
    parser.add_argument(
        "--resume_checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to the feature and label directories. (Default: None)",
    )
    parser.add_argument(
        "--exp_dir",
        type=pathlib.Path,
        help="Suffix of the exp directory name."
    )
    parser.add_argument(
        "--log_interval",
        default=50,
        type=int,
        help="Log interval in steps."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0001,
        type=float,
        help="The peak learning rate. (Default: 0.0001)",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay (L2 penalty) (Default: 0.0)",
    )
    parser.add_argument(
        "--warmup_updates",
        default=5000,
        type=int,
        help="Number of steps for warm up the learning rate. (Default: 5000)",
    )
    parser.add_argument(
        "--max_updates",
        default=25000,
        type=int,
        help="Total number of training steps. (Default: 25000)",
    )
    parser.add_argument(
        "--clip_norm",
        default=10.0,
        type=float,
        help="The gradient norm value to clip. (Default: 10.0)",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--accum_grad",
        default=1,
        type=int,
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Precision for training."
    )

    # distillation related
    parser.add_argument(
        "--teacher_ckpt",
        default=pathlib.Path("pretrained_ckpts/hubert-base-ls960.pth"),
        type=pathlib.Path,
        help="Path to the teacher model checkpoint."
    )
    parser.add_argument(
        "--student_ckpt",
        type=pathlib.Path,
        help="Path to the student model checkpoint (for initialization)."
    )
    parser.add_argument(
        "--distill_layers",
        default="0.4,8,12",
        type=str,
        help="Distill layer indices (use period to separate groups and comma to separate layers within a group)."
    )
    parser.add_argument(
        "--distill_mode",
        type=str,
        default="layer2layer",
        choices=["layer2layer", "predlayer"],
        help="Distill mode, either layer2layer or predlayer."
    )
    parser.add_argument(
        "--l2_weight",
        default=0.0,
        type=float,
        help="Weight of MSE loss."
    )
    parser.add_argument(
        "--l1_weight",
        default=1.0,
        type=float,
        help="Weight of L1 loss."
    )
    parser.add_argument(
        "--cos_weight",
        default=1.0,
        type=float,
        help="Weight of cosine similarity loss."
    )
    parser.add_argument(
        "--cos_type",
        default="raw",
        type=str,
        choices=["raw", "log_sig"],
        help="Type of the cosine similarity loss."
    )
    
    return parser.parse_args()


def _init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    if _get_rank() == 0:
        _LG.setLevel(logging.INFO)
    else:
        _LG.setLevel(logging.WARN)


def cli_main():
    _init_logger()
    args = _parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
