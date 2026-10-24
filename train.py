import os
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from lib.wavelet_boundary_cod import Network
from utils.data_val import get_loader, test_dataset


# ============================================================
# Loss Functions
# ============================================================

def structure_loss(pred, mask):
    """
    Structure-aware loss:
    weighted BCE + weighted IoU.
    """

    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(
            mask,
            kernel_size=31,
            stride=1,
            padding=15,
        ) - mask
    )

    wbce = F.binary_cross_entropy_with_logits(
        pred,
        mask,
        reduction="none",
    )

    wbce = (
        (weight * wbce).sum(dim=(2, 3))
        / weight.sum(dim=(2, 3))
    )

    pred = torch.sigmoid(pred)

    intersection = (
        (pred * mask) * weight
    ).sum(dim=(2, 3))

    union = (
        (pred + mask) * weight
    ).sum(dim=(2, 3))

    wiou = 1 - (
        (intersection + 1)
        / (union - intersection + 1)
    )

    return (wbce + wiou).mean()


def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for auxiliary edge supervision.
    """

    pred = torch.sigmoid(pred)

    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (
        pred * target
    ).sum(dim=1)

    numerator = (
        2.0 * intersection + smooth
    )

    denominator = (
        pred.pow(2).sum(dim=1)
        + target.pow(2).sum(dim=1)
        + smooth
    )

    loss = 1.0 - numerator / denominator

    return loss.mean()


# ============================================================
# Checkpoint Utilities
# ============================================================

def get_model_state_dict(model):
    """
    Return model weights without the DataParallel 'module.' prefix.
    """

    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()

    return model.state_dict()


def save_checkpoint(model, save_path):
    """
    Save model checkpoint.
    """

    torch.save(
        get_model_state_dict(model),
        save_path,
    )


def load_checkpoint(model, checkpoint_path):
    """
    Load checkpoint and automatically remove the
    optional 'module.' prefix.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        checkpoint = checkpoint["state_dict"]

    state_dict = {}

    for key, value in checkpoint.items():

        if key.startswith("module."):
            key = key[len("module."):]

        state_dict[key] = value

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    device,
    epoch,
    total_epochs,
    edge_weight,
    grad_clip,
    writer=None,
):
    """
    Train the network for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_steps = len(train_loader)

    for step, (
        images,
        masks,
        edges,
    ) in enumerate(
        train_loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        edges = edges.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        #
        # Network output:
        # pred3, pred2, pred1, edge_pred, pred4
        # ----------------------------------------------------

        (
            pred3,
            pred2,
            pred1,
            edge_pred,
            pred4,
        ) = model(images)

        # ----------------------------------------------------
        # Object prediction losses
        # ----------------------------------------------------

        loss1 = structure_loss(
            pred1,
            masks,
        )

        loss2 = structure_loss(
            pred2,
            masks,
        )

        loss3 = structure_loss(
            pred3,
            masks,
        )

        loss_coarse = structure_loss(
            pred4,
            masks,
        )

        # ----------------------------------------------------
        # Edge supervision
        # ----------------------------------------------------

        loss_edge = dice_loss(
            edge_pred,
            edges,
        )

        # ----------------------------------------------------
        # Total Loss
        #
        # L_total =
        # L1 + L2 + L3 + Lc + lambda_e * Le
        #
        # lambda_e = 4.0 by default
        # ----------------------------------------------------

        loss = (
            loss1
            + loss2
            + loss3
            + loss_coarse
            + edge_weight * loss_edge
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        if grad_clip > 0:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        total_loss += loss.item()

        # ----------------------------------------------------
        # Console Output
        # ----------------------------------------------------

        if (
            step == 1
            or step % 100 == 0
            or step == total_steps
        ):

            print(
                "Epoch [{:03d}/{:03d}] "
                "Step [{:04d}/{:04d}] | "
                "L1 {:.4f} | "
                "L2 {:.4f} | "
                "L3 {:.4f} | "
                "Lc {:.4f} | "
                "Le {:.4f} | "
                "Total {:.4f}".format(
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    loss1.item(),
                    loss2.item(),
                    loss3.item(),
                    loss_coarse.item(),
                    loss_edge.item(),
                    loss.item(),
                )
            )

    average_loss = (
        total_loss / total_steps
    )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    if writer is not None:

        writer.add_scalar(
            "Train/TotalLoss",
            average_loss,
            epoch,
        )

        writer.add_scalar(
            "Train/Loss1",
            loss1.item(),
            epoch,
        )

        writer.add_scalar(
            "Train/Loss2",
            loss2.item(),
            epoch,
        )

        writer.add_scalar(
            "Train/Loss3",
            loss3.item(),
            epoch,
        )

        writer.add_scalar(
            "Train/CoarseLoss",
            loss_coarse.item(),
            epoch,
        )

        writer.add_scalar(
            "Train/EdgeLoss",
            loss_edge.item(),
            epoch,
        )

    return average_loss


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    val_loader,
    model,
    device,
):
    """
    Evaluate the finest prediction using MAE.
    """

    model.eval()

    mae_sum = 0.0

    for _ in range(
        val_loader.size
    ):

        image, gt, _, _ = (
            val_loader.load_data()
        )

        gt = np.asarray(
            gt,
            dtype=np.float32,
        )

        if gt.max() > 1:
            gt /= 255.0

        image = image.to(device)

        # ----------------------------------------------------
        # The third output corresponds to pred1,
        # i.e. the finest prediction.
        # ----------------------------------------------------

        _, _, pred, _, _ = model(
            image
        )

        pred = F.interpolate(
            pred,
            size=gt.shape,
            mode="bilinear",
            align_corners=False,
        )

        pred = torch.sigmoid(pred)

        pred = (
            pred.squeeze()
            .cpu()
            .numpy()
        )

        mae = np.mean(
            np.abs(pred - gt)
        )

        mae_sum += mae

    return (
        mae_sum / val_loader.size
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Training script for "
            "Wavelet Boundary-guided "
            "Camouflaged Object Detection"
        )
    )

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size.",
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=384,
        help="Input image size.",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=48,
        help="Unified feature channel dimension.",
    )

    parser.add_argument(
        "--edge_weight",
        type=float,
        default=4.0,
        help="Weight of auxiliary edge loss.",
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.5,
        help="Gradient clipping threshold.",
    )

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.1,
        help="Learning rate decay rate.",
    )

    parser.add_argument(
        "--decay_epoch",
        type=int,
        default=50,
        help="Learning rate decay interval.",
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    parser.add_argument(
        "--train_root",
        type=str,
        required=True,
        help="Root directory of training dataset.",
    )

    parser.add_argument(
        "--val_root",
        type=str,
        required=True,
        help="Root directory of validation dataset.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers.",
    )

    # --------------------------------------------------------
    # Backbone weights
    # --------------------------------------------------------

    parser.add_argument(
        "--backbone_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained SMT-Tiny weights.",
    )

    # --------------------------------------------------------
    # Resume training
    # --------------------------------------------------------

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training.",
    )

    # --------------------------------------------------------
    # Checkpoint saving
    #
    # Empty by default:
    # no model weights will be saved.
    # --------------------------------------------------------

    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help=(
            "Directory for saving checkpoints. "
            "Leave empty to disable checkpoint saving."
        ),
    )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help=(
            "TensorBoard log directory. "
            "Leave empty to disable TensorBoard logging."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(
        "Using device:",
        device,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    save_enabled = bool(
        args.save_path.strip()
    )

    if save_enabled:

        os.makedirs(
            args.save_path,
            exist_ok=True,
        )

        logging.basicConfig(
            filename=os.path.join(
                args.save_path,
                "train.log",
            ),
            level=logging.INFO,
            format=(
                "[%(asctime)s] "
                "%(levelname)s: "
                "%(message)s"
            ),
        )

        print(
            "Checkpoint saving enabled:",
            args.save_path,
        )

    else:

        print(
            "Checkpoint saving disabled."
        )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    writer = None

    if args.log_dir.strip():

        os.makedirs(
            args.log_dir,
            exist_ok=True,
        )

        writer = SummaryWriter(
            log_dir=args.log_dir
        )

        print(
            "TensorBoard logging enabled:",
            args.log_dir,
        )

    # --------------------------------------------------------
    # Build Model
    # --------------------------------------------------------

    model = Network(
        channels=args.channels,
        backbone_checkpoint=args.backbone_checkpoint,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if args.resume is not None:

        model = load_checkpoint(
            model,
            args.resume,
        )

        print(
            "Loaded checkpoint:",
            args.resume,
        )

    model = model.to(device)

    # --------------------------------------------------------
    # Multi-GPU
    # --------------------------------------------------------

    if torch.cuda.device_count() > 1:

        print(
            "Using {} GPUs.".format(
                torch.cuda.device_count()
            )
        )

        model = nn.DataParallel(
            model
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # --------------------------------------------------------
    # Learning Rate Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.decay_epoch,
            gamma=args.decay_rate,
        )
    )

    # --------------------------------------------------------
    # Dataset Paths
    # --------------------------------------------------------

    train_image_root = os.path.join(
        args.train_root,
        "Imgs",
    )

    train_gt_root = os.path.join(
        args.train_root,
        "GT",
    )

    train_edge_root = os.path.join(
        args.train_root,
        "Edge",
    )

    val_image_root = os.path.join(
        args.val_root,
        "Imgs",
    )

    val_gt_root = os.path.join(
        args.val_root,
        "GT",
    )

    # --------------------------------------------------------
    # Data Loaders
    # --------------------------------------------------------

    print(
        "Loading training data..."
    )

    train_loader = get_loader(
        image_root=train_image_root,
        gt_root=train_gt_root,
        edge_root=train_edge_root,
        batchsize=args.batch_size,
        trainsize=args.train_size,
        num_workers=args.num_workers,
    )

    print(
        "Loading validation data..."
    )

    val_loader = test_dataset(
        image_root=val_image_root,
        gt_root=val_gt_root,
        testsize=args.train_size,
    )

    # --------------------------------------------------------
    # Training Information
    # --------------------------------------------------------

    print("=" * 60)

    print(
        "Training Configuration"
    )

    print(
        "Epochs       :",
        args.epochs,
    )

    print(
        "Learning Rate:",
        args.lr,
    )

    print(
        "Batch Size   :",
        args.batch_size,
    )

    print(
        "Input Size   :",
        args.train_size,
    )

    print(
        "Channels     :",
        args.channels,
    )

    print(
        "Edge Weight  :",
        args.edge_weight,
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Best Validation Result
    # --------------------------------------------------------

    best_mae = float("inf")
    best_epoch = 0

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            "\nEpoch {:03d} | LR {:.8f}".format(
                epoch,
                current_lr,
            )
        )

        if writer is not None:

            writer.add_scalar(
                "Train/LearningRate",
                current_lr,
                epoch,
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            edge_weight=args.edge_weight,
            grad_clip=args.grad_clip,
            writer=writer,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if (
            epoch % 5 == 0
            or epoch == args.epochs
        ):

            mae = validate(
                val_loader,
                model,
                device,
            )

            print(
                "Validation | "
                "Epoch {:03d} | "
                "MAE {:.6f} | "
                "Best {:.6f}".format(
                    epoch,
                    mae,
                    best_mae,
                )
            )

            if writer is not None:

                writer.add_scalar(
                    "Validation/MAE",
                    mae,
                    epoch,
                )

            # ------------------------------------------------
            # Save best model only when save_path is provided
            # ------------------------------------------------

            if mae < best_mae:

                best_mae = mae
                best_epoch = epoch

                print(
                    "New best MAE: {:.6f}".format(
                        best_mae
                    )
                )

                if save_enabled:

                    best_model_path = os.path.join(
                        args.save_path,
                        "best_model.pth",
                    )

                    save_checkpoint(
                        model,
                        best_model_path,
                    )

                    print(
                        "Best model saved:",
                        best_model_path,
                    )

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if (
            save_enabled
            and epoch % 20 == 0
        ):

            checkpoint_path = os.path.join(
                args.save_path,
                "epoch_{:03d}.pth".format(
                    epoch
                ),
            )

            save_checkpoint(
                model,
                checkpoint_path,
            )

        # ----------------------------------------------------
        # Learning Rate Update
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        if save_enabled:

            logging.info(
                "Epoch %d | "
                "Loss %.6f | "
                "Best MAE %.6f | "
                "Best Epoch %d",
                epoch,
                train_loss,
                best_mae,
                best_epoch,
            )

    # --------------------------------------------------------
    # Final Model
    # --------------------------------------------------------

    if save_enabled:

        final_model_path = os.path.join(
            args.save_path,
            "final_model.pth",
        )

        save_checkpoint(
            model,
            final_model_path,
        )

        print(
            "Final model saved:",
            final_model_path,
        )

    # --------------------------------------------------------
    # Close TensorBoard
    # --------------------------------------------------------

    if writer is not None:
        writer.close()

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\nTraining completed.")

    print(
        "Best Epoch:",
        best_epoch,
    )

    print(
        "Best MAE:",
        best_mae,
    )


if __name__ == "__main__":
    main()
