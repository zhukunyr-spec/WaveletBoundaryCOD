import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.smtb48_edge import Network
from utils.data_val import test_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing script for Camouflaged Object Detection"
    )

    parser.add_argument(
        "--trainsize",
        type=int,
        default=384,
        help="Input image size."
    )

    parser.add_argument(
        "--net_channel",
        type=int,
        default=48,
        help="Channel dimension of the network."
    )

    parser.add_argument(
        "--pth_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint."
    )

    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=True,
        help="Root directory of the test datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="COD10K",
        choices=["CAMO", "CHAMELEON", "COD10K", "NC4K"],
        help="Dataset used for testing."
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Directory for saving prediction maps."
    )

    return parser.parse_args()


def main():
    opt = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model = Network(opt.net_channel)

    checkpoint = torch.load(
        opt.pth_path,
        map_location=device
    )

    # Remove "module." prefix if the model was trained with DataParallel
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    checkpoint = {
        k.replace("module.", ""): v
        for k, v in checkpoint.items()
    }

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Prepare dataset
    # ---------------------------------------------------------
    data_path = os.path.join(
        opt.test_dataset_path,
        opt.dataset
    )

    image_root = os.path.join(data_path, "Imgs")
    gt_root = os.path.join(data_path, "GT")

    save_path = os.path.join(
        opt.save_path,
        opt.dataset
    )
    os.makedirs(save_path, exist_ok=True)

    test_loader = test_dataset(
        image_root,
        gt_root,
        opt.trainsize
    )

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    with torch.no_grad():

        for i in range(test_loader.size):

            image, gt, name, _ = test_loader.load_data()

            print(
                "[{}/{}] {} - {}".format(
                    i + 1,
                    test_loader.size,
                    opt.dataset,
                    name
                )
            )

            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8

            image = image.to(device)

            _, _, res, _, _ = model(image)

            res = F.interpolate(
                res,
                size=gt.shape,
                mode="bilinear",
                align_corners=False
            )

            res = torch.sigmoid(res)
            res = res.cpu().numpy().squeeze()

            save_file = os.path.join(
                save_path,
                name
            )

            cv2.imwrite(
                save_file,
                (res * 255).astype(np.uint8)
            )


if __name__ == "__main__":
    main()
