import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageEnhance


# ============================================================
# Data Augmentation
# ============================================================

def random_flip(image, mask, edge):
    """Random horizontal flip."""

    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return image, mask, edge


def random_crop(image, mask, edge, border=30):
    """Random center-region crop."""

    width, height = image.size

    crop_width = np.random.randint(
        max(width - border, 1),
        width + 1,
    )

    crop_height = np.random.randint(
        max(height - border, 1),
        height + 1,
    )

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    crop_region = (
        left,
        top,
        right,
        bottom,
    )

    return (
        image.crop(crop_region),
        mask.crop(crop_region),
        edge.crop(crop_region),
    )


def random_rotation(image, mask, edge):
    """Random rotation within [-15, 15) degrees."""

    if random.random() > 0.8:

        angle = np.random.randint(-15, 15)

        # Bicubic interpolation is used for RGB images.
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
        )

        # Nearest-neighbor interpolation preserves binary labels.
        mask = mask.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
        )

        edge = edge.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
        )

    return image, mask, edge


def color_enhance(image):
    """Random color augmentation."""

    brightness = random.randint(5, 15) / 10.0
    contrast = random.randint(5, 15) / 10.0
    color = random.randint(0, 20) / 10.0
    sharpness = random.randint(0, 30) / 10.0

    image = ImageEnhance.Brightness(
        image
    ).enhance(brightness)

    image = ImageEnhance.Contrast(
        image
    ).enhance(contrast)

    image = ImageEnhance.Color(
        image
    ).enhance(color)

    image = ImageEnhance.Sharpness(
        image
    ).enhance(sharpness)

    return image


def random_pepper(image, noise_ratio=0.0015):
    """Add sparse salt-and-pepper noise."""

    image = np.array(image)

    num_noise = int(
        noise_ratio
        * image.shape[0]
        * image.shape[1]
    )

    for _ in range(num_noise):

        x = random.randint(
            0,
            image.shape[0] - 1,
        )

        y = random.randint(
            0,
            image.shape[1] - 1,
        )

        image[x, y] = (
            0 if random.random() < 0.5 else 255
        )

    return Image.fromarray(image)


# ============================================================
# Training Dataset
# ============================================================

class CODDataset(data.Dataset):
    """
    Dataset for camouflaged object detection training.

    Expected directory structure:

        TrainDataset/
        ├── Imgs/
        ├── GT/
        └── Edge/
    """

    def __init__(
        self,
        image_root,
        gt_root,
        edge_root,
        trainsize,
    ):
        super().__init__()

        self.trainsize = trainsize

        self.images = self._collect_files(
            image_root,
            extensions=(".jpg", ".jpeg", ".png"),
        )

        self.masks = self._collect_files(
            gt_root,
            extensions=(".jpg", ".jpeg", ".png"),
        )

        self.edges = self._collect_files(
            edge_root,
            extensions=(".jpg", ".jpeg", ".png"),
        )

        self._filter_files()

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (trainsize, trainsize),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (trainsize, trainsize),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

        self.edge_transform = transforms.Compose(
            [
                transforms.Resize(
                    (trainsize, trainsize),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

        self.edge_kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        self.size = len(self.images)

    @staticmethod
    def _collect_files(root, extensions):
        """Collect and sort files from a directory."""

        files = [
            os.path.join(root, filename)
            for filename in os.listdir(root)
            if filename.lower().endswith(extensions)
        ]

        return sorted(files)

    def _filter_files(self):
        """Keep image-mask-edge triplets with matching sizes."""

        if not (
            len(self.images)
            == len(self.masks)
            == len(self.edges)
        ):
            raise RuntimeError(
                "The numbers of images, masks, and edges "
                "must be identical."
            )

        valid_images = []
        valid_masks = []
        valid_edges = []

        for image_path, mask_path, edge_path in zip(
            self.images,
            self.masks,
            self.edges,
        ):

            with Image.open(image_path) as image:
                image_size = image.size

            with Image.open(mask_path) as mask:
                mask_size = mask.size

            with Image.open(edge_path) as edge:
                edge_size = edge.size

            if (
                image_size == mask_size
                and image_size == edge_size
            ):
                valid_images.append(image_path)
                valid_masks.append(mask_path)
                valid_edges.append(edge_path)

        self.images = valid_images
        self.masks = valid_masks
        self.edges = valid_edges

    @staticmethod
    def _load_rgb(path):
        """Load an RGB image."""

        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _load_binary(path):
        """Load a grayscale mask."""

        with Image.open(path) as image:
            return image.convert("L")

    @staticmethod
    def _binarize_edge(edge):
        """Convert edge tensor into a binary tensor."""

        return (edge > 0).float()

    def __getitem__(self, index):

        image = self._load_rgb(
            self.images[index]
        )

        mask = self._load_binary(
            self.masks[index]
        )

        # ----------------------------------------------------
        # Edge preprocessing
        # ----------------------------------------------------

        edge = cv2.imread(
            self.edges[index],
            cv2.IMREAD_GRAYSCALE,
        )

        if edge is None:
            raise RuntimeError(
                "Failed to load edge map: {}".format(
                    self.edges[index]
                )
            )

        edge = cv2.dilate(
            edge,
            self.edge_kernel,
            iterations=1,
        )

        edge = Image.fromarray(edge)

        # ----------------------------------------------------
        # Data augmentation
        # ----------------------------------------------------

        image, mask, edge = random_flip(
            image,
            mask,
            edge,
        )

        image, mask, edge = random_crop(
            image,
            mask,
            edge,
        )

        image, mask, edge = random_rotation(
            image,
            mask,
            edge,
        )

        image = color_enhance(image)

        mask = random_pepper(mask)
        edge = random_pepper(edge)

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        edge = self.edge_transform(edge)

        edge = self._binarize_edge(edge)

        return image, mask, edge

    def __len__(self):
        return self.size


# ============================================================
# DataLoader
# ============================================================

def seed_worker(worker_id):
    """Set random seeds for each DataLoader worker."""

    worker_seed = (
        torch.initial_seed() % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(
    image_root,
    gt_root,
    edge_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
):
    """Build the training DataLoader."""

    dataset = CODDataset(
        image_root=image_root,
        gt_root=gt_root,
        edge_root=edge_root,
        trainsize=trainsize,
    )

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    return loader


# ============================================================
# Test / Validation Dataset
# ============================================================

class TestDataset:
    """
    Dataset loader for validation and testing.

    Expected directory structure:

        Dataset/
        ├── Imgs/
        └── GT/
    """

    def __init__(
        self,
        image_root,
        gt_root,
        testsize,
    ):

        self.testsize = testsize

        self.images = CODDataset._collect_files(
            image_root,
            extensions=(
                ".jpg",
                ".jpeg",
                ".png",
            ),
        )

        self.masks = CODDataset._collect_files(
            gt_root,
            extensions=(
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff",
            ),
        )

        if len(self.images) != len(self.masks):
            raise RuntimeError(
                "The numbers of test images and masks "
                "must be identical."
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (testsize, testsize),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.size = len(self.images)
        self.index = 0

    def load_data(self):

        image_path = self.images[
            self.index
        ]

        mask_path = self.masks[
            self.index
        ]

        image = self._load_rgb(
            image_path
        )

        mask = self._load_binary(
            mask_path
        )

        image_tensor = (
            self.transform(image)
            .unsqueeze(0)
        )

        image_for_post = image.resize(
            mask.size,
            resample=Image.Resampling.BILINEAR,
        )

        name = os.path.basename(
            image_path
        )

        name = os.path.splitext(
            name
        )[0] + ".png"

        self.index += 1
        self.index %= self.size

        return (
            image_tensor,
            mask,
            name,
            np.array(image_for_post),
        )

    @staticmethod
    def _load_rgb(path):

        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _load_binary(path):

        with Image.open(path) as image:
            return image.convert("L")

    def __len__(self):
        return self.size


# ============================================================
# Backward Compatibility
# ============================================================

# Keep compatibility with the original training/testing scripts:
#
# from utils.data_val import test_dataset
#
test_dataset = TestDataset
