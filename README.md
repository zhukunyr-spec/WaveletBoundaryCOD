# WaveletBoundaryCOD
Official implementation of A Lightweight Progressive Wavelet Boundary-Guided Difference Fusion Network for Camouflaged Object Detection

# WaveletBoundaryCOD

Official PyTorch implementation of:

**A Lightweight Progressive Wavelet Boundary-Guided Difference Fusion Network for Camouflaged Object Detection**

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
torchvision
numpy
Pillow
opencv-python
tensorboard
```

## Dataset

## Dataset

Training data:

```text
TrainDataset/
├── Imgs/
├── GT/
└── Edge/
```

TrainDataset:
https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing

Testing data:

```text
TestDataset/
├── CAMO/
├── CHAMELEON/
├── COD10K/
└── NC4K/
```

TestDataset:
https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing

Each testing dataset should contain:

```text
Imgs/
GT/
```


## Training

```bash
python train.py \
    --train_root path/to/TrainDataset \
    --val_root path/to/COD10K \
    --backbone_checkpoint path/to/smt_tiny.pth
```

The default feature channel is `48`.

## Testing

```bash
python test.py \
    --pth_path path/to/model.pth \
    --test_dataset_path path/to/TestDataset \
    --dataset COD10K \
    --save_path ./results
```

Supported datasets:

```text
CAMO
CHAMELEON
COD10K
NC4K
```

## Pretrained Model

Pretrained weights will be released soon.

## Results

Our model contains **12.55M parameters** with **9.05G FLOPs**.

## Citation

If you find this work useful, please consider citing our paper.

```bibtex
@article{zhu2026waveletboundarycod,
  title={A Lightweight Progressive Wavelet Boundary-Guided Difference Fusion Network for Camouflaged Object Detection},
  author={Zhu, Kun and Zhao, Dexin},
  year={2026}
}
```

## License

This project is released under the MIT License.
