# SI23DCQA

Code and data resources for **SI23DCQA: Perceptual Quality Assessment of Single Image-to-3D Content**.

[Paper](https://ieeexplore.ieee.org/document/11210072) · [Dataset on ModelScope](https://modelscope.cn/datasets/WeiWuWeiwu/SI23DCQA)

## Overview

This repository provides a multi-modal quality assessment model and image/video baselines for single-image-to-3D content. The proposed model predicts three perceptual quality scores: **overall**, **color**, and **shape**.

The model combines input-image features, rendered-frame features, SlowFast motion features, PointNet++ point-cloud patch features, and mPLUG-Owl2 features through cross-modal attention. It supports within-category, mixed-category, and cross-category experiments.

## Installation

Use Python 3.10 or 3.11 in a separate environment. Run the commands below from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install a matching PyTorch/torchvision pair for your CUDA setup using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), within the version ranges in `requirements.txt`. Then install the training dependencies:

```bash
python -m pip install -r requirements.txt
```

If you need to generate feature caches, also install the preprocessing dependencies:

```bash
python -m pip install -r requirements-preprocess.txt
```

The requirements specify compatibility constraints, not a lockfile of the original paper environment. CUDA is recommended for training and feature extraction. The proposed model loads pretrained Swin weights through `timm`; feature extraction also loads SlowFast and mPLUG-Owl2 weights. Make these weights available locally or allow their initial download.

## Data preparation

### Dataset and annotations

Download the dataset from [ModelScope](https://modelscope.cn/datasets/WeiWuWeiwu/SI23DCQA). Place it in `data/`, or pass `--data-root /path/to/I23DCQA` to the commands below.

Training scripts resolve the dataset root in this order: `--data-root`, the `SI23DCQA_DATA_ROOT` environment variable, then `dataset.root` in the YAML configuration. Preprocessing and checking scripts use their own `--data-root` option, which defaults to `./data`.

The supplied configurations refer to these annotation files:

| Experiment subset | Annotation file |
| --- | --- |
| All categories | `all-result.txt` |
| `true` | `true-result.txt` |
| `aigi` | `aigi-result.txt` |
| `mri` | `mri-result.txt` |

Each annotation row contains six pipe-separated fields, with no header:

```text
image_path|mesh_path|video_path|overall_MOS|color_MOS|shape_MOS
```

Paths are relative to the dataset root. Training divides each mean opinion score (MOS) by 100. Preserve the image and mesh paths specified in the annotations. Rendered videos and derived features use the following layout, where `<category>`, `<method>`, and `<object>` stand for the corresponding dataset names:

```text
data/
├── all-result.txt
├── true-result.txt
├── aigi-result.txt
├── mri-result.txt
├── video_single/<category>/<method>/<object>.mp4
├── pointcloud/<category>/<method>/<object>.ply
└── features/<category>/<method>/<object>/
    ├── frames/                  # 000.png through 007.png are required
    ├── feature_0_slow_feature.npy
    ├── feature_0_fast_feature.npy
    ├── ...                      # slow/fast pairs for indices 0 through 7
    ├── patch.npy
    ├── 3D.pt
    └── compare2.pt
```

Input images and meshes are omitted from this schematic because their locations are defined by the annotations. `convert_o2pc.py` reads the OBJ file at each annotated `mesh_path` and writes a PLY point cloud under `pointcloud/`, using the corresponding video path to determine its location. it does not sample mesh surfaces or extract texture-map colors. `get_patch_i23dcqa.py` then reads these PLY files to generate point-cloud patches.

### Feature extraction

The proposed model requires all caches listed below. If they are already prepared, skip extraction and run the data check.

| Cache | Contents expected by the model |
| --- | --- |
| `frames/000.png`–`007.png` | Eight rendered frames |
| `feature_{i}_slow_feature.npy` | 2,048-dimensional SlowFast slow features, `i = 0, …, 7` |
| `feature_{i}_fast_feature.npy` | 256-dimensional SlowFast fast features, `i = 0, …, 7` |
| `patch.npy` | Point coordinates; the extractor writes shape `(6, 2048, 3)` |
| `3D.pt` | mPLUG-Owl2 video-quality feature, shape `(1, 4096)` |
| `compare2.pt` | mPLUG-Owl2 image-comparison feature, shape `(1, 4096)` |

Run the following commands to generate the caches:

```bash
python extract_frames.py --data-root /path/to/I23DCQA
python extract_SlowFast_features.py --data-root /path/to/I23DCQA --device cuda:0
python convert_o2pc.py --data-root /path/to/I23DCQA
python get_patch_i23dcqa.py --data-root /path/to/I23DCQA
python getlmmfeatures.py --data-root /path/to/I23DCQA --mode video --device cuda:0
python getlmmfeatures.py --data-root /path/to/I23DCQA --mode compare --device cuda:0
```

These commands use `all-result.txt` by default. Pass `--ann aigi-result.txt`, for example, to process a single subset. Frame extraction samples approximately one frame per second and pads short videos to at least eight frames; the proposed model reads the first eight cached frames and SlowFast feature pairs.

The default mPLUG-Owl2 model is `MAGAer13/mplug-owl2-llama2-7b`; use `--model-path /path/to/model` to load local weights. Both `video` and `compare` modes are required. The optional `identity` mode produces `compare.pt`, which the current model does not read.

### Data validation

Check annotation parsing and the existence of input images and rendered videos:

```bash
python check_data.py --data-root /path/to/I23DCQA
```

For the proposed model, also check the required caches and NumPy array dimensions:

```bash
python check_data.py --data-root /path/to/I23DCQA --features --check-shapes
```

Use `--ann` to select another annotation file or `--limit 10` for a quick check. The checker does not validate meshes, source point clouds, or the tensor contents of `.pt` caches.

## Training and evaluation

### Proposed model

Train on all categories with an 80/20 split of unique input images:

```bash
python train_my.py -c config/my-all.yml --data-root /path/to/I23DCQA --device 0
```

Use `config/my-true.yml`, `config/my-aigi.yml`, or `config/my-mri.yml` for an individual category. To run all four supplied experiments sequentially:

```bash
bash run-my.sh --data-root /path/to/I23DCQA --device 0
```

### Cross-category experiments

The supplied cross-category configurations train on `true` and evaluate on `aigi` (`ta`) or `mri` (`tm`):

```bash
python train_my_cross.py -c config-cross/my-ta.yml --data-root /path/to/I23DCQA --device 0
python train_my_cross.py -c config-cross/my-tm.yml --data-root /path/to/I23DCQA --device 0
```

Alternatively, run both experiments with:

```bash
bash run-cross.sh --data-root /path/to/I23DCQA --device 0
```

### Image and video baselines

The baseline loaders sample rendered videos directly and do not require the proposed model's feature caches.

```bash
python train_iqa.py -c config/resnet50-all.yml --data-root /path/to/I23DCQA --device 0
python train_vqa.py -c config/mc318-all.yml --data-root /path/to/I23DCQA --device 0
```

Cross-category baselines use the corresponding entry points and configurations:

```bash
python train_iqa_cross.py -c config-cross/swinl-ta.yml --data-root /path/to/I23DCQA --device 0
python train_vqa_cross.py -c config-cross/swinl3dt-ta.yml --data-root /path/to/I23DCQA --device 0
```

## Citation

If you use SI23DCQA in your research, please consider citing the following works:

```bibtex
@article{fu2026multi,
  title={Multi-Dimensional Quality Assessment for Single-Image-to-3D Contents: Dataset and Model},
  author={Fu, Kang and Duan, Huiyu and Zhang, Zicheng and Liu, Jing and Liu, Yun and Liu, Xiaohong and Wang, Jia and Min, Xiongkuo and Le Callet, Patrick and Zhai, Guangtao},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  publisher={IEEE}
}

@inproceedings{fu2025si23dcqa,
  title={Si23dcqa: perceptual quality assessment of single image-to-3d content},
  author={Fu, Kang and Duan, Huiyu and Zhang, Zicheng and Liu, Xiaohong and Min, Xiongkuo and Wang, Jia and Zhai, Guangtao},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

## License

The repository includes an [MIT license](LICENSE.txt). Bundled third-party components retain their respective license notices.
