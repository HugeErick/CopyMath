---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: latex
    dtype: string
  - name: sample_id
    dtype: string
  - name: split_tag
    dtype: string
  - name: data_type
    dtype: string
  splits:
  - name: train
    num_bytes: 1308313988.28
    num_examples: 229864
  - name: test
    num_bytes: 50449700.38
    num_examples: 7644
  - name: val
    num_bytes: 92725986.108
    num_examples: 15674
  download_size: 1247446895
  dataset_size: 1451489674.7680001
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: val
    path: data/val-*
task_categories:
- image-to-text
tags:
- math
- latex
- handwritten
- ocr
size_categories:
- 100K<n<1M
---
# Dataset Card for MathWriting

## Dataset Summary

The **MathWriting** dataset contains online handwritten mathematical expressions collected through a prompted interface and rendered to RGB images. It consists of **230,000 human-written expressions**, each paired with its corresponding LaTeX string. The dataset is intended to support research in **online and offline handwritten mathematical expression (HME) recognition**.

Key features:

- Online handwriting converted to rendered RGB images.
- Each sample is labeled with a LaTeX expression.
- Includes splits: `train`, `val`, and `test`.
- All samples in this release are **human-written** (no synthetic data).
- Image preprocessing includes resizing (max dimension ≤ 512 px), stroke width jitter, and subtle color perturbations.

---

## Supported Tasks and Leaderboards

**Primary Task:**  
- *Handwritten Mathematical Expression Recognition (HMER)*: Given an image of a handwritten formula, predict its LaTeX representation.

This dataset is also suitable for:
- Offline HME recognition (from rendered images).
- Sequence modeling and encoder-decoder learning.
- Symbol layout analysis and parsing in math.

---

## Dataset Structure

Each example has the following structure:

```python
{
    'image': <PIL.Image.Image in RGB mode>,
    'latex': str,  # the latex string"
    'sample_id': str,  # unique identifier
    'split_tag': str,  # "train", "val", or "test"
    'data_type': str,  # always "human" in this version
}
```

All samples are rendered from digital ink into JPEG images with randomized stroke width and light RGB variations for augmentation and realism.

## Usage

To load the dataset:

```python
from datasets import load_dataset

ds = load_dataset("deepcopy/MathWriting-Human")
sample = ds["train"][0]
image = sample["image"]
latex = sample["latex"]
```

## Licensing Information

The dataset is licensed by **Google LLC** under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

---

## Citation

Please cite the following paper if you use this dataset:

```
@misc{gervais2025mathwritingdatasethandwrittenmathematical,
      title={MathWriting: A Dataset For Handwritten Mathematical Expression Recognition}, 
      author={Philippe Gervais and Anastasiia Fadeeva and Andrii Maksai},
      eprint={2404.10690},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.10690}, 
}
```