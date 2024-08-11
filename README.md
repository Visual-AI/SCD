# What’s in a Name? Beyond Class Indices for Image Recognition

**This repository is the official implementation of the CVPR2024 CVinW workshop paper: "What’s in a Name? Beyond Class Indices for Image Recognition"**

Kai Han, Xiaohu Huang, Yandong Li, Sagar Vaze, Jie Li, and Xuhui Jia
 [[`Paper`]](https://arxiv.org/abs/2304.02364)

# Introduction

Our paper (SCD) introduces leverages a vast vocabulary to semantically name image objects without relying on predefined classes. Utilizing non-parametric clustering and a voting system, the model effectively narrows down candidate names, enhancing the image recognition process with semantic depth.

# License

SCD is released under the [`CC BY-NC-SA 4.0 license`](https://creativecommons.org/licenses/by-nc-sa/4.0/).

# Performance

We conduct experiments on two settings, i.e., unsupervised and partially supervised.

**Table 1. Results in the unsupervised setting.** We use DINO features for the initial clustering step and report metrics for semantic accuracy (involving class naming, left) and clustering (right). ‘TE’ denotes using the textual enhancement technique.

| Method                                 | ImageNet-100 sACC | Soft-sACC | ACC               | Stanford Dogs sACC | Soft-sACC | ACC               | CUB sACC | Soft-sACC | ACC               |
|:--------------------------------------:|:-----------------:|:---------:|:-----------------:|:------------------:|:---------:|:-----------------:|:--------:|:---------:|:-----------------:|
| Zero-shot transfer (UB)                | 85.0              | 92.0      | 85.1              | 60.4               | 83.2      | 60.8              | 54.1     | 83.2      | 55.8              |
| Zero-shot transfer (Baseline)          | 22.7              | 57.7      | 73.2              | 51.7               | 77.4      | 47.2              | 20.2     | 77.4      | 34.4              |
| Ours (Semantic Naming)                 | 41.2              | 71.3      | 78.2              | 53.8               | 79.1      | 57.9              | 24.5     | 79.1      | **46.5**          |
| Ours (Semantic Naming) w/TE            | **43.0**          | **72.5**  | **81.3**          | **54.1**           | **80.0**  | **58.7**          | **33.5** | **80.0**  | 42.6              |

**Table 2. Results in the partially supervised setting.** We use GCD features for the initial clustering step and report metrics for semantic accuracy (involving class naming, left) and clustering (right). ‘TE’ denotes using the textual enhancement technique.

| Method                                 | ImageNet-100 sACC | Soft-sACC | ACC               | Stanford Dogs sACC | Soft-sACC | ACC               | CUB sACC | Soft-sACC | ACC               |
|:--------------------------------------:|:-----------------:|:---------:|:-----------------:|:------------------:|:---------:|:-----------------:|:--------:|:---------:|:-----------------:|
| Zero-shot transfer (UB)                | 85.0              | 92.0      | 85.1              | 60.4               | 83.2      | 60.8              | 54.1     | 55.8      | 55.8              |
| Zero-shot transfer (Baseline)          | 22.7              | 57.7      | 74.1              | 51.7               | 77.4      | 60.8              | 20.2     | 57.7      | **54.0**          |
| Ours (Semantic Naming)                 | 54.8              | **77.5**  | 78.7              | 53.7               | 79.6      | **62.1**          | 35.3     | 79.6      | 52.9              |
| Ours (Semantic Naming) w/TE            | **55.7**          | 76.5      | **80.6**          | **55.5**           | **80.6**  | 58.8              | **35.3** | **80.6**  | 42.5              |


# Dependency

To install the dependencies, you can use the the following command:

```bash
pip install -r requirements.txt
```

# Data Preparation

The used datasets can be donwloaded from the links below:

| Dataset                                | Link |
|:--------------------------------------:|:-----------------:|
|        CUB         |    [Link](https://www.vision.caltech.edu/datasets/cub_200_2011/)      |
|   Standford Dogs   |       [link](http://vision.stanford.edu/aditya86/ImageNetDogs/)      |
|   ImageNet  |    [link](https://www.image-net.org/download.php)     |

You can also download the [extracted features](xxx), [gcd pretrained weights](xxx), and [zero-shot weights](xxx) and put them into the respective folders.

# Evaluation

- **Unsupervised Setting**

You can modify the configurations based on your need.

```bash
sh script/evaluate_unsupervised.sh
```

- **Partially Supervised Setting**

You can modify the configurations based on your need.

```bash
sh script/evaluate_unsupervised.sh
```

# Citation

```bibtex
@inproceedings{
  han2024s,
  title={What's in a Name? Beyond Class Indices for Image Recognition},
  author={Han, Kai and Li, Yandong and Vaze, Sagar and Li, Jie and Jia, Xuhui},
  booktitle={Computer Vision in the Wild (CVinW) Workshop of CVPR2024},
  year={2024}
}
```