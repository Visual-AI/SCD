# What’s in a Name? Beyond Class Indices for Image Recognition

**This repository is the official implementation of the CVPR2024 CVinW workshop paper (Spotlight): "What’s in a Name? Beyond Class Indices for Image Recognition"**

Kai Han, Xiaohu Huang, Yandong Li, Sagar Vaze, Jie Li, and Xuhui Jia

 [[`Paper`]](https://arxiv.org/abs/2304.02364)

# Introduction

[teaser image](assets/SCD_teaser.png)

Our paper (SCD) leverages an unconstrained vocabulary to semantically name image objects without relying on predefined classes. The model effectively narrows down candidate names by utilizing non-parametric clustering and a voting method, enhancing the image recognition process with semantic depth.

# License

SCD is released under the [`CC BY-NC-SA 4.0 license`](https://creativecommons.org/licenses/by-nc-sa/4.0/).

# Performance

We conduct experiments in two settings, i.e., unsupervised and partially supervised.

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

Besides, you need to get into the `local_utils/k_means_constrained` folder, and install the package:

```bash
python setup.py install
```

# Data Preparation

The used datasets can be donwloaded from the links below:

| Dataset                                | Link |
|:--------------------------------------:|:-----------------:|
|        CUB         |    [Link](https://www.vision.caltech.edu/datasets/cub_200_2011/)      |
|   Standford Dogs   |       [link](http://vision.stanford.edu/aditya86/ImageNetDogs/)      |
|   ImageNet  |    [link](https://www.image-net.org/download.php)     |

You also need to download the [extracted features](https://drive.google.com/file/d/1ZLFK3US7ZrF7Rs3TpZQ9IyI-7IThfips/view?usp=drive_link), [gcd pretrained weights](https://drive.google.com/file/d/1BU9eqfriF0tRKfeYfn88yOoW9P-GUqR7/view?usp=drive_link), and [zero-shot weights](https://drive.google.com/file/d/1ZpMNSJdKakYi5RIQwtpoxesAwagv5wci/view?usp=drive_link) and put them into the respective folders.

# Evaluation

- **Unsupervised Setting**

You can just modify the configurations based on what you needs in the script.

```bash
sh script/evaluate_unsupervised.sh
```

- **Partially Supervised Setting**

You can just modify the configurations based on what you needs in the script.

```bash
sh script/evaluate_unsupervised.sh
```

# Citation

```bibtex
@inproceedings{han2024whats,
  title={What's in a Name? Beyond Class Indices for Image Recognition},
  author={Kai Han and Xiaohu Huang and Yandong Li and Vaze Sagar and Jie Li and Xuhui Jia},
  booktitle={CVPR Workshops},
  year={2024}
}
```
