# FedProTIP: Task-Agnostic Federated Continual Learning via Replay-Free Gradient Projection

<p align="center">
  <b>Official implementation for our TMLR 2026 paper: FedProTIP</b><br>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=GW4aw0fUKC">
    <img src="https://img.shields.io/badge/OpenReview-Paper-blue" alt="OpenReview">
  </a>
  <img src="https://img.shields.io/badge/Venue-TMLR%202026-red" alt="TMLR 2026">
  <img src="https://img.shields.io/badge/Task-Federated%20Continual%20Learning-green" alt="Federated Continual Learning">
</p>

---

## Overview

**FedProTIP** is a federated continual learning method designed for task-agnostic settings, where clients learn from sequentially arriving tasks without relying on explicit task identities during inference.

The method combines replay-free gradient projection with task identity prediction to mitigate catastrophic forgetting while preserving privacy constraints in federated learning.

This repository provides code to reproduce experiments on:

- **10-split CIFAR-100**
- **6-split DomainNet**
- **ImageNet-R** with 5, 10, and 20 task splits

For more details, please refer to our paper:

**OpenReview:** [https://openreview.net/forum?id=GW4aw0fUKC](https://openreview.net/forum?id=GW4aw0fUKC)

---

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
````

---

## Running Experiments

### 10-split CIFAR-100

```bash
python main.py --config configs/cifar100/fedprotip.json
```

---

### 6-split DomainNet

```bash
python main.py --config configs/domainnet/dom_fedprotip.json
```

---

### ImageNet-R

#### 5-split ImageNet-R

```bash
python main.py --config configs/imagenetr/imagenet-r_fedprotip.json --n_tasks 5 --increment 40
```

#### 10-split ImageNet-R

```bash
python main.py --config configs/imagenetr/imagenet-r_fedprotip.json --n_tasks 10 --increment 20
```

#### 20-split ImageNet-R

```bash
python main.py --config configs/imagenetr/imagenet-r_fedprotip.json --n_tasks 20 --increment 10
```

---

## Code References

This repository builds on and adapts components from the following open-source implementations:

* [LANDER](https://github.com/tmtuan1307/LANDER/tree/main)
* [GPM](https://github.com/sahagobinda/GPM/tree/main)

We sincerely thank the authors for making their code publicly available.

---

## Citation

If you find this repository useful, please cite our paper!

```bibtex
@article{cha2025task,
  title={Task-Agnostic Federated Continual Learning via Replay-Free Gradient Projection},
  author={Cha, Seohyeon and Chen, Huancheng and Vikalo, Haris},
  journal={arXiv preprint arXiv:2509.21606},
  year={2025}
}
```


