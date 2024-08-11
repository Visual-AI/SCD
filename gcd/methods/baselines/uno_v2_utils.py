import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import torchvision.transforms as T

from PIL import ImageFilter, ImageOps
import random

import math
import warnings
from typing import List

from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from typing import Union

# --------------------------
# MODELS
# --------------------------

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        arch: Union[torch.nn.Module, str],
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=5,
        num_hidden_layers=1,
        pretrain_state_dict=None,
        feat_dim=None
    ):
        super().__init__()

        # backbone
        if isinstance(arch, str):
            self.encoder = models.__dict__[arch]()
            # Following is only relevant for a convolutional backbone
            self.encoder.fc = nn.Identity()
            # modify the encoder for lower resolution
            if low_res:
                self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.encoder.maxpool = nn.Identity()
                self._reinit_all_layers()

        else:
            self.encoder = arch

        if pretrain_state_dict is not None:
            self.encoder.load_state_dict(pretrain_state_dict)

        self.feat_dim = feat_dim

        self.head_lab = Prototypes(self.feat_dim, num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out

# --------------------------
# SINKHORN KNOPP
# --------------------------

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


# --------------------------
# AUGMENTATIONS
# --------------------------

class DiscoverTargetTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, y):
        y = self.mapping[y]
        return y


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        return ImageOps.equalize(img)


def get_multicrop_transform(dataset, mean, std):
    if dataset == "ImageNet":
        return T.Compose(
            [
                T.RandomResizedCrop(size=96, scale=(0.08, 0.5)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    elif "CIFAR" in dataset:
        return T.Compose(
            [
                T.RandomResizedCrop(size=18, scale=(0.3, 0.8)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                Solarize(p=0.2),
                Equalize(p=0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    elif dataset == 'scars':
        return T.Compose(
            [
                T.RandomResizedCrop(size=192, scale=(0.08, 0.5)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )


class MultiTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return [t(x) for t in self.transforms]


def get_transforms(mode, dataset, multicrop=False, num_large_crops=2, num_small_crops=2):

    mean, std = {
        "CIFAR10": [(0.491, 0.482, 0.447), (0.202, 0.199, 0.201)],
        "CIFAR100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
        "ImageNet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'scars': [(0, 0, 0), (0.5, 0.5, 0.5)]
    }[dataset]

    transform = {
        "ImageNet": {
            "unsupervised": T.Compose(
                [
                    T.RandomResizedCrop(224, (0.5, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "eval": T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
        },
        "scars": {
            "unsupervised": T.Compose(
                [
                    T.RandomResizedCrop(448, (0.5, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "supervised": T.Compose(
                [
                    T.RandomResizedCrop(448, (0.5, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomGrayscale(p=0.2),
                    T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "eval": T.Compose(
                [
                    T.Resize(int(448 / 0.875)),
                    T.CenterCrop(448),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
        },
        "CIFAR100": {
            "unsupervised": T.Compose(
                [
                    T.RandomChoice(
                        [
                            T.RandomCrop(32, padding=4),
                            T.RandomResizedCrop(32, (0.5, 1.0)),
                        ]
                    ),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                    Solarize(p=0.1),
                    Equalize(p=0.1),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "supervised": T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "eval": T.Compose(
                [
                    T.CenterCrop(32),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
        },
        "CIFAR10": {
            "unsupervised": T.Compose(
                [
                    T.RandomChoice(
                        [
                            T.RandomCrop(32, padding=4),
                            T.RandomResizedCrop(32, (0.5, 1.0)),
                        ]
                    ),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                    Solarize(p=0.1),
                    Equalize(p=0.1),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "supervised": T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "eval": T.Compose(
                [
                    T.CenterCrop(32),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
        },
    }[dataset][mode]

    if mode == "unsupervised":
        transforms = [transform] * num_large_crops
        if multicrop:
            multicrop_transform = get_multicrop_transform(dataset, mean, std)
            transforms += [multicrop_transform] * num_small_crops
        transform = MultiTransform(transforms)

    return transform

# --------------------------
# LR SCHEDULE
# --------------------------

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]