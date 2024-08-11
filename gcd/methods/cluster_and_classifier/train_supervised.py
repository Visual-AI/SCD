
from copy import deepcopy

import timm
from torch.nn import CrossEntropyLoss
from project_utils.general_utils import init_experiment, seed_torch, AverageMeter, accuracy, str2bool
from project_utils.schedulers import get_scheduler
from data.augmentations import get_transform

import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits, sub_sample_class_funcs

from tqdm import tqdm

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from methods.baselines.uno_v2_utils import get_transforms as get_transforms_uno
from methods.baselines.uno_v2_utils import LinearWarmupCosineAnnealingLR

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


def opt_threshold_acc(y_true, y_pred):
    A = list(zip(y_true, y_pred))
    A = sorted(A, key=lambda x: x[1])
    total = len(A)
    tp = len([1 for x in A if x[0]==1])
    tn = 0
    th_acc = []
    for x in A:
        th = x[1]
        if x[0] == 1:
            tp -= 1
        else:
            tn += 1
        acc = (tp + tn) / total
        th_acc.append((th, acc))
    return max(th_acc, key=lambda x: x[1])


def test_osr(model, train_loader):

    model.eval()
    accuracy_meter = AverageMeter()

    model.eval()

    all_uq_idxs = np.array([])
    all_osr_scores = np.array([])
    all_preds = np.array([])
    targets = np.array([])
    mask = np.array([])

    # First extract all features
    for batch_idx, (images, label, uq_idxs) in enumerate(tqdm(train_loader)):

        images = images.to(args.device)
        label = label.to(args.device)

        # forward
        logits = model(images)
        osr_score, pred = logits.max(dim=-1)
        osr_score *= -1

        all_osr_scores = np.append(all_osr_scores, osr_score.cpu().numpy())
        all_uq_idxs = np.append(all_uq_idxs, uq_idxs.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        all_preds = np.append(all_preds, pred.cpu().numpy())
        batch_mask = np.array([True if x.item() in range(len(args.train_classes))
                               else False for x in label])
        mask = np.append(mask, batch_mask)

        logits_lab, label_lab = logits[batch_mask], label[batch_mask]
        accuracy_meter.update(accuracy(logits_lab, label_lab, (1,))[0], n=len(label_lab))

    print(f'AUROC on open-set decision: {roc_auc_score(1 - mask, all_osr_scores):.4f}')
    print(f'Test Acc on Labelled Set: {accuracy_meter.avg.item():.4f}')

    all_osr_scores -= all_osr_scores.min()
    all_osr_scores /= all_osr_scores.max()

    print(f'Threshold for optimal acc: {opt_threshold_acc(1 - mask, all_osr_scores)}')

    return all_osr_scores, all_uq_idxs, all_preds


def get_optimizer(params, args):

    optimizer = SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    return optimizer


def train(model, train_loader, val_loader, unlabelled_train_loader):

    # Optimizer
    params = model.parameters()
    optimizer = get_optimizer(params, args)
    scheduler = get_scheduler(optimizer, args)
    # scheduler = LinearWarmupCosineAnnealingLR(
    #     optimizer,
    #     warmup_epochs=10,
    #     max_epochs=args.epochs,
    #     warmup_start_lr=args.lr / 100,
    #     eta_min=args.lr / 100,
    # )

    best_acc = 0

    if args.label_smoothing > 0:
        loss_function = LabelSmoothingLoss(smoothing=args.label_smoothing)
    else:
        loss_function = CrossEntropyLoss(reduction='mean')

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        acc_record_top1 = AverageMeter()
        acc_record_top5 = AverageMeter()

        model.train()
        torch.cuda.empty_cache()
        # ----------------
        # TRAIN EPOCH
        # ----------------
        for batch_idx, batch in enumerate(tqdm(train_loader)):

            imgs, labels, uq_idxs = [x.to(args.device) for x in batch]

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                preds = model(imgs)
                loss = loss_function(preds / args.temperature, labels)
                acc = accuracy(preds, labels, topk=(1, 5, 10))

                loss.backward()
                optimizer.step()

            # Update loss meters
            loss_record.update(loss.item(), imgs.size(0))
            acc_record_top1.update(acc[0].item(), imgs.size(0))
            acc_record_top5.update(acc[1].item(), imgs.size(0))

        # --------------------------
        # LOG
        # --------------------------
        print('Epoch: {} | Train Loss: {:.3f} | Train Top1: {:.3f}'.format(epoch, loss_record.avg, acc_record_top1.avg))

        # ----------------
        # EVAL EPOCH
        # ----------------
        with torch.no_grad():
            top1_val_acc, top5_val_acc, top10_val_acc = test(model, val_loader, topk=(1, 5, 10))

        # if epoch % 10 == 0:
        #     with torch.no_grad():
        #         test_osr(model, unlabelled_train_loader)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalars('Train Acc', {'Top 1': acc_record_top1.avg, 'Top 5': acc_record_top5.avg}, epoch)
        args.writer.add_scalar('Test Acc Top 1', top1_val_acc, epoch)
        args.writer.add_scalars('Test Acc', {'Top 1': acc_record_top1.avg, 'Top 5': acc_record_top5.avg}, epoch)
        args.writer.add_scalar('Learning rate', get_mean_lr(optimizer), epoch)

        # ----------------
        # SAVE
        # ----------------
        if top1_val_acc > best_acc:
            best_acc = np.copy(top1_val_acc)

        # Save every epoch if we don't have a validation set
        print('Saving model to {}'.format(args.model_dir))
        torch.save(model.state_dict(), args.model_dir + f'/vit_dino_linear.pt')

        print('Best val accuracy: {:.3f}'.format(best_acc))

        # ----------------
        # STEP SCHEDULE
        # ----------------
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(loss_record.avg, epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)


def test(model, val_loader, topk=(1, 2)):

    model.eval()

    acc_records = {}
    for k in topk:
        acc_records[k] = AverageMeter()
    torch.cuda.empty_cache()
    # Val epoch
    for batch_idx, batch in enumerate(tqdm(val_loader)):

        imgs, labels, uq_idxs = [x.to(args.device) for x in batch]
        with torch.set_grad_enabled(False):
            preds = model(imgs)
            # Update acc meters
            acc = accuracy(preds, labels, topk=topk)
            for idx, k in enumerate(topk):
                acc_records[k].update(acc[idx].item(), imgs.size(0))

    strings = ['Top{} {:.4f}'.format(k, v.avg) for k, v in sorted(acc_records.items())]
    acc_string = " ".join(strings)

    print_string = "Test Facing Level Avg Acc: " + acc_string

    print(print_string)

    return acc_records[topk[0]].avg, acc_records[topk[1]].avg, acc_records[topk[2]].avg



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default='/work/sagar/osr_novel_categories/extracted_features')
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_custom_fgvc_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default='/work/sagar/osr_novel_categories/')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--rand_aug_m', type=int, default=2)
    parser.add_argument('--rand_aug_n', type=int, default=30)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=True)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
    device = torch.device('cuda:0')
    seed_torch(args.seed)

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['supervised_train_rebuttal'])
    args.device = device

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/work/sagar/pretrained_models/dino/dino_vitbase16_pretrain.pth'

        # model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        model = vits.__dict__['vit_base']()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:

        raise NotImplementedError

    # --------------------
    # MODEL WITH LINEAR LAYER
    # --------------------
    # model = timm.create_model('resnet18', pretrained=True, num_classes=len(args.train_classes))
    model = vits.VisionTransformerWithLinear(base_vit=model, num_classes=len(args.train_classes))
    model.to(device)

    # --------------------
    # DATASETS
    # --------------------
    # train_transform = get_transforms_uno(
    #     "unsupervised",
    #     'ImageNet',
    #     multicrop=False,
    #     num_large_crops=1,
    #     num_small_crops=1,
    # ).transforms[0]
    #
    # test_transform = get_transforms_uno("eval", 'ImageNet')
    train_transform, test_transform = get_transform(args.transform, image_size=224, args=args)
    subsample_classes = sub_sample_class_funcs[args.dataset_name]

    _, _, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    test_dataset = subsample_classes(deepcopy(datasets['test']), include_classes=args.train_classes)

    train_loader = DataLoader(datasets['train_labelled'], num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             num_workers=args.num_workers, batch_size=args.batch_size,
                             shuffle=False)
    unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=False)

    train(model, train_loader, test_loader, unlabelled_train_loader)