import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, lr_scheduler
import timm
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score as ari_score
from project_utils.cluster_utils import BCE, PairEnum, cluster_acc, AverageMeter, seed_torch
from methods.baselines import autonovel_ramps
from copy import deepcopy
from models.resnet_twohead import ResNet, BasicBlock

# Data
from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.stanford_cars import get_scars_datasets
from data.data_utils import MergedDataset
from methods.baselines.auto_novel_augs import get_aug

# Utils
import argparse
from tqdm import tqdm
import numpy as np
from methods.baselines.auto_novel_utils import transform_moco_state_dict, ResNet50Wrapper
from project_utils.general_utils import init_experiment, get_mean_lr
from data import get_class_splits

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def train_rankstats(model, train_loader, test_loader, unlabelled_train_loader, args):

    # TODO: Add this as an option in the __main__ rather than here
    # Make sure the transforms are correct
    # train_loader.dataset.labelled_dataset.transform = get_aug('twice', image_size=args.image_size)
    # train_loader.dataset.unlabelled_dataset.transform = get_aug('twice', image_size=args.image_size)
    # test_loader.dataset.transform = get_aug(None, image_size=args.image_size)
    # unlabelled_train_loader.dataset.transform = get_aug(None, image_size=args.image_size)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

        for batch_idx, ((x, x_bar),  label, idx, mask_lb) in enumerate(tqdm(train_loader)):

            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            output1, _, feat = model(x)
            output1_bar, _, _ = model(x_bar)
            prob1, prob1_bar = F.softmax(output1, dim=1),  F.softmax(output1_bar, dim=1)

            mask_lb = mask_lb.to(device)[:, 0]

            rank_feat = feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob1)
            _, prob2_ulb = PairEnum(prob1_bar)

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob1, prob1_bar)
            kl_div_loss = F.kl_div(prob1, torch.ones_like(prob1) / prob1.size(1))
            loss = loss_ce + loss_bce + w * consistency_loss + args.kl_div_loss * kl_div_loss

            # Known preds
            _, pred = output1[mask_lb].max(1)
            known_acc = (pred == label[mask_lb]).float().mean().item()
            train_acc_record.update(known_acc, pred.size(0))

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))

        with torch.no_grad():
            print('Testing on unlabelled examples in the training data...')
            (labelled_acc, labelled_nmi, labelled_ari), (unlabelled_acc, unlabelled_nmi, unlabelled_ari) =\
                test(model, unlabelled_train_loader, args)
            print('Testing on disjoint test set...')
            (labelled_acc_test, labelled_nmi_test, labelled_ari_test), (unlabelled_acc_test, unlabelled_nmi_test, unlabelled_ari_test) = \
                test(model, test_loader, args)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalars('Train ACC Unlabelled Data', {'Labelled': labelled_acc, 'Unlabelled': unlabelled_acc,
                                             'Mean': 0.5 * (unlabelled_acc + labelled_acc)},
                                             epoch)
        args.writer.add_scalars('Eval NMI', {'Labelled': labelled_nmi_test, 'Unlabelled': unlabelled_nmi_test,
                                             'Mean': 0.5 * (labelled_nmi_test + unlabelled_nmi_test)},
                                            epoch)
        args.writer.add_scalars('Eval ACC', {'Labelled': labelled_acc_test, 'Unlabelled': unlabelled_acc_test,
                                             'Mean': 0.5 * (unlabelled_acc_test + labelled_acc_test)},
                                            epoch)
        args.writer.add_scalars('Eval NMI', {'Labelled': labelled_nmi_test, 'Unlabelled': unlabelled_nmi_test,
                                             'Mean': 0.5 * (labelled_nmi_test + unlabelled_nmi_test)},
                                            epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)


def test(model, test_loader, args):

    model.eval()
    preds = np.array([])
    targets = np.array([])
    mask = np.array([])

    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):

            x, label = x.to(device), label.to(device)
            output, _, _ = model(x)
            _, pred = output.max(1)

            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.detach().cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in label]))

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask.astype(bool)

    # Labelled examples
    if mask.sum() == 0:     # All examples come from unlabelled classes

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int), preds.astype(int)),\
                                                         nmi_score(targets, preds),\
                                                         ari_score(targets, preds)

        print('Unlabelled Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'
              .format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

        return (unlabelled_acc, unlabelled_nmi, unlabelled_ari)

    else:

        labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]),\
                                                   nmi_score(targets[mask], preds[mask]), \
                                                   ari_score(targets[mask], preds[mask])

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                     preds.astype(int)[~mask]), \
                                                   nmi_score(targets[~mask], preds[~mask]), \
                                                   ari_score(targets[~mask], preds[~mask])

        print('Laballed Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi,
                                                                                labelled_ari))
        print('Unlabelled Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi,
                                                                                  unlabelled_ari))

        return (labelled_acc, labelled_nmi, labelled_ari), (unlabelled_acc, unlabelled_nmi, unlabelled_ari)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--kl_div_loss', type=float, default=0.5)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--exp_root', type=str, default='/work/sagar/osr_novel_categories/')
    parser.add_argument('--warmup_model_dir', type=str, default='/scratch/shared/beegfs/khan/AutoNovel/pretrained/supervised_learning/resnet_rotnet_cifar10.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, scars')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--exp_id', type=str, default=None)

    # --------------------
    # INIT
    # --------------------
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    if args.dataset_name == 'cifar10':
        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':
        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 448
        args.train_classes, args.unlabeled_classes = get_class_splits('scars')

    else:

        raise NotImplementedError

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['rerun_autonovel'], exp_id=args.exp_id)

    args.model_dir = args.model_dir + '/' + '{}.pth'.format(args.model_name)
    args.device = device

    # --------------------
    # DATASETS
    # --------------------
    if 'cifar' in args.dataset_name:

        train_transform = get_aug('twice', image_size=args.image_size)
        test_transform = get_aug(None, image_size=args.image_size)

        if args.dataset_name == 'cifar10':

            datasets = get_cifar_10_datasets(train_transform=train_transform, test_transform=test_transform,
                                            train_classes=args.train_classes,
                                             prop_train_labels=args.prop_train_labels,
                                             split_train_val=False)

        elif args.dataset_name == 'cifar100':

            datasets = get_cifar_100_datasets(train_transform=train_transform, test_transform=test_transform,
                                             train_classes=args.train_classes,
                                              prop_train_labels=args.prop_train_labels,
                                              split_train_val=False)

        else:

            raise NotImplementedError

        # Train split, labelled and unlabelled classes, for training
        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

        test_dataset = datasets['test']
        unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
        unlabelled_train_examples_test.transform = test_transform

    elif args.dataset_name == 'scars':

        train_transform = get_aug('twice', image_size=args.image_size)
        test_transform = get_aug(None, image_size=args.image_size)

        datasets = get_scars_datasets(train_transform=train_transform, test_transform=test_transform,
                                      train_classes=args.train_classes, prop_train_labels=args.prop_train_labels)

        # Train split, labelled and unlabelled classes, for training
        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

        test_dataset = datasets['test']
        unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
        unlabelled_train_examples_test.transform = test_transform

    else:

        raise NotImplementedError

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    # --------------------
    # MODEL
    # --------------------

    if args.dataset_name == 'scars':

        # Get R50 backbone
        print('Using ResNet50 model pretrained with MoCov2 on ImageNet')
        model = timm.create_model('resnet50', num_classes=len(args.train_classes), pretrained=True)
        state_dict = torch.load('/work/sagar/pretrained_models/imagenet/moco_v2_800ep_pretrain.pth.tar')['state_dict']
        state_dict = transform_moco_state_dict(state_dict, num_classes=len(args.train_classes))
        model.load_state_dict(state_dict)

        # Load TwoHead ResNet
        model = ResNet50Wrapper(base_model=model, num_labelled_classes=len(args.train_classes),
                                num_unlabelled_classes=len(args.unlabeled_classes)).to(device)


    else:

        model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes + args.num_unlabeled_classes,
                       args.num_unlabeled_classes).to(device)
        state_dict = torch.load(args.warmup_model_dir, map_location=device)

        state_dict['head1.weight'] = torch.randn((args.num_labeled_classes + args.num_unlabeled_classes, 512))
        state_dict['head1.bias'] = torch.randn((args.num_labeled_classes + args.num_unlabeled_classes,))

        model.load_state_dict(state_dict, strict=False)

    # --------------------
    # FREEZE FIRST FEW BLOCKS OF NETWORK
    # --------------------
    if args.mode == 'train':
        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False

    # --------------------
    # TRAIN
    # --------------------
    if args.mode == 'train':

        train_rankstats(model, train_loader=train_loader,
                        test_loader=test_loader_labelled,
                        unlabelled_train_loader=test_loader_unlabelled,
                        args=args)

        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))