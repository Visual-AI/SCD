import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from project_utils.cluster_utils import BCE, PairEnum, AverageMeter, seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

# Data
from data.get_datasets import get_datasets, get_class_splits
from methods.baselines.auto_novel_augs import get_aug
from methods.baselines import autonovel_ramps as ramps

# Utils
import argparse
from tqdm import tqdm
import numpy as np
from methods.baselines.auto_novel_utils import ViTDINOWrapper
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_rankstats(model, train_loader, test_loader, unlabelled_train_loader, args):

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    best_test_acc_lab = 0

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

            mask_lb = mask_lb.to(device)[:, 0].bool()

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

            # TODO: Make sure the following is true
            # target_ulb is of shape B * B, assume this is a viewed version of a B * B correspondence matrix
            labelled_set_labels = label[mask_lb]
            labelled_set_correspondences = (labelled_set_labels == labelled_set_labels[:, None])   # B x B
            labelled_set_correspondences = labelled_set_correspondences.float()
            labelled_set_correspondences[labelled_set_correspondences == 0] = -1

            # Manually set labelled set targets to true or false
            target_ulb = target_ulb.view(x.size(0), x.size(0))
            target_ulb[mask_lb][:, mask_lb] = labelled_set_correspondences
            target_ulb = target_ulb.view(-1)        # B * B

            prob1_ulb, _ = PairEnum(prob1)
            _, prob2_ulb = PairEnum(prob1_bar)

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob1, prob1_bar)
            kl_div_loss = F.kl_div(prob1, torch.ones_like(prob1) / prob1.size(1))
            loss = loss_bce + \
                   w * consistency_loss + \
                   args.ce_loss * loss_ce + args.kl_div_loss * kl_div_loss

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
            all_acc, old_acc, new_acc = test(model, unlabelled_train_loader,
                                                    epoch=epoch, save_name='Train ACC Unlabelled',
                                                    args=args)

            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test(model, test_loader,
                                                                   epoch=epoch, save_name='Test ACC',
                                                                   args=args)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                              new_acc))
        print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                             new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))

        if old_acc_test > best_test_acc_lab:

            print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
            print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                       new_acc))

            torch.save(model.state_dict(), args.model_dir[:-3] + f'_best.pt')
            print("model saved to {}.".format(args.model_dir[:-3] + f'_best.pt'))

            best_test_acc_lab = old_acc_test


def test(model, test_loader, epoch, save_name, args, print=False):

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
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer, print_output=print)

    return all_acc, old_acc, new_acc


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
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--kl_div_loss', type=float, default=0.0)
    parser.add_argument('--ce_loss', type=float, default=0.0)

    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--exp_root', type=str, default='/work/sagar/osr_novel_categories/')
    parser.add_argument('--warmup_model_dir', type=str,
                        default='/scratch/shared/beegfs/khan/AutoNovel/pretrained/supervised_learning/resnet_rotnet_cifar100.pth')
    parser.add_argument('--save_best_epoch', default=None, type=int)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model', type=str, default='vit_dino')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--exp_id', type=str, default=None)

    parser.add_argument('--freeze_backbone', type=str2bool, default=True)
    parser.add_argument('--grad_from_block', type=int, default=11)

    # --------------------
    # INIT
    # --------------------
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)
    args = get_class_splits(args)

    # If using DINO frozen weights, then use the Imagenet transform regardless of dataset
    args.image_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['autonovel_gcd'], exp_id=args.exp_id)
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    args.model_dir = args.model_dir + '/' + '{}.pt'.format(args.model)
    args.device = device

    # --------------------
    # TRANSFORMS
    # --------------------
    train_transform = get_aug('twice', image_size=args.image_size, mean=mean, std=std)
    test_transform = get_aug(None, image_size=args.image_size, mean=mean, std=std)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform, test_transform, args)

    # --------------------
    # MODEL
    # --------------------
    if args.model == 'vit_dino':

        pretrain_path = '/work/sagar/pretrained_models/dino/dino_vitbase16_pretrain.pth'
        base_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        base_model.load_state_dict(state_dict)

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in base_model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in base_model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

        model = ViTDINOWrapper(base_model=base_model, num_labelled_classes=len(args.train_classes),
                               num_unlabelled_classes=len(args.unlabeled_classes)).to(device)

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
    # TRAIN
    # --------------------
    if args.mode == 'train':

        train_rankstats(model, train_loader=train_loader,
                        test_loader=test_loader_labelled,
                        unlabelled_train_loader=test_loader_unlabelled,
                        args=args)

        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))

    elif args.mode == 'test':

        args.warmup_model_dir = '/work/sagar/osr_novel_categories/autonovel_gcd/log/(29.01.2022_|_00.331)/' \
                                'checkpoints/model_best.pt'
        print(f'Note: Manually loading from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir))

        args.writer = None
        args.eval_funcs = ['v1', 'v2', 'v3']

        test(model, test_loader_unlabelled, None, 'Train ACC Unlabelled', args, print=False)