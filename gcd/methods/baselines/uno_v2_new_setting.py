import torch
import torch.nn.functional as F

from torch.optim import SGD
from torch.utils.data import DataLoader
from project_utils.cluster_utils import AverageMeter, seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

# Data
from data.get_datasets import get_datasets, get_class_splits

from methods.baselines.uno_v2_utils import get_transforms, SinkhornKnopp, MultiHeadModel, LinearWarmupCosineAnnealingLR

# Utils
import argparse
from tqdm import tqdm
import numpy as np
from project_utils.general_utils import init_experiment, get_mean_lr
import os

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def cross_entropy_loss(preds, targets):
    preds = F.log_softmax(preds / args.temperature, dim=-1)
    return -torch.mean(torch.sum(targets * preds, dim=-1))


def swapped_prediction(logits, targets):
    loss = 0
    for view in range(args.num_large_crops):
        for other_view in np.delete(range(args.num_crops), view):
            loss += cross_entropy_loss(logits[other_view], targets[view])
    return loss / (args.num_large_crops * (args.num_crops - 1))


def train_uno_v2(model, train_loader, test_loader, unlabelled_train_loader, args):

    optimizer = SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum_opt,
                    weight_decay=args.weight_decay_opt)

    exp_lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            warmup_start_lr=args.min_lr,
            eta_min=args.min_lr,
        )

    sk = SinkhornKnopp(num_iters=args.num_iters_sk, epsilon=args.epsilon_sk)
    best_test_acc_lab = 0

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        model.train()
        loss_per_head = torch.zeros(args.num_heads).to(device)

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            views, labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            labels, mask_lab = labels.to(device), mask_lab.to(device).bool()
            views = [x.to(device) for x in views]
            nlc = args.num_labeled_classes

            # normalize prototypes
            model.normalize_prototypes()

            # forward
            outputs = model(views)

            # gather outputs
            outputs["logits_lab"] = (
                outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1)
            )
            logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
            logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)

            # create targets
            targets_lab = (
                F.one_hot(labels[mask_lab], num_classes=nlc)
                    .float()
                    .to(device)
            )
            targets = torch.zeros_like(logits)
            targets_over = torch.zeros_like(logits_over)

            # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
            for v in range(args.num_large_crops):
                for h in range(args.num_heads):
                    targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                    targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                    targets[v, h, ~mask_lab] = sk(                  # Use all logits to generate pseudo labels instead of just unlabelled head
                        logits[v, h, ~mask_lab]
                    ).type_as(targets)
                    targets_over[v, h, ~mask_lab] = sk(             # Use all logits to generate pseudo labels instead of just unlabelled head
                        logits_over[v, h, ~mask_lab]
                    ).type_as(targets)

            # compute swapped prediction loss
            loss_cluster = swapped_prediction(logits, targets)
            loss_overcluster = swapped_prediction(logits_over, targets_over)

            # total loss
            loss = (loss_cluster + loss_overcluster) / 2

            # update best head tracker
            loss_per_head += loss_cluster.clone().detach()

            # Known preds
            _, pred = logits[0, :, mask_lab, 0].max(0)
            known_acc = (pred == labels[mask_lab]).float().mean().item()

            loss_record.update(loss.item(), labels.size(0))
            train_acc_record.update(known_acc, pred.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))

        with torch.no_grad():

            print('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test_uno_v2(model, unlabelled_train_loader, best_head=loss_per_head.argmin(),
                                                    epoch=epoch, save_name='Train ACC Unlabelled',
                                                    args=args)

            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_uno_v2(model, test_loader, best_head=loss_per_head.argmin(),
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


def test_uno_v2(model, test_loader, best_head, epoch, save_name, args,
                print_output=False, pred_save_path=None):

    model.eval()
    preds = np.array([])
    targets = np.array([])
    mask = np.array([])

    if pred_save_path is not None:
        logits_to_save = []

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(test_loader)):

            images, labels = images.to(device), labels.to(device)

            # forward
            outputs = model(images)

            logits = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )

            pred = logits.max(dim=-1)[1]

            targets = np.append(targets, labels.cpu().numpy())
            preds = np.append(preds, pred.detach().cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in labels]))
            if pred_save_path is not None:
                logits_to_save.append(logits.cpu().numpy())

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer, print_output=print_output)

    # Save preds if specified
    if pred_save_path is not None:
        print(f'Saving model predictions to {pred_save_path}')
        logits_to_save = np.concatenate(logits_to_save)
        torch.save(logits_to_save, pred_save_path)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--exp_root', type=str, default='/work/sagar/osr_novel_categories/')
    parser.add_argument('--pretrain_dir', type=str, default='vit_dino')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train', help='supervised or ncd')
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
    parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
    parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
    parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
    parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")

    # --------------------
    # INIT
    # --------------------
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)
    args.num_crops = args.num_large_crops + args.num_small_crops if args.multicrop else args.num_large_crops
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['uno_v2_gcd'], exp_id=args.exp_id)
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    args.model_dir = args.model_dir + '/' + '{}.pt'.format(args.model_name)
    args.device = device

    # --------------------
    # TRANSFORMS
    # --------------------
    dataset_ = 'ImageNet'

    train_transform = get_transforms(
        "unsupervised",
        dataset_,
        multicrop=args.multicrop,
        num_large_crops=args.num_large_crops,
        num_small_crops=args.num_small_crops,
    )

    test_transform = get_transforms("eval", dataset_)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform, args)
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
    # BASE MODEL
    # --------------------
    if args.pretrain_dir == 'vit_dino':

        pretrain_path = '/work/sagar/pretrained_models/dino/dino_vitbase16_pretrain.pth'
        base_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        base_model.load_state_dict(state_dict)

        feat_dim = 768

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

    else:

        raise NotImplementedError

    # --------------------
    # MODEL
    # --------------------
    model = MultiHeadModel(base_model, low_res=True,
                           num_labeled=args.num_labeled_classes, num_unlabeled=args.num_unlabeled_classes,
                           proj_dim=args.proj_dim, hidden_dim=args.hidden_dim,
                           overcluster_factor=args.overcluster_factor, num_heads=args.num_heads,
                           num_hidden_layers=args.num_hidden_layers, feat_dim=feat_dim,
                           pretrain_state_dict=None)

    model.to(device)

    # --------------------
    # TRAIN
    # --------------------
    if args.mode == 'train':

        train_uno_v2(model, train_loader=train_loader,
                        test_loader=test_loader_labelled,
                        unlabelled_train_loader=test_loader_unlabelled,
                        args=args)

        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))

    elif args.mode == 'test':
        with torch.no_grad():
            model.load_state_dict(torch.load(args.pretrain_path))

            args.writer = None
            args.eval_funcs = ['v1', 'v2', 'v3']
            pred_save_path = os.path.join(args.pretrain_path.split('checkpoint')[0], 'model_preds.pt')

            all, old, new = test_uno_v2(model, test_loader_unlabelled, best_head=0,
                                        epoch=None, save_name='Train ACC Unlabelled',
                                        args=args, print_output=True, pred_save_path=pred_save_path)

    else:
        raise NotImplementedError