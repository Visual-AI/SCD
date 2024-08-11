import torch
import torch.nn.functional as F

from torch.optim import SGD
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score as ari_score
from project_utils.cluster_utils import cluster_acc, AverageMeter, seed_torch
from copy import deepcopy

# Data
from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.stanford_cars import get_scars_datasets
from data.data_utils import MergedDataset
from methods.baselines.uno_v2_utils import get_transforms, SinkhornKnopp, MultiHeadModel, LinearWarmupCosineAnnealingLR

from data.cifar import subsample_classes as subsample_classes_cifar
from data.stanford_cars import subsample_classes as subsample_classes_scars

# Utils
import argparse
from tqdm import tqdm
import numpy as np
from project_utils.general_utils import init_experiment, get_mean_lr, transform_moco_state_dict, accuracy
from data import get_class_splits

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

PRETRAIN_PATHS = {
    'cifar10': {
        0.5: '/work/sagar/osr_novel_categories/uno_v2/log/(18.10.2021_|_11.230)/checkpoints/resnet18.pth',
        1.0: '/work/sagar/osr_novel_categories/uno_v2/log/(18.10.2021_|_07.268)/checkpoints/resnet18.pth'
    },
    'cifar100': {
        0.5: '/work/sagar/osr_novel_categories/uno_v2/log/(18.10.2021_|_34.740)/checkpoints/resnet18.pth',
        1.0: '/work/sagar/osr_novel_categories/uno_v2/log/(18.10.2021_|_07.295)/checkpoints/resnet18.pth'
    },
    'scars': {
        0.5: '/work/sagar/osr_novel_categories/uno_v2/log/(18.10.2021_|_04.903)/checkpoints/resnet18.pth',
        1.0: '/work/sagar/osr_novel_categories/uno_v2/log/(18.10.2021_|_06.076)/checkpoints/resnet18.pth'
    }
}


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
                    targets[v, h, ~mask_lab, nlc:] = sk(
                        outputs["logits_unlab"][v, h, ~mask_lab]
                    ).type_as(targets)
                    targets_over[v, h, ~mask_lab, nlc:] = sk(
                        outputs["logits_unlab_over"][v, h, ~mask_lab]
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

            if args.prop_train_labels == 1.0:

                print('Testing on unlabelled examples in the training data...')
                (unlabelled_acc, unlabelled_nmi, unlabelled_ari) =\
                    test_uno_v2(model, unlabelled_train_loader, best_head=loss_per_head.argmin(), args=args)
                print('Testing on disjoint test set...')
                (labelled_acc_test, labelled_nmi_test, labelled_ari_test), (unlabelled_acc_test, unlabelled_nmi_test, unlabelled_ari_test) = \
                    test_uno_v2(model, test_loader, best_head=loss_per_head.argmin(), args=args)

                # ----------------
                # LOG
                # ----------------

                args.writer.add_scalars('Train ACC Unlabelled Data', {'Labelled': 0, 'Unlabelled': unlabelled_acc,
                                                     'Mean': 0.5 * (unlabelled_acc + 0)},
                                                     epoch)


            else:

                print('Testing on unlabelled examples in the training data...')
                (labelled_acc, labelled_nmi, labelled_ari), (unlabelled_acc, unlabelled_nmi, unlabelled_ari) = \
                    test_uno_v2(model, unlabelled_train_loader, best_head=loss_per_head.argmin(), args=args)
                print('Testing on disjoint test set...')
                (labelled_acc_test, labelled_nmi_test, labelled_ari_test), (
                unlabelled_acc_test, unlabelled_nmi_test, unlabelled_ari_test) = \
                    test_uno_v2(model, test_loader, best_head=loss_per_head.argmin(), args=args)

                # ----------------
                # LOG
                # ----------------

                args.writer.add_scalars('Train ACC Unlabelled Data',
                                        {'Labelled': labelled_acc, 'Unlabelled': unlabelled_acc,
                                         'Mean': 0.5 * (unlabelled_acc + labelled_acc)},
                                        epoch)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)

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

        # Step schedule
        exp_lr_scheduler.step()


def test_uno_v2(model, test_loader, best_head, args):

    model.eval()
    preds = np.array([])
    targets = np.array([])
    mask = np.array([])


    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(test_loader)):

            images, labels = images.to(device), labels.to(device)

            # forward
            outputs = model(images)

            pred = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )

            pred = pred.max(dim=-1)[1]

            targets = np.append(targets, labels.cpu().numpy())
            preds = np.append(preds, pred.detach().cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in labels]))

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


def train_supervised(model, train_loader, test_loader, args):

    optimizer = SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum_opt,
                    weight_decay=args.weight_decay_opt)
    exp_lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            warmup_start_lr=args.min_lr,
            eta_min=args.min_lr,
        )

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, labels, uq_idxs = batch
            images, labels = images.to(device), labels.to(device)

            # normalize prototypes
            model.normalize_prototypes()

            # forward
            outputs = model(images)

            # supervised loss
            # loss = torch.stack(
            #     [F.cross_entropy(o / args.temperature, labels) for o in outputs["logits_lab"]]
            # ).mean()
            loss = F.cross_entropy(outputs["logits_lab"] / args.temperature, labels)

            # Known preds
            _, pred = outputs["logits_lab"].max(dim=-1)
            known_acc = (pred == labels).float().mean().item()

            loss_record.update(loss.item(), labels.size(0))
            train_acc_record.update(known_acc, pred.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))

        with torch.no_grad():
            print('Testing on labelled test set...')
            test_acc = test_supervised(model, test_loader)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss Pretrain', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Pretrain', train_acc_record.avg, epoch)
        args.writer.add_scalar('Test Acc Pertrain', test_acc, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        # Step schedule
        exp_lr_scheduler.step()


def test_supervised(model, test_loader):

    model.eval()
    accuracy_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(test_loader)):

            images, labels = images.to(device), labels.to(device)

            # forward
            logits = model(images)["logits_lab"]
            _, pred = logits.max(dim=-1)

            accuracy_meter.update(accuracy(logits, labels, (1,))[0], n=len(labels))

    print(f'Test Acc on Labelled Set: {accuracy_meter.avg.item():.4f}')

    return accuracy_meter.avg

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
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='supervised', help='supervised or ncd')
    parser.add_argument('--exp_id', type=str, default=None)

    parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
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

    init_experiment(args, runner_name=['uno_v2'], exp_id=args.exp_id)

    args.model_dir = args.model_dir + '/' + '{}.pth'.format(args.model_name)
    args.device = device

    # --------------------
    # TRANSFORMS
    # --------------------
    if args.dataset_name == 'cifar10':
        dataset_ = 'CIFAR10'
    elif args.dataset_name == 'cifar100':
        dataset_ = 'CIFAR100'
    elif args.dataset_name == 'scars':
        dataset_ = 'scars'
    else:
        raise NotImplementedError

    if args.mode == 'ncd':
        train_transform = get_transforms(
            "unsupervised",
            dataset_,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )

    elif args.mode == 'supervised':
        train_transform = get_transforms(
            "supervised",
            dataset_,
            multicrop=False
        )

    else:

        raise NotImplementedError

    test_transform = get_transforms("eval", dataset_)

    # --------------------
    # DATASETS
    # --------------------
    if 'cifar' in args.dataset_name:


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
        state_dict = torch.load('/work/sagar/pretrained_models/imagenet/moco_v2_800ep_pretrain.pth.tar')['state_dict']
        state_dict = transform_moco_state_dict(state_dict, num_classes=1000)
        model = MultiHeadModel('resnet50', low_res=False,
                               num_labeled=args.num_labeled_classes, num_unlabeled=args.num_unlabeled_classes,
                               proj_dim=args.proj_dim, hidden_dim=args.hidden_dim,
                               overcluster_factor=args.overcluster_factor, num_heads=args.num_heads,
                               num_hidden_layers=args.num_hidden_layers)

    else:

        model = MultiHeadModel(args.model_name, low_res=True,
                               num_labeled=args.num_labeled_classes, num_unlabeled=args.num_unlabeled_classes,
                               proj_dim=args.proj_dim, hidden_dim=args.hidden_dim,
                               overcluster_factor=args.overcluster_factor, num_heads=args.num_heads,
                               num_hidden_layers=args.num_hidden_layers)

    model.to(device)

    # --------------------
    # TRAIN
    # --------------------
    if args.mode == 'ncd':

        if args.pretrain_dir == 'fixed_paths':
            args.pretrain_dir = PRETRAIN_PATHS[args.dataset_name][args.prop_train_labels]
            print(f'Using pretrained weights from {args.pretrain_dir}')
            model.load_state_dict(torch.load(args.pretrain_dir))

        if args.pretrain_dir is not None:
            print(f'Using pretrained weights from {args.pretrain_dir}')
            model.load_state_dict(torch.load(args.pretrain_dir))

        else:

            print('Training for NCD from scratch')

        train_uno_v2(model, train_loader=train_loader,
                        test_loader=test_loader_labelled,
                        unlabelled_train_loader=test_loader_unlabelled,
                        args=args)

        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))

    elif args.mode == 'supervised':

        subsample_classes = subsample_classes_scars if args.dataset_name == 'scars' else subsample_classes_cifar

        train_loader = DataLoader(datasets['train_labelled'], num_workers=args.num_workers, batch_size=args.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(subsample_classes(deepcopy(datasets['test']), include_classes=args.train_classes),
                                 num_workers=args.num_workers, batch_size=args.batch_size,
                                  shuffle=False)

        train_supervised(model, train_loader=train_loader, test_loader=test_loader, args=args)

        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))

    else:

        raise NotImplementedError