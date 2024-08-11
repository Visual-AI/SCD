import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_utils import cluster_acc, str2bool, mixed_eval, AverageMeter, accuracy, evaluate_clustering

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from methods.clustering.faster_mix_k_means_pytorch import pairwise_distance
from data.get_datasets import get_datasets, get_class_splits
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from methods.clustering.faster_mix_k_means_pytorch import pairwise_distance

from tqdm import tqdm
from copy import deepcopy
from models.vision_transformer import VisionTransformerWithLinear
from methods.baselines.uno_v2_utils import MultiHeadModel
from methods.baselines.uno_v2_utils import get_transforms as get_transforms_uno

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pretrain_models_and_exp_ids = {
    'herbarium_19': {
        'supervised_path': '/work/sagar/osr_novel_categories/supervised_train_rebuttal/log/(29.01.2022_|_05.038)/checkpoints/vit_dino_linear.pt',
        'feat_id': '(29.12.2021_|_48.847)',
    },
    'scars': {
        # 'supervised_path': '/work/sagar/osr_novel_categories/supervised_train_rebuttal/log/(29.01.2022_|_16.002)/checkpoints/vit_dino_linear.pt',
        # 'supervised_path': '/work/sagar/osr_novel_categories/uno_v2_gcd/log/(25.01.2022_|_01.953)/checkpoints/vit_dino.pth',
        'supervised_path': '/work/sagar/osr_novel_categories/supervised_train_rebuttal/log/(29.01.2022_|_56.076)/checkpoints/vit_dino_linear.pt',
        'feat_id': '(29.12.2021_|_48.744)',
    },
    'cub': {
        'supervised_path': '/work/sagar/osr_novel_categories/supervised_train_rebuttal/log/(29.01.2022_|_24.407)/checkpoints/vit_dino_linear.pt',
        'feat_id': '(29.12.2021_|_48.696)',
    }
}


def test_kmeans_semi_sup(merge_test_loader, args, K=None):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (feats, label, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):

        feats = feats.to(device)

        # feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets

    print('Fitting Semi-Supervised K-Means...')
    kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                           n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    u_targets = u_targets.cpu().numpy()

    # -----------------------
    # EVALUATE
    # -----------------------
    # Get preds corresponding to unlabelled set
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    # -----------------------
    # EVALUATE
    # -----------------------
    (labelled_acc, labelled_nmi, labelled_ari), (
        unlabelled_acc, unlabelled_nmi, unlabelled_ari), weight = mixed_eval(targets=u_targets, preds=preds, mask=mask)

    print('Old Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi,
                                                                       labelled_ari))
    print('New Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi,
                                                                       unlabelled_ari))

    # Also return ratio between labelled and unlabelled examples
    return (labelled_acc, labelled_nmi, labelled_ari), (unlabelled_acc, unlabelled_nmi, unlabelled_ari), weight, kmeans


def combine_preds(model, feat_loader, image_loader, args):

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (feats, label, _, mask_lab_) in enumerate(tqdm(feat_loader)):

        feats = feats.to(device)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                 else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------

    # Preprocessing
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets
    print('Computing Raw SS-KMeans Acc...')
    u_feats = torch.from_numpy(u_feats).to(args.device)

    # Pairwise distances
    dist = pairwise_distance(u_feats, CLUSTER_CENTERS, args.pairwise_batch_size)
    u_mindist, preds = torch.min(dist, dim=1)
    preds = preds.cpu().numpy()

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    labelled_acc, ind_o, w_o = cluster_acc(u_targets.astype(int)[mask], preds.astype(int)[mask], return_ind=True)
    unlabelled_acc, ind_l, w_l = cluster_acc(u_targets.astype(int)[~mask], preds.astype(int)[~mask], return_ind=True)
    weight = mask.mean()

    mean_acc_train = (1 - weight) * unlabelled_acc + weight * labelled_acc
    print('Train Accuracies: Mean {:.4f} | Lab {:.4f} | Unlab {:.4f}'.format(mean_acc_train, labelled_acc, unlabelled_acc))
    print('All Classes Test acc {:.4f}'.format(evaluate_clustering(u_targets.astype(int), preds.astype(int))[0]))

    # -----------------------
    # GET CLASSIFIER PREDS
    # -----------------------
    model.eval()
    accuracy_meter = AverageMeter()
    all_logits = []
    with torch.no_grad():
        for batch_idx, (images, labels, _, mask_lab_) in enumerate(tqdm(image_loader)):

            images, labels = images.to(device), labels.to(device)
            not_mask_lab = ~mask_lab_.bool()[:, 0]
            mask_cls = torch.Tensor([True if x.item() in range(len(args.train_classes)) else False for x in labels]).bool()
            mask_cls_not_in_lab_set = mask_cls[not_mask_lab]

            # forward
            logits = model(images)              #["logits_lab"]
            _, pred = logits.max(dim=-1)

            # if not_mask_lab.float().sum() > 0:
            #     accuracy_meter.update(accuracy(logits[not_mask_lab][mask_cls_not_in_lab_set],
            #                                    labels[not_mask_lab][mask_cls_not_in_lab_set],
            #                                    (1,))[0], n=len(labels))

            logits = logits.cpu().numpy()
            all_logits.append(logits)

    all_logits = np.concatenate(all_logits)
    # print(f'Test Acc on Labelled Set: {accuracy_meter.avg.item():.4f}')

    # -----------------------
    # Adjust some preds with predictions from base classifier
    # -----------------------
    adjusted_preds = deepcopy(preds)
    adjusted_dists = dist.clone()
    classifier_softmax = torch.nn.Softmax(dim=-1)(torch.from_numpy(all_logits))

    # -----------------------
    # Adjust scheme 2
    # -----------------------
    mask_preds = torch.Tensor([True if x in range(len(args.train_classes)) else False for x in preds]).bool()
    for l in (np.arange(11) / 10):

        print(l)
        preds_to_adjust = adjusted_dists[mask_preds]
        preds_to_adjust = torch.nn.Softmax(dim=-1)(-preds_to_adjust[:, :len(args.train_classes)])
        preds_from_classifier = classifier_softmax[~mask_lab][mask_preds]

        interpolated_preds = l * preds_from_classifier + (1 - l) * preds_to_adjust

        adjusted_preds[mask_preds] = interpolated_preds.argmax(dim=-1).cpu().numpy()

        labelled_acc, ind_o, w_o = cluster_acc(u_targets.astype(int)[mask], adjusted_preds.astype(int)[mask], return_ind=True)
        unlabelled_acc, ind_l, w_l = cluster_acc(u_targets.astype(int)[~mask], adjusted_preds.astype(int)[~mask], return_ind=True)
        weight = mask.mean()

        mean_acc_train = (1 - weight) * unlabelled_acc + weight * labelled_acc
        print('Train Accuracies: Mean {:.4f} | Lab {:.4f} | Unlab {:.4f}'.format(mean_acc_train, labelled_acc, unlabelled_acc))
        print('All Classes Test acc {:.4f}'.format(cluster_acc(u_targets.astype(int), adjusted_preds.astype(int))))

    # -----------------------
    # Adjust scheme 4
    # -----------------------
    # mask_preds = torch.Tensor([True if x in range(len(args.train_classes)) else False for x in preds]).bool()
    #
    # # Get quantiles
    # quantile_list = torch.arange(1, 10)/ 10
    # dist_quantiles = torch.quantile(torch.matmul(CLUSTER_CENTERS, CLUSTER_CENTERS.t()).flatten(), quantile_list.to(device))
    #
    # # Get dists
    # preds_to_adjust = adjusted_dists[mask_preds]
    # preds_to_adjust = torch.nn.Softmax(dim=-1)(-preds_to_adjust[:, :len(args.train_classes)])
    # preds_from_classifier = classifier_softmax[~mask_lab][mask_preds]
    # for q, t in zip(quantile_list, dist_quantiles):
    #
    #     print(q.item(), t.item())
    #     for i in range(len(preds_to_adjust)):
    #
    #         ss_k_p = preds_to_adjust[i].argmax()
    #         classif_p = preds_from_classifier[i].argmax()
    #
    #         assign_dist = torch.cdist(CLUSTER_CENTERS[ss_k_p][None, :], CLUSTER_CENTERS[classif_p][None, :], p=2)[0, 0]
    #
    #         if ss_k_p != classif_p:
    #             debug = 0
    #
    #         if assign_dist < t:
    #             adjusted_preds[mask_preds][i] = classif_p

        # interpolated_preds = l * preds_from_classifier + (1 - l) * preds_to_adjust
        #
        # adjusted_preds[mask_preds] = interpolated_preds.argmax(dim=-1).cpu().numpy()

        # labelled_acc, ind_o, w_o = cluster_acc(u_targets.astype(int)[mask], adjusted_preds.astype(int)[mask], return_ind=True)
        # unlabelled_acc, ind_l, w_l = cluster_acc(u_targets.astype(int)[~mask], adjusted_preds.astype(int)[~mask], return_ind=True)
        # weight = mask.mean()
        #
        # mean_acc_train = (1 - weight) * unlabelled_acc + weight * labelled_acc
        # print('Train Accuracies: Mean {:.4f} | Lab {:.4f} | Unlab {:.4f}'.format(mean_acc_train, labelled_acc, unlabelled_acc))

    # -----------------------
    # Adjust scheme 3
    # -----------------------
    # mask_preds = torch.Tensor([True if x in range(len(args.train_classes)) else False for x in preds]).bool()
    # for l in (np.arange(11) / 10):
    #
    #     print(l)
    #     preds_to_adjust = torch.nn.Softmax(dim=-1)(-adjusted_dists)
    #     preds_from_classifier = classifier_softmax[~mask_lab]
    #     preds_from_classifier = torch.cat([preds_from_classifier,
    #                                        torch.zeros(preds_from_classifier.size(0),
    #                                                                           len(args.unlabeled_classes))]
    #                                       , dim=-1)
    #
    #     interpolated_preds = l * preds_from_classifier + (1 - l) * preds_to_adjust
    #     adjusted_preds = interpolated_preds.argmax(dim=-1).cpu().numpy()
    #
    #     labelled_acc, ind_o, w_o = cluster_acc(u_targets.astype(int)[mask], adjusted_preds.astype(int)[mask], return_ind=True)
    #     unlabelled_acc, ind_l, w_l = cluster_acc(u_targets.astype(int)[~mask], adjusted_preds.astype(int)[~mask], return_ind=True)
    #     weight = mask.mean()
    #
    #     mean_acc_train = (1 - weight) * unlabelled_acc + weight * labelled_acc
    #     print('Train Accuracies: Mean {:.4f} | Lab {:.4f} | Unlab {:.4f}'.format(mean_acc_train, labelled_acc, unlabelled_acc))

    # -----------------------
    # Adjust scheme 1
    # -----------------------
    # dists_softmax = torch.nn.Softmax(dim=-1)(adjusted_dists)
    # adjusted_preds = deepcopy(preds)
    # mask_preds = torch.Tensor([True if x in range(len(args.train_classes)) else False for x in preds]).bool()
    # classifier_preds = all_logits.argmax(axis=-1)
    #
    # # Do adjust
    # adjusted_preds[mask_preds] = classifier_preds[~mask_lab][mask_preds]

    # Recompute metrics
    # labelled_acc, ind_o, w_o = cluster_acc(u_targets.astype(int)[mask], adjusted_preds.astype(int)[mask], return_ind=True)
    # unlabelled_acc, ind_l, w_l = cluster_acc(u_targets.astype(int)[~mask], adjusted_preds.astype(int)[~mask], return_ind=True)
    # weight = mask.mean()
    #
    # mean_acc_train = (1 - weight) * unlabelled_acc + weight * labelled_acc
    # print('Train Accuracies: Mean {:.4f} | Lab {:.4f} | Unlab {:.4f}'.format(mean_acc_train, labelled_acc, unlabelled_acc))
    # print('All Classes Test acc {:.4f}'.format(evaluate_clustering(u_targets.astype(int), adjusted_preds.astype(int))[0]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pairwise_batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--K', default=None, type=int, help='Set manually to run with custom K')
    parser.add_argument('--root_dir', type=str, default='/work/sagar/osr_novel_categories/extracted_features')
    parser.add_argument('--warmup_model_exp_id', type=str, default='(29.12.2021_|_48.847)')
    # parser.add_argument('--pretrain_path', type=str, default='/work/sagar/osr_novel_categories/uno_v2_gcd/log/(25.01.2022_|_01.953)/checkpoints/vit_dino.pth')
    parser.add_argument('--pretrain_path', type=str, default='/work/sagar/osr_novel_categories/supervised_train_rebuttal/log/(29.01.2022_|_46.411)/checkpoints/vit_dino_linear.pt')

    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--spatial', type=str2bool, default=False)
    parser.add_argument('--semi_sup', type=str2bool, default=True)
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--optimal', type=str2bool, default=False)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='herbarium_19', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)

    # UNO Params
    parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
    parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
    parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
    parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    cluster_accs = {}

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    device = torch.device('cuda:0')
    args.device = device
    print(args)

    args.pretrain_path = pretrain_models_and_exp_ids[args.dataset_name]['supervised_path']
    args.warmup_model_exp_id = pretrain_models_and_exp_ids[args.dataset_name]['feat_id']

    if args.warmup_model_exp_id is not None:
        args.save_dir += '_' + args.warmup_model_exp_id
        args.save_dir += '_best'
        print(f'Using features from experiment: {args.warmup_model_exp_id}')

    args.cluster_save_dir = os.path.join(args.save_dir, 'ss_kmeans_cluster_centres.pt')

    # --------------------
    # LOAD CLUSTER CENTERS
    # --------------------
    print('Loading cluster centers...')
    CLUSTER_CENTERS = torch.load(args.cluster_save_dir).to(args.device)

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    train_transform, test_transform = [get_transforms_uno("eval", 'ImageNet')] * 2
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                             train_transform, test_transform, args)

    # --------------------
    # FEAT VEC DATASETS
    # --------------------
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    # Convert to feature vector dataset
    unlabelled_train_examples_test_feats = FeatureVectorDataset(base_dataset=unlabelled_train_examples_test,
                                                          feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset_feats = FeatureVectorDataset(base_dataset=train_dataset, feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset_feats.target_transform = target_transform

    unlabelled_train_loader_feats = DataLoader(unlabelled_train_examples_test_feats, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    train_loader_feats = DataLoader(train_dataset_feats, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False)

    # --------------------
    # IMAGE DATALOADERS
    # --------------------
    unlabelled_train_examples_test.transform = test_transform
    train_dataset.transform = test_transform

    unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                                    batch_size=args.batch_size, shuffle=False)

    # --------------------
    # BASE MODEL
    # --------------------
    pretrain_path = '/work/sagar/pretrained_models/dino/dino_vitbase16_pretrain.pth'
    base_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

    state_dict = torch.load(pretrain_path, map_location='cpu')
    base_model.load_state_dict(state_dict)

    feat_dim = 768

    # --------------------
    # MODEL
    # --------------------
    # model = MultiHeadModel(base_model, low_res=True,
    #                        num_labeled=args.num_labeled_classes, num_unlabeled=args.num_unlabeled_classes,
    #                        proj_dim=args.proj_dim, hidden_dim=args.hidden_dim,
    #                        overcluster_factor=args.overcluster_factor, num_heads=args.num_heads,
    #                        num_hidden_layers=args.num_hidden_layers, feat_dim=feat_dim,
    #                        pretrain_state_dict=None)

    model = VisionTransformerWithLinear(base_vit=base_model, num_classes=len(args.train_classes))

    print(f'Loading supervised weights from {args.pretrain_path}')
    model.load_state_dict(torch.load(args.pretrain_path))
    model.to(device)

    # --------------------
    # RUN
    # --------------------
    combine_preds(model, train_loader_feats, train_loader, args)