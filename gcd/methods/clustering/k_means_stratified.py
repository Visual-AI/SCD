import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_utils import cluster_acc, str2bool
from project_utils.cluster_and_log_utils import log_accs_from_preds
import pickle

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from data.get_datasets import get_datasets, get_class_splits, osr_split_dir
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from methods.clustering.faster_mix_k_means_pytorch import pairwise_distance

from methods.clustering.k_means import test_kmeans

from tqdm import tqdm

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_fgvc_osr_class_splits(args):

    if args.dataset_name == 'scars':

        split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')

    elif args.dataset_name == 'aircraft':

        split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')

    elif args.dataset_name == 'cub':

        split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')

    else:

        raise ValueError

    with open(split_path, 'rb') as handle:
        class_info = pickle.load(handle)

    open_set_classes = class_info['unknown_classes']
    args.stratified_osr_classes = open_set_classes

    return args


def test_kmeans_semi_sup(merge_test_loader, args, K=None):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    Meant to work on FGVC datasets with stratified difficulty
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

        feats = torch.nn.functional.normalize(feats, dim=-1)

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
    log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs,
                        save_name='SS-K-Means All Data Train ACC', print_output=True)

    # -----------------------
    # EVALUATE STRATIFIED
    # -----------------------
    target_transform_state_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_state_dict[cls] = i

    for diff in ('Easy', 'Medium', 'Hard'):

        diff_classes = args.stratified_osr_classes[diff]
        transformed_diff_classes = [target_transform_state_dict[cls] for cls in diff_classes]

        # Create a boolean mask which extracts all instances in unlabelled set which are in the unlabelled classes
        # of this difficulty
        diff_mask = []
        for cls in enumerate(u_targets):
            if cls in transformed_diff_classes:
                diff_mask.append(True)
            else:
                diff_mask.append(False)

        diff_mask = np.array(diff_mask).astype(bool)
        diff_preds = preds[diff_mask]
        diff_targets = u_targets[diff_mask]

        log_accs_from_preds(y_true=diff_targets, y_pred=diff_preds, mask=mask[diff_mask], eval_funcs=args.eval_funcs,
                            save_name=f'Train ACC Unlabelled {diff}', print_output=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--K', default=None, type=int, help='Set manually to run with custom K')
    parser.add_argument('--root_dir', type=str, default='/work/sagar/osr_novel_categories/extracted_features_old')
    parser.add_argument('--warmup_model_exp_id', type=str, default='(05.01.2022_|_23.648)')
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--spatial', type=str2bool, default=False)
    parser.add_argument('--semi_sup', type=str2bool, default=False)
    parser.add_argument('--max_kmeans_iter', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_custom_fgvc_splits', type=str2bool, default=True)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2', 'v3'])

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

    if args.warmup_model_exp_id is not None:
        args.save_dir += '_' + args.warmup_model_exp_id

        if args.spatial:
            args.save_dir += '_spat'

        if args.use_best_model:
            args.save_dir += '_best'

        print(f'Using features from experiment: {args.warmup_model_exp_id}')
    else:
        print(f'Using pretrained {args.model_name} features...')

    # --------------------
    # GET CLASS INDICES OF DIFFERENT DIFFICULTY
    # --------------------
    args = get_fgvc_osr_class_splits(args)

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    train_transform, test_transform = None, None
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                             train_transform, test_transform, args)

    # Convert to feature vector dataset
    test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=os.path.join(args.save_dir, 'test'))
    unlabelled_train_examples_test = FeatureVectorDataset(base_dataset=unlabelled_train_examples_test,
                                                          feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset = FeatureVectorDataset(base_dataset=train_dataset, feature_root=os.path.join(args.save_dir, 'train'))

    unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False)

    print('Testing on all in the training data...')
    test_kmeans_semi_sup(train_loader, args, K=args.K)