import torchvision
import numpy as np
import torch

import os
import sys
from copy import deepcopy
# sys.path.append("/users/khan/language_ncd/gcd/")
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from gcd.data.data_utils import subsample_instances
# from data.data_utils import subsample_instances


imagenet_root = '/disk/datasets/ILSVRC12/'
imagenet21k_root = '/work/sagar/datasets/imagenet21k_resized_new'
imagenet127_root = '/disk/work/xhhuang/scd_v1/language_ncd_yandong/imagenet127'
osr_split_save_dir = '/users/sagar/open_world_learning/open_set_recognition/data/' \
    'open_set_splits/imagenet_class_sim_matrix_total_path.pt'

class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


def pad_to_longest(list1, list2):

    if len(list2) > len(list1):

        list1 = [None] * (len(list2) - len(list1)) + list1

    elif len(list1) > len(list2):

        list2 = [None] * (len(list1) - len(list2)) + list2

    else:

        pass

    return list1, list2


def preprocess_semantic_tree_v2(tree, compute_sim_matrix=False, save_dir=osr_split_save_dir, total_path=True):

    # Raw data
    class_tree_list_ints = tree['class_tree_list']
    class_list_wnids = tree['class_list']

    # Get semantic tree in terms of WNIDS
    class_tree_list_wnids = {}
    for cls_ in class_tree_list_ints:

        cls_wnid = [class_list_wnids[c] for c in cls_]

        class_tree_list_wnids[cls_wnid[0]] = cls_wnid

    # Get WNIDS
    imagenet_1k_wnids = os.listdir(os.path.join(imagenet_root, 'val'))
    imagenet_21k_wnids = list(set(os.listdir(os.path.join(imagenet21k_root, 'val')))
                              - set(imagenet_1k_wnids))

    intersection = list(set.intersection(set(imagenet_1k_wnids), set(class_tree_list_wnids.keys())))
    sim_matrix = np.zeros((len(intersection), len(imagenet_21k_wnids)))

    if compute_sim_matrix:
        # Get distances between elements of I21K and I1K
        for i, k1_id in enumerate(intersection):
            for j, k2_id in enumerate(imagenet_21k_wnids):

                tree_i1k = class_tree_list_wnids[k1_id]
                tree_i21k = class_tree_list_wnids[k2_id]
                depth_i1k, depth_i21k = len(tree_i1k), len(tree_i21k)

                tree_i1k, tree_i21k = pad_to_longest(tree_i1k, tree_i21k)

                assert len(tree_i1k) == len(tree_i21k)

                dist = np.where(np.array(tree_i1k) == np.array(tree_i21k))[0]

                if total_path:

                    # Compute semantic distance as total distance up and down the semantic tree between
                    # the two classes

                    # If there is no common ancestor (even at the highest level of the hierarchy)
                    if len(dist) == 0:

                        path_length_i1k = depth_i1k
                        path_length_i21k = depth_i21k

                        dist = path_length_i1k + path_length_i21k + 1

                    else:

                        path_length_i1k = dist[0] - (max(depth_i1k, depth_i21k) - depth_i1k)
                        path_length_i21k = dist[0] - (max(depth_i1k, depth_i21k) - depth_i21k)

                        dist = path_length_i1k + path_length_i21k

                else:

                    # Compute semantic distance as total distance UP semantic tree between the two classes

                    # If there is no common ancestor (even at the highest level of the hierarchy)
                    if len(dist) == 0:
                        dist = len(tree_i1k) - 1
                    else:
                        dist = dist[0] - 1

                sim_matrix[i, j] = dist

        to_save = {
            'sim_matrix': sim_matrix,
            'imagenet_1k_wnids': intersection,
            'imagenet_21k_wnids': imagenet_21k_wnids
        }

        print(f'Saving sim matrix to {save_dir}...')
        torch.save(to_save, save_dir)

    else:

        to_save = torch.load(save_dir)
        sim_matrix, imagenet_1k_wnids, imagenet_21k_wnids = to_save['sim_matrix'],\
                                                            to_save['imagenet_1k_wnids'],\
                                                            to_save['imagenet_21k_wnids']

    # Split into easy, medium and hard open set classes
    total_dists = torch.from_numpy(sim_matrix).sum(dim=0)
    sorted_dists, sort_idxs = total_dists.sort()

    hard_classes = sort_idxs[:1000]
    hard_classes = [imagenet_21k_wnids[c] for c in hard_classes]

    middleIndex = int((len(sort_idxs) - 1) / 2)
    medium_classes = sort_idxs[middleIndex-500:middleIndex+500]
    medium_classes = [imagenet_21k_wnids[c] for c in medium_classes]

    easy_classes = sort_idxs[-1000:]
    easy_classes = [imagenet_21k_wnids[c] for c in easy_classes]

    to_save['hard_i21k_classes'] = hard_classes
    to_save['medium_i21k_classes'] = medium_classes
    to_save['easy_i21k_classes'] = easy_classes

    torch.save(to_save, save_dir)


def get_imagenet_osr_class_splits(imagenet21k_class_to_idx, num_imagenet21k_classes=1000,
                                  imagenet_root=imagenet_root, imagenet21k_root=imagenet21k_root,
                                  osr_split='random', precomputed_split_dir=osr_split_save_dir):

    if osr_split == 'random':

        """
        Find which classes in ImageNet21k are not in Imagenet1k, and select some of these classes as open-set classes
        """
        imagenet1k_classes = os.listdir(os.path.join(imagenet_root, 'val'))
        imagenet21k_classes = os.listdir(os.path.join(imagenet21k_root, 'val'))

        # Find which classes in I21K are not in I1K
        disjoint_imagenet21k_classes = set(imagenet21k_classes) - set(imagenet1k_classes)
        disjoint_imagenet21k_classes = list(disjoint_imagenet21k_classes)

        # Randomly select a number of OSR classes from them (must be less than ~10k as only ~11k valid classes in I21K)
        np.random.seed(0)
        selected_osr_classes = np.random.choice(disjoint_imagenet21k_classes, replace=False, size=(num_imagenet21k_classes,))

        # Convert class names to class indices
        selected_osr_classes_class_indices = [imagenet21k_class_to_idx[cls_name] for cls_name in selected_osr_classes]

        return selected_osr_classes_class_indices

    elif osr_split in ('Easy', 'Medium', 'Hard'):

        split_to_key = {
            'Easy': 'easy_i21k_classes',
            'Medium': 'medium_i21k_classes',
            'Hard': 'hard_i21k_classes'
        }

        precomputed_info = torch.load(precomputed_split_dir)
        osr_wnids = precomputed_info[split_to_key[osr_split]]
        selected_osr_classes_class_indices = \
            [imagenet21k_class_to_idx[cls_name] for cls_name in osr_wnids]

        return selected_osr_classes_class_indices


def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_imagenet_100_datasets(train_transform, test_transform, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet_root, 'train'), transform=train_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'class_map': cls_map,
    }

    return all_datasets

def get_imagenet_1000_datasets(train_transform, test_transform, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    subsampled_100_classes = np.random.choice(range(1000), size=(1000,), replace=False)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-1000 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(1000))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=train_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'class_map': cls_map,
    }

    return all_datasets

def get_imagenet_127_datasets(train_transform, test_transform, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    subsampled_100_classes = np.random.choice(range(127), size=(127,), replace=False)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-127 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(127))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet127_root, 'val'), transform=train_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet127_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'class_map': cls_map,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_imagenet_100_datasets(None, None, split_train_val=False,
                               train_classes=range(50), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
