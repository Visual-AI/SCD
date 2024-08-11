import torch
from torch.utils.data import DataLoader
import timm
from torchvision import transforms
import torchvision

import argparse
import os
from tqdm import tqdm

from data.stanford_cars import CarsDataset
from data.cifar import CustomCIFAR10, CustomCIFAR100, cifar_10_root, cifar_100_root
from data.herbarium_19 import HerbariumDataset19, herbarium_dataroot
from data.augmentations import get_transform
from data.imagenet import get_imagenet_100_datasets
from data.data_utils import MergedDataset
from data.cub import CustomCub2011, cub_root

from project_utils.general_utils import strip_state_dict, str2bool
from copy import deepcopy

def extract_features_dino(model, loader, save_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


def extract_features_timm(model, loader, save_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model.forward_features(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default='/work/sagar/osr_novel_categories/extracted_features')
    parser.add_argument('--warmup_model_dir', type=str,
                        default='/work/sagar/osr_novel_categories/metric_learn_spat/log/(08.12.2021_|_36.526)/checkpoints/model.pt')
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset', type=str, default='scars', help='options: cifar10, cifar100, scars')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset}')
    print(args)

    print('Loading model...')
    # ----------------------
    # MODEL
    # ----------------------
    if args.model_name == 'vit_dino':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/work/sagar/pretrained_models/dino/dino_vitbase16_pretrain.pth'

        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        _, val_transform = get_transform('imagenet', image_size=224, args=args)

    elif args.model_name == 'vit_s_dino_pass':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/work/sagar/pretrained_models/pass/dino_deit_300ep_ttemp0o07_warumup30ep_normlayerF.pth.tar'

        model = torch.hub.load('yukimasano/PASS:main', 'dino_vits16', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        state_dict = strip_state_dict(state_dict['teacher'])

        del_keys = [k for k, v in state_dict.items() if 'head' in k]
        for k in del_keys:
            del state_dict[k]

        model.load_state_dict(state_dict)

        _, val_transform = get_transform('imagenet', image_size=224, args=args)

    elif args.model_name == 'resnet50_dino':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/work/sagar/pretrained_models/dino/dino_resnet50_pretrain.pth'

        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        _, val_transform = get_transform('imagenet', image_size=224, args=args)

    elif args.model_name == 'vit_supervised':

        model = timm.create_model('vit_base_patch16_224_miil', pretrained=True)
        model.to(device)

        cfg = model.default_cfg
        image_size, interpolation, crop_pct, mean, std = \
            cfg['input_size'][1], cfg['interpolation'], cfg['crop_pct'], cfg['mean'], cfg['std']

        interpolation_ = torchvision.transforms.InterpolationMode.BILINEAR \
            if interpolation == 'bilinear' else torchvision.transforms.InterpolationMode.BICUBIC

        val_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation_),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        extract_features_func = extract_features_timm

    elif args.model_name == 'resnet50_supervised':

        model = timm.create_model('resnet50', pretrained=True)
        model.to(device)

        cfg = model.default_cfg
        image_size, interpolation, crop_pct, mean, std = \
            cfg['input_size'][1], cfg['interpolation'], cfg['crop_pct'], cfg['mean'], cfg['std']

        interpolation_ = torchvision.transforms.InterpolationMode.BILINEAR \
            if interpolation == 'bilinear' else torchvision.transforms.InterpolationMode.BICUBIC

        val_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation_),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        extract_features_func = extract_features_timm

    else:

        raise NotImplementedError

    if args.warmup_model_dir is not None:

        warmup_id = args.warmup_model_dir.split('(')[1].split(')')[0]

        if args.use_best_model:
            args.warmup_model_dir = args.warmup_model_dir[:-3] + '_best.pt'
            args.save_dir += '_(' + args.warmup_model_dir.split('(')[1].split(')')[0] + ')_best'
        else:
            args.save_dir += '_(' + args.warmup_model_dir.split('(')[1].split(')')[0] + ')'

        print(f'Using weights from {args.warmup_model_dir} ...')
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict)

        print(f'Saving to {args.save_dir}')

    print('Loading data...')
    # ----------------------
    # DATASET
    # ----------------------
    if args.dataset == 'cifar10':

        train_dataset = CustomCIFAR10(root=cifar_10_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR10(root=cifar_10_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'cifar100':

        train_dataset = CustomCIFAR100(root=cifar_100_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR100(root=cifar_100_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'scars':

        train_dataset = CarsDataset(train=True, transform=val_transform)
        test_dataset = CarsDataset(train=False, transform=val_transform)
        targets = list(set(train_dataset.target))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 197. Change to 0 - 196

    elif args.dataset == 'herbarium_19':

        train_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-train'),
                                           transform=val_transform)

        test_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-validation'),
                                           transform=val_transform)

        targets = list(set(train_dataset.targets))

    elif args.dataset == 'imagenet_100':

        datasets = get_imagenet_100_datasets(train_transform=val_transform, test_transform=val_transform,
                                             train_classes=range(50),
                                             prop_train_labels=0.5)

        datasets['train_labelled'].target_transform = None
        datasets['train_unlabelled'].target_transform = None

        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

        test_dataset = datasets['test']
        targets = list(set(test_dataset.targets))

    elif args.dataset == 'cub':

        train_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=True)
        test_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=False)
        targets = list(set(train_dataset.data.target.values))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 200. Change to 0 - 199

    else:

        raise NotImplementedError

    # ----------------------
    # DATALOADER
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Creating base directories...')
    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for fold in ('train', 'test'):

        fold_dir = os.path.join(args.save_dir, fold)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        for t in targets:
            target_dir = os.path.join(fold_dir, f'{t}')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

    # ----------------------
    # EXTRACT FEATURES
    # ----------------------
    # Extract train features
    train_save_dir = os.path.join(args.save_dir, 'train')
    print('Extracting features from train split...')
    extract_features_func(model=model, loader=train_loader, save_dir=train_save_dir)

    # Extract test features
    test_save_dir = os.path.join(args.save_dir, 'test')
    print('Extracting features from test split...')
    extract_features_func(model=model, loader=test_loader, save_dir=test_save_dir)

    print('Done!')