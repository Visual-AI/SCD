import argparse
import os
import sys
from matplotlib.pyplot import get
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append("./gcd/")
sys.path.append("./local_utils/")
# from gcd.data.cifar import get_cifar_100_datasets
import torch
import torch.nn.functional as F
import clip
import numpy as np
from gcd.data.augmentations import get_transform
from gcd.data.get_datasets import get_datasets, get_class_splits
from gcd.project_utils.cluster_and_log_utils import split_cluster_acc_v2

from torch.utils.data import DataLoader
from tqdm import tqdm

from local_utils.util import str2bool
from sklearn.cluster import KMeans
from local_utils.sskm_constrained import K_Means as ConSemiSupKMeans
from gcd.methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from local_utils.clip_lang_util import imagenet_templates, get_nouns, get_wordnet_dict, zeroshot_classifier, accuracy, assign_name
import copy
from collections import Counter

import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_topk_name_indices(loader, cidx_to_cname, nouns, zeroshot_weights):
    name_idx_top5 = []
    name_logits_top5 = []
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target, _, _) in enumerate(tqdm(loader)):
            # convert the target from ImageNet100 labels to WordNet nouns labels
            target = torch.tensor([nouns.index(cidx_to_cname[t]) for t in target.numpy()])
            images = images.cuda()
            target = target.cuda()
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            name_idx_top5.append(logits.topk(5, 1, True, True)[1])
            logit_val_top5 = logits.topk(5, 1, True, True)[0]
            # name_logits_top5.append(logit_val_top5/torch.unsqueeze(logit_val_top5.sum(dim=1),1))
            name_logits_top5.append(logit_val_top5)
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            # break

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    name_idx_top5 = torch.cat(name_idx_top5, dim=0)
    name_logits_top5 = torch.cat(name_logits_top5, dim=0)
    return name_idx_top5, name_logits_top5

def get_clip_preds_fast(clip_feats, targets, cidx_to_cname, nouns, zeroshot_weights):
    batch_feat_size = 1024
    num_batch = int(clip_feats.shape[0] / batch_feat_size)
    clip_preds = []
    for batch_idx in tqdm(range(num_batch+1)):
        if ((batch_idx+1) * batch_feat_size) > clip_feats.shape[0]:
            end_batch_id = clip_feats.shape[0]
        else:
            end_batch_id = (batch_idx+1)*batch_feat_size
        clip_batch_feat = clip_feats[batch_idx*batch_feat_size:end_batch_id]
        batch_targets = targets[batch_idx*batch_feat_size:end_batch_id]
        batch_targets = torch.tensor([nouns.index(cidx_to_cname[t]) for t in batch_targets])
        batch_targets = batch_targets.cuda()
        if torch.is_tensor(clip_batch_feat):
            logits = 100. * clip_batch_feat @ zeroshot_weights
        else:
            logits = 100. * torch.from_numpy(clip_batch_feat).cuda() @ zeroshot_weights
        # logits = F.softmax(logits)
        clip_preds.append(logits.argmax(dim=-1).view(-1))
    clip_preds = torch.cat(clip_preds, dim=0)

    return clip_preds


def evaluate_semantic_acc_ub_lb(clip_feats, targets, cidx_to_cname, nouns, zeroshot_weights):
    batch_feat_size = 1024
    num_batch = int(clip_feats.shape[0] / batch_feat_size)
    top1, top5, n = 0., 0., 0.
    for batch_idx in tqdm(range(num_batch+1)):
        if ((batch_idx+1) * batch_feat_size) > clip_feats.shape[0]:
            end_batch_id = clip_feats.shape[0]
        else:
            end_batch_id = (batch_idx+1)*batch_feat_size
        clip_batch_feat = clip_feats[batch_idx*batch_feat_size:end_batch_id]
        batch_targets = targets[batch_idx*batch_feat_size:end_batch_id]
        batch_targets = torch.tensor([nouns.index(cidx_to_cname[t]) for t in batch_targets])
        batch_targets = batch_targets.cuda()
        if torch.is_tensor(clip_batch_feat):
            logits = 100. * clip_batch_feat @ zeroshot_weights
        else:
            logits = 100. * torch.from_numpy(clip_batch_feat).cuda() @ zeroshot_weights
        # logits = F.softmax(logits)
        acc1, acc5 = accuracy(logits, batch_targets, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += clip_batch_feat.shape[0]
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    # print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")
    return top1


def extract_feature(model, loader, args):
    # train_classes = range(80)
    train_classes = args.train_classes
    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    # First extract all features
    path_list = []
    for batch_idx, (images, label, _, mask_lab_) in enumerate(tqdm(loader)):
        images = images.cuda()
        if args.feat_model == 'clip':
            feats = model.encode_image(images)
        else:
            feats = model(images)
        
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().detach().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in train_classes
                                        else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)
    all_feats = np.concatenate(all_feats)

    data_dict = {}
    data_dict['all_feats']=all_feats
    data_dict['mask_lab']=mask_lab
    data_dict['mask_cls']=mask_cls
    data_dict['targets']=targets

    return data_dict

def evaluate_semantic_acc(u_targets, cidx_to_cname, u_preds, cand_names):
    cname_to_matchedlist = defaultdict(list)
    matched_all = []
    for u_target, u_pred in zip(u_targets, u_preds):
        if cidx_to_cname[u_target] == cand_names[u_pred]:
            cname_to_matchedlist[cidx_to_cname[u_target]].append(1)
            matched_all.append(1)
        else:
            cname_to_matchedlist[cidx_to_cname[u_target]].append(0)
            matched_all.append(0)

    cname_to_semantic_acc = {}
    for cname in cname_to_matchedlist:
        matched_list_perclass = cname_to_matchedlist[cname]
        cname_to_semantic_acc[cname] = sum(matched_list_perclass) / float(len(matched_list_perclass))

    semantic_acc_all = sum(matched_all) / float(len(matched_all))
    semantic_acc_avg = float(sum(cname_to_semantic_acc.values())) / len(cname_to_semantic_acc.values())
    return semantic_acc_avg, semantic_acc_all

def calucate_dis_between_names(pred_name, target_name, wnid_to_synset, name_to_wnids):
    pred_wnids = name_to_wnids[pred_name]
    target_wnids = name_to_wnids[target_name]
    if 0 == len(pred_wnids):
        print(f"pred_name: {pred_name}, {pred_wnids}")
        return
    elif 0 == len(target_wnids):
        print(f"pred_name: {target_name}, {target_wnids}")
        return 

    # print(pred_wnids)
    # print(target_wnids)
    sim_list = []
    for pred_wnid in pred_wnids:
        for target_wnid in target_wnids:
            # sim_list.append(wnid_to_synset[target_wnid].lin_similarity(wnid_to_synset[pred_wnid], nltk.corpus.wordnet_ic.ic('ic-brown.dat')))
            sim_list.append(wnid_to_synset[target_wnid].lch_similarity(wnid_to_synset[pred_wnid]))
    # print(sorted(sim_list))
    return max(sim_list)
    

def evaluate_soft_semantic_acc(u_targets, cidx_to_cname, u_preds, cand_names, wnid_to_synset, name_to_wnids, return_score=False):
    cname_to_matchedlist = defaultdict(list)
    matched_all = []
    for u_target, u_pred in zip(u_targets, u_preds):
        matched_all.append(calucate_dis_between_names(cand_names[u_pred], cidx_to_cname[u_target], wnid_to_synset, name_to_wnids))
    matched_all = np.array(matched_all) / max(matched_all)
    semantic_acc_all = sum(matched_all) / float(len(matched_all))
    if not return_score:
        return semantic_acc_all
    else:
        return semantic_acc_all, matched_all


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--root_dir', type=str, default='/Your_data_dirg')
    parser.add_argument('--dataset_name', type=str, default='imagenet_100', help='options: cifar10, cifar100, imagenet_100, cub, sdogs')
    parser.add_argument('--feat_model', type=str, default='clip', help='option: clip, dino_vit, gcd')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--extract_feat', type=str2bool, default=True)
    parser.add_argument('--run_cluster', type=str2bool, default=False)
    parser.add_argument('--cluster', type=str, default='ConSSKM', help='options: KM, SSKM, ConSSKM')
    parser.add_argument('--save_cluster', type=str2bool, default=False)
    parser.add_argument('--n_cluster', type=int, default=100)
    parser.add_argument('--cluster_size_min', type=int, default=50)
    parser.add_argument('--cluster_size_max', type=int, default=1000)
    parser.add_argument('--corpus', type=str, default='wordnet', help='options: wordnet, wikibird, wikidog')
    parser.add_argument('--topk', type=int, default=5, help='options: imagenet 5, sdogs 1, cub 3')
    parser.add_argument('--num_common_vote', type=int, default=20, help='options: imagenet 20, sdogs 5, cub 10')
    parser.add_argument('--num_common_linear', type=int, default=4, help='options: imagenet 4, sdogs 2, cub 2')

    args = parser.parse_args()


    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    # for cub ssb setting
    if args.dataset_name == 'cub':
        args.use_ssb_splits = True
    args = get_class_splits(args)
    
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)

    # get CLIP model
    model, preprocess = clip.load("ViT-B/16")
    model.cuda().eval()

    if args.feat_model == 'dino_vit':
        feat_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
        feat_model.cuda().eval()
    elif args.feat_model == 'gcd':
        feat_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        if args.dataset_name == 'imagenet_100':
            pretrain_path = args.root_dir + '/GCD_pretrained_weights_VIT16/imagenet100_model_best.pt'
        elif args.dataset_name == 'sdogs':
            pretrain_path = args.root_dir + '/GCD_pretrained_weights_VIT16/sdogs_model_best.pt'
        elif args.dataset_name == 'cub':
            pretrain_path = args.root_dir + '/GCD_pretrained_weights_VIT16/cub_model_best.pt'
        else:
            raise NotImplementedError
        state_dict = torch.load(pretrain_path, map_location='cpu')
        feat_model.load_state_dict(state_dict)
        feat_model.cuda().eval()
    elif args.feat_model == 'clip':
        feat_model = model
        # input_resolution = model.visual.input_resolution
        # context_length = model.context_length
        # vocab_size = model.vocab_size
    elif 'img2text' in args.feat_model:
        pass
    else:
        raise NotImplementedError


    # --------------------
    # DATASETS
    # --------------------
    test_transform = preprocess
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         test_transform,
                                                                                         test_transform,
                                                                                         args)
    # --------------------
    # DATALOADERS
    # --------------------
    # train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
    #                           sampler=sampler, drop_last=True)
    test_loader_all = DataLoader(train_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    # --------------------
    # FEATURE EXTRACTION
    # --------------------
    save_dir = os.path.join(args.root_dir, 'extracted_features', f'{args.feat_model}_{args.dataset_name}_all.pt')
    # if args.extract_feat:
    # if not os.path.exists(save_dir):
    if args.extract_feat:
        data_dict = extract_feature(feat_model, test_loader_all, args)
        torch.save(data_dict, save_dir)
    else:
        data_dict = torch.load(save_dir)


    # --------------------
    # CILP FEATURE EXTRACTION
    # --------------------
    clip_save_dir = os.path.join(args.root_dir, 'extracted_features', f'clip_{args.dataset_name}_all.pt')
    if not os.path.exists(clip_save_dir):
        clip_data_dict = extract_feature(model, test_loader_all, args)
        torch.save(clip_data_dict, clip_save_dir)
    else:
        clip_data_dict = torch.load(clip_save_dir)
    clip_all_feats, clip_mask_lab, clip_mask_cls, clip_targets = clip_data_dict['all_feats'], clip_data_dict['mask_lab'], clip_data_dict['mask_cls'], clip_data_dict['targets']
    clip_u_feats = clip_all_feats[~clip_mask_lab]      # Get unlabelled set
   

    # --------------------
    # CLUSTERING
    # --------------------
    ## algorithms: kmeans / sskmeans /constrained sskmeans
    all_feats, mask_lab, mask_cls, targets = data_dict['all_feats'], data_dict['mask_lab'], data_dict['mask_cls'], data_dict['targets']
    print(all_feats.shape)


    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    if args.run_cluster:
        cluster_result = {}
        cluster_result['all_preds'] = None
        if args.cluster == 'ConSSKM':
            print('Fitting Constrained Semi-Supervised K-Means...')
            kmeans = ConSemiSupKMeans(k=args.n_cluster, tolerance=1e-4, max_iterations=10, init='k-means++', size_min=args.cluster_size_min, size_max=args.cluster_size_max, n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=1024)
            l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).cuda() for
                                                        x in (l_feats, u_feats, l_targets, u_targets))
            kmeans.fit_mix(u_feats, l_feats, l_targets)
            all_preds = kmeans.labels_.cpu().numpy()
            preds = all_preds[~mask_lab]
            u_targets = u_targets.cpu().numpy()
            cluster_result['all_preds'] = all_preds

        elif args.cluster == 'SSKM':
            print('Fitting Semi-Supervised K-Means...')
            kmeans = SemiSupKMeans(k=args.n_cluster, tolerance=1e-4, max_iterations=10, init='k-means++',
                                    n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)
            l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).cuda() for
                                                        x in (l_feats, u_feats, l_targets, u_targets))
            kmeans.fit_mix(u_feats, l_feats, l_targets)
            all_preds = kmeans.labels_.cpu().numpy()
            preds = all_preds[~mask_lab]
            u_targets = u_targets.cpu().numpy()
            cluster_result['all_preds'] = all_preds

        elif args.cluster == 'KM':
            print('Fitting K-Means...')
            kmeans = KMeans(n_clusters=args.n_cluster, random_state=0).fit(u_feats)
            preds = kmeans.labels_
            u_feats = torch.from_numpy(u_feats).cuda()
    # cluster_estimate_number
    save_dir = os.path.join(args.root_dir, 'cluster', f'{args.cluster}_{args.feat_model}_{args.dataset_name}.pt')
    if args.save_cluster:
        cluster_result['u_preds'] = preds
        cluster_result['u_targets'] = u_targets
        cluster_result['mask'] = mask
        torch.save(cluster_result, save_dir)
    else:
        cluster_result = torch.load(save_dir)
        all_preds, preds, u_targets, mask = cluster_result['all_preds'], cluster_result['u_preds'], cluster_result['u_targets'], cluster_result['mask']
    
    all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=u_targets, y_pred=preds, mask=mask)
    print(f"{args.cluster} Accuracies: All {all_acc} | Old {old_acc} | New {new_acc}")

    # --------------------
    # CLIP VOTING
    nouns = get_nouns(corpus=args.corpus)
    nouns = [n.lower().replace('-','_') for n in nouns]

    # load the zeroshort weights on all nouns
    if 'wordnet' == args.corpus:
        wnid_to_synset, wnid_to_name, name_to_wnids = get_wordnet_dict()
        zeroshot_weights = torch.load(args.root_dir + '/zeroshot_weights/zeroshot_weights_all_nouns_vit_b_16.pt')
    elif 'wikibird' == args.corpus:
        zeroshot_weights = torch.load(args.root_dir + '/zeroshot_weights/zeroshot_weights_all_wikibird_vit_b_16.pt')
        nouns = [n.lower().replace("'s","").replace(' ','_') for n in nouns]
    elif 'wikidog' == args.corpus:
        zeroshot_weights = torch.load(args.root_dir + '/zeroshot_weights/zeroshot_weights_all_wikidog_vit_b_16.pt')
        nouns = [n.lower().replace("'s","").replace(' ','_') for n in nouns]

    # prepare class index to class name mapping with corpus
    if args.dataset_name in ['cifar10', 'cifar100', 'aircraft']:
        original_names = list(datasets['test'].class_to_idx.keys())
        miss_names = [n for n in original_names if n not in nouns]
        print(f'Finding closest names in WordNet for {len(miss_names)}/{len(original_names)} missing names ... ')
        miss_name_weights = zeroshot_classifier(miss_names, imagenet_templates, model)
        logits = 100. * miss_name_weights.t() @ zeroshot_weights
        new_name_idx_top1= logits.topk(1, 1, True, True)[1]
        new_name_idx_top1 = new_name_idx_top1.view(-1).cpu().numpy()
        matched_names = [nouns[i] for i in new_name_idx_top1]
        cidx_to_cname = {}
        for name, idx in datasets['test'].class_to_idx.items():
            if name not in miss_names:
                cidx_to_cname[idx] = name
            else:
                cidx_to_cname[idx] = matched_names[miss_names.index(name)]
        print(f'Missed {len(miss_names)} names and matched {len(set(matched_names))} names ... ')
        for name_pair in zip(miss_names, matched_names):
            print(name_pair)
    elif 'imagenet_100' == args.dataset_name or 'imagenet_1000' == args.dataset_name:
        idx1000_to_idx100 = datasets['class_map']
        idx100_to_idx1000={}
        for k, v in idx1000_to_idx100.items():
            idx100_to_idx1000[v]=k
        imagenet_root = args.root_dir + '/ILSVRC12'
        directory = os.path.join(imagenet_root, 'train')
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        wnid_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_wnid = {i: cls_name for i, cls_name in enumerate(classes)}
        cidx_to_cname = {}
        # cidx_to_ori_cname = {}
        for k, v in idx100_to_idx1000.items():
            cidx_to_cname[k]=wnid_to_name[idx_to_wnid[v]].lower().replace('-','_')
            # cidx_to_ori_cname[k]=wnid_to_name[idx_to_wnid[v]]
    elif 'imagenet_127' == args.dataset_name:
        idx1000_to_idx100 = datasets['class_map']
        idx100_to_idx1000={}
        for k, v in idx1000_to_idx100.items():
            idx100_to_idx1000[v]=k
        imagenet_root = args.root_dir + '/imagenet127'
        directory = os.path.join(imagenet_root, 'val')
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        wnid_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_wnid = {i: cls_name for i, cls_name in enumerate(classes)}
        cidx_to_cname = {}
        for k, v in idx100_to_idx1000.items():
            cidx_to_cname[k]=wnid_to_name[idx_to_wnid[v]].lower().replace('-','_')

    elif 'sdogs' == args.dataset_name:
        wnid_names = sorted(test_dataset._breeds)
        wnid_to_cname = {}
        cidx_to_cname = {}
        class_to_idx = {}
        for i, w_n in enumerate(wnid_names):
            w, n = w_n[:9], w_n[10:]
            n = n.lower().replace('-','_')
            wnid_to_cname[w] = n
            cidx_to_cname[i] = n
            class_to_idx[n] = i
        if 'wikidog' == args.corpus:
            miss_names = [n for n in wnid_to_cname.values() if n not in nouns]
            nouns_truncated = [n for n in nouns if n not in wnid_to_cname.values()]
            print('Finding closet names in Wikidogs for missing names ... ')
            miss_name_weights = zeroshot_classifier(miss_names, imagenet_templates, model)
            logits = 100. * miss_name_weights.t() @ torch.stack([zeroshot_weights[:, nouns.index(n)] for n in nouns_truncated], dim=1)
            new_name_idx_top5= logits.topk(5, 1, True, True)[1]
            matched_names=[]
            for i in range(len(miss_names)):
                j = 0
                idx = new_name_idx_top5[i, j].item()
                while nouns_truncated[idx] in matched_names:
                    j+=1
                    idx = new_name_idx_top5[i, j].item()
                matched_names.append(nouns_truncated[idx]) 
            print(f'Missed {len(miss_names)} names and matched {len(set(matched_names))} names ... ')
            for name_pair in zip(miss_names, matched_names):
                print(name_pair)   

            cidx_to_cname = {}
            for name, idx in class_to_idx.items():
                if name not in miss_names:
                    cidx_to_cname[idx] = name
                else:
                    cidx_to_cname[idx] = matched_names[miss_names.index(name)]           
    elif 'cub' == args.dataset_name:
        classnames = test_dataset.classnames
        class_to_idx = {name.split('.')[1].lower().replace('-','_'):int(name.split('.')[0])-1 for name in classnames}
        original_names = list(class_to_idx.keys())
        miss_names = [n for n in original_names if n not in nouns]
        nouns_truncated = [n for n in nouns if n not in original_names]
        print('Finding closet names in WikiBird for missing names ... ')
        miss_name_weights = zeroshot_classifier(miss_names, imagenet_templates, model)
        logits = 100. * miss_name_weights.t() @ torch.stack([zeroshot_weights[:, nouns.index(n)] for n in nouns_truncated], dim=1)
        new_name_idx_top1= logits.topk(1, 1, True, True)[1]
        new_name_idx_top1 = new_name_idx_top1.view(-1).cpu().numpy()
        matched_names = [nouns_truncated[i] for i in new_name_idx_top1]

        cidx_to_cname = {}
        for name, idx in class_to_idx.items():
            if name not in miss_names:
                cidx_to_cname[idx] = name
            else:
                cidx_to_cname[idx] = matched_names[miss_names.index(name)]
        print(f'Missed {len(miss_names)} names and matched {len(set(matched_names))} names ... ')
        for name_pair in zip(miss_names, matched_names):
            print(name_pair)        
        print('raw classnames', len(set(classnames)), 'updated names', len(set(cidx_to_cname.values()))) 

    ## obtain top 5 predictions by CLIP 
    # name_idx_top5, name_logits_top5  = get_topk_name_indices(test_loader_all, cidx_to_cname,  nouns, zeroshot_weights)
    # name_idx_top5 = torch.load('/work/khan/language_ncd/cifar100_name_idx_top5_vit_b_16_merged.pt')

    TOP_K = 5 # imagenet (5); sdogs (2)
    name_idx_top5 = []
    name_logits_top5 = []
    batch_feat_size = 1024
    num_batch = int(clip_all_feats.shape[0] / batch_feat_size)
    for batch_idx in tqdm(range(num_batch+1)):
        if ((batch_idx+1) * batch_feat_size) > clip_all_feats.shape[0]:
            end_batch_id = clip_all_feats.shape[0]
        else:
            end_batch_id = (batch_idx+1)*batch_feat_size
        clip_batch_feat = clip_all_feats[batch_idx*batch_feat_size:end_batch_id]
        if torch.is_tensor(clip_batch_feat):
            logits = 100. * clip_batch_feat @ zeroshot_weights
        else:
            logits = 100. * torch.from_numpy(clip_batch_feat).cuda() @ zeroshot_weights
        # logits = F.softmax(logits)
        name_idx_top5.append(logits.topk(TOP_K, 1, True, True)[1])
        name_logits_top5.append(logits.topk(TOP_K, 1, True, True)[0])
    name_idx_top5 = torch.cat(name_idx_top5, dim=0)
    name_logits_top5 = torch.cat(name_logits_top5, dim=0)


    # get semantic acc upper bound and lower bound
    print('=====sACC lower bound=====')
    sACC_all = evaluate_semantic_acc_ub_lb(clip_all_feats[~mask_lab], u_targets, cidx_to_cname, nouns, zeroshot_weights)
    sACC_old = evaluate_semantic_acc_ub_lb(clip_all_feats[~mask_lab][mask], u_targets[mask], cidx_to_cname, nouns, zeroshot_weights)
    sACC_new = evaluate_semantic_acc_ub_lb(clip_all_feats[~mask_lab][~mask], u_targets[~mask], cidx_to_cname, nouns, zeroshot_weights)
    print(f"sACC all {sACC_all},sACC old {sACC_old}, sACC new {sACC_new}")
    print('=====sACC upper bound=====')
    cand_names = [n.lower().replace('-','_') for n in list(cidx_to_cname.values())]
    zeroshot_weights_selected = [zeroshot_weights[:, nouns.index(n)] for n in cand_names]
    zeroshot_weights_selected = torch.stack(zeroshot_weights_selected, dim=1)
    sACC_all = evaluate_semantic_acc_ub_lb(clip_all_feats[~mask_lab], u_targets, cidx_to_cname, cand_names, zeroshot_weights_selected)
    sACC_old = evaluate_semantic_acc_ub_lb(clip_all_feats[~mask_lab][mask], u_targets[mask], cidx_to_cname, cand_names, zeroshot_weights_selected)
    sACC_new = evaluate_semantic_acc_ub_lb(clip_all_feats[~mask_lab][~mask], u_targets[~mask], cidx_to_cname, cand_names, zeroshot_weights_selected)
    print(f"sACC all {sACC_all},sACC old {sACC_old}, sACC new {sACC_new}")


    if 'cub' != args.dataset_name:
        # get acc upper bound and lower bound
        print('=====Soft sACC lower bound=====')
        clip_u_preds = get_clip_preds_fast(clip_all_feats[~mask_lab], u_targets, cidx_to_cname, nouns, zeroshot_weights)
        clip_u_preds = clip_u_preds.cpu().numpy()
        all_semantic_acc_all = evaluate_soft_semantic_acc(u_targets, cidx_to_cname, clip_u_preds, nouns, wnid_to_synset, name_to_wnids)
        old_semantic_acc_all = evaluate_soft_semantic_acc(u_targets[mask], cidx_to_cname, clip_u_preds[mask], nouns, wnid_to_synset, name_to_wnids)
        new_semantic_acc_all = evaluate_soft_semantic_acc(u_targets[~mask], cidx_to_cname, clip_u_preds[~mask], nouns, wnid_to_synset, name_to_wnids)
        print(f"Soft sACC all {all_semantic_acc_all},sACC old {old_semantic_acc_all}, sACC new {new_semantic_acc_all}")
        print('=====Soft sACC upper bound=====')
        cand_names = [n.lower().replace('-','_') for n in list(cidx_to_cname.values())]
        zeroshot_weights_selected = [zeroshot_weights[:, nouns.index(n)] for n in cand_names]
        zeroshot_weights_selected = torch.stack(zeroshot_weights_selected, dim=1)
        clip_u_preds = get_clip_preds_fast(clip_all_feats[~mask_lab], u_targets, cidx_to_cname, cand_names, zeroshot_weights_selected)
        clip_u_preds = clip_u_preds.cpu().numpy()
        print(len(list(set(clip_u_preds))))
        all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=u_targets, y_pred=clip_u_preds, mask=mask, return_ind_map=False)
        print(f"clip ACC: All {all_acc} | Old {old_acc} | New {new_acc}")
        all_semantic_acc_all = evaluate_soft_semantic_acc(u_targets, cidx_to_cname, clip_u_preds, cand_names, wnid_to_synset, name_to_wnids)
        old_semantic_acc_all = evaluate_soft_semantic_acc(u_targets[mask], cidx_to_cname, clip_u_preds[mask], cand_names, wnid_to_synset, name_to_wnids)
        new_semantic_acc_all = evaluate_soft_semantic_acc(u_targets[~mask], cidx_to_cname, clip_u_preds[~mask], cand_names, wnid_to_synset, name_to_wnids)
        print(f"Soft sACC all {all_semantic_acc_all},sACC old {old_semantic_acc_all}, sACC new {new_semantic_acc_all}")


    name_idx_top5 = name_idx_top5[~mask_lab]
    name_logits_top5 = name_logits_top5[~mask_lab]

    u_preds = all_preds[~mask_lab]
    l_preds = all_preds[mask_lab]
    gt_names = list(cidx_to_cname.values())
    lab_class_index = list(set(l_preds))
    all_class_index = list(set(all_preds))
    cand_names = nouns
    lab_names = [gt_names[cid] for cid in args.train_classes]

    num_lab_classes = len(args.train_classes)
    # changed by yandong
    # num_unlab_classes = len(args.unlabeled_classes)
    num_unlab_classes = args.n_cluster - num_lab_classes
    known_name_idx = [cand_names.index(n) for n in lab_names]

    ## get the child names of the known names (one level), can add more levels if needed
    # child_names = []
    # for n in lab_names:
    #     for s in wn.synsets(n):
    #         children = s.hyponyms()
    #         child_names += [c.lemma_names()[0] for c in children]
    # child_names = list(set(child_names))
    # child_name_idx = [cand_names.index(n) for n in child_names if n in cand_names]
    # known_name_idx = list(set(known_name_idx + child_name_idx))

    cur_voted_names = [0]
    prev_voted_names = [1]
    top_k = args.topk
    it = 0


    all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=u_targets, y_pred=u_preds, mask=mask, return_ind_map=False)
    print(f"{args.cluster} Accuracies: All {all_acc} | Old {old_acc} | New {new_acc}")

    # voting without unlabeled images in labeled classes
    unlab_cluster_idx = list(set(all_class_index) - set(lab_class_index))
    print(unlab_cluster_idx)
    print(num_unlab_classes)

    while (set(cur_voted_names)!=set(prev_voted_names)):
        it = it+1
        print('iter:', it)
        # print(set(cur_voted_names)-set(prev_voted_names))
        # here we use the cluster id to match the class id 
        # may have some issue

        cluster_to_counter={}
        for i in unlab_cluster_idx:
            cluster_to_counter[i] = Counter(x for x in name_idx_top5[u_preds==i, :top_k].view(-1).cpu().numpy() if x not in known_name_idx)
            ## prun known name idx list

        voted_unique_name_idx = []
        for i in unlab_cluster_idx:
            # imagenet(20) sdogs(2) cub(10)
            candidates = cluster_to_counter[i].most_common(args.num_common_vote)
            for c in candidates:
                voted_unique_name_idx += [c[0]]
        print(len(set(voted_unique_name_idx)))
        voted_unique_name_idx = list(set(voted_unique_name_idx))
        ind, w = assign_name(voted_unique_name_idx, cluster_to_counter, num_common = args.num_common_linear)
        newidx_to_unameidx = {nidx:uidx for nidx, uidx in enumerate(voted_unique_name_idx)}
        newcidx_to_unlabcidx = {ncidx:ucidx for ncidx, ucidx in enumerate(unlab_cluster_idx)}

        print('num_unlab_classes', num_unlab_classes, 'newidx_to_unameidx', len(newidx_to_unameidx))
        prev_voted_names = copy.deepcopy(cur_voted_names)
        cur_voted_names = [nouns[newidx_to_unameidx[x[1]]] for x in ind[:num_unlab_classes]]

        cand_names = copy.deepcopy(list(set(cur_voted_names+lab_names)))
        # Make the results of different run consistent
        cand_names = sorted(cand_names)
        print(len(cand_names))

        lab_class_index = [cand_names.index(n) for n in lab_names]

        unlab_cluster_idx = [cand_names.index(n) for n in list(set(cand_names)-set(lab_names))]

        known_name_idx = copy.deepcopy(lab_class_index)

        zeroshot_weights_selected = [zeroshot_weights[:, nouns.index(n)] for n in cand_names]
        zeroshot_weights_selected = torch.stack(zeroshot_weights_selected, dim=1)

        if torch.is_tensor(clip_u_feats):
            logits = 100. * clip_u_feats @ zeroshot_weights_selected
        else:
            logits = 100. * torch.from_numpy(clip_u_feats).cuda() @ zeroshot_weights_selected

        u_preds = logits.argmax(dim=-1).view(-1).cpu().numpy()

        # name_idx_top5 = logits.topk(5, 1, True, True)[1]

        all_acc, old_acc, new_acc, ind_map = split_cluster_acc_v2(y_true=u_targets, y_pred=u_preds, mask=mask, return_ind_map=True)
        print(f"Accuracies: All {all_acc} | Old {old_acc} | New {new_acc}")

        all_semantic_acc_avg, all_semantic_acc_all = evaluate_semantic_acc(u_targets, cidx_to_cname, u_preds, cand_names)
        # print(f"Semantic Accuracies (all): avg {semantic_acc_avg} | all {semantic_acc_all} ")
        old_semantic_acc_avg, old_semantic_acc_all = evaluate_semantic_acc(u_targets[mask], cidx_to_cname, u_preds[mask], cand_names)
        # print(f"Semantic Accuracies (old): avg {semantic_acc_avg} | all {semantic_acc_all} ")
        new_semantic_acc_avg, new_semantic_acc_all = evaluate_semantic_acc(u_targets[~mask], cidx_to_cname, u_preds[~mask], cand_names)
        # print(f"Semantic Accuracies (new): avg {semantic_acc_avg} | all {semantic_acc_all} ")

        print(f"ACC/sACC_avg/sACC_all: All {round(all_acc*100,2)}/{round(all_semantic_acc_avg*100,2)}/{round(all_semantic_acc_all*100,2)} ")
        print(f"ACC/sACC_avg/sACC_all: old {round(old_acc*100,2)}/{round(old_semantic_acc_avg*100,2)}/{round(old_semantic_acc_all*100,2)} ")
        print(f"ACC/sACC_avg/sACC_all: new {round(new_acc*100,2)}/{round(new_semantic_acc_avg*100,2)}/{round(new_semantic_acc_all*100,2)} ")

        if 'cub' != args.dataset_name:

            all_semantic_acc_all = evaluate_soft_semantic_acc(u_targets, cidx_to_cname, u_preds, cand_names, wnid_to_synset, name_to_wnids)
            # print(f"Semantic Accuracies (all): avg {semantic_acc_avg} | all {semantic_acc_all} ")
            old_semantic_acc_all = evaluate_soft_semantic_acc(u_targets[mask], cidx_to_cname, u_preds[mask], cand_names, wnid_to_synset, name_to_wnids)
            # # print(f"Semantic Accuracies (old): avg {semantic_acc_avg} | all {semantic_acc_all} ")
            new_semantic_acc_all = evaluate_soft_semantic_acc(u_targets[~mask], cidx_to_cname, u_preds[~mask], cand_names, wnid_to_synset, name_to_wnids)
            # # print(f"Semantic Accuracies (new): avg {semantic_acc_avg} | all {semantic_acc_all} ")

            print(f"ACC/Soft sACC: All {round(all_acc*100,2)}/{round(all_semantic_acc_all*100,2)}")
            print(f"ACC/Soft sACC: old {round(old_acc*100,2)}/{round(old_semantic_acc_all*100,2)}")
            print(f"ACC/Soft sACC: new {round(new_acc*100,2)}/{round(new_semantic_acc_all*100,2)}")


# obtain IoU of predicted names and GT names

inter = set.intersection(set(cand_names), set(gt_names))
union = set.union(set(cand_names), set(gt_names))
print(f'IoU: {len(inter)*1.0/len(union)}')



# get soft semantic acc score
all_semantic_acc_all,  match_score_all = evaluate_soft_semantic_acc(u_targets, cidx_to_cname, u_preds, cand_names, wnid_to_synset, name_to_wnids, return_score=True)

soft_sACC_dict = {}
soft_sACC_dict['match_score'] = match_score_all
soft_sACC_dict['u_targets'] = u_targets
soft_sACC_dict['cidx_to_cname'] = cidx_to_cname
soft_sACC_dict['u_preds'] = u_preds
soft_sACC_dict['cand_names'] = cand_names
soft_sACC_dict['mask_seen'] = mask

if 0:
    save_dir = os.path.join(args.root_dir, 'soft_sACC_dict', f'{args.feat_model}_{args.dataset_name}_soft_sACC_dict.pt')
    torch.save(soft_sACC_dict, save_dir)