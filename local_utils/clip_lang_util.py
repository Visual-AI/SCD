import nltk
from nltk.corpus import wordnet as wn
import torch
import os
from tqdm import tqdm
import numpy as np
import clip

from collections import defaultdict
from gcd.project_utils.cluster_utils import linear_assignment


imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights




def get_wordnet_dict():
    all_noun = wn.all_synsets('n')
    print(all_noun)
    print(wn.all_synsets('n'))
    all_num = len(set(all_noun))
    noun_have_hypon = [word for word in wn.all_synsets('n') if len(word.hyponyms()) >= 1]
    noun_have_num = len(noun_have_hypon)
    print('There are %d nouns, and %d nouns without hyponyms, the percentage is %f' %
    (all_num, noun_have_num, (all_num-noun_have_num)/all_num*100))

    nouns_with_hyper = list(filter(lambda ss: len(ss.hypernyms()) > 0, wn.all_synsets('n')))
    offsets = [n.offset() for n in nouns_with_hyper]
    wnids = list(["n{:08d}".format(o) for o in offsets])

    wnid_to_synset = {}
    wnid_to_name = {}
    name_to_wnids = defaultdict(list)
    # for n in nouns_with_hyper:
    for n in wn.all_synsets('n'):
        wnid_to_synset["n{:08d}".format(n.offset())] = n
        name = n.lemma_names()[0].lower().replace('-','_')
        wnid_to_name["n{:08d}".format(n.offset())] = name
        name_to_wnids[name].append("n{:08d}".format(n.offset()))

    return wnid_to_synset, wnid_to_name, name_to_wnids

def get_nouns(corpus='wordnet'):
    if 'wordnet' == corpus:
        with open('/disk/work/xhhuang/scd_v1/language_ncd_yandong/data/wordnet_all_noun.txt') as f:
            nouns = [line.rstrip('\n') for line in f]
    elif 'wikibird' == corpus:
        with open('/disk/work/xhhuang/scd_v1/language_ncd_yandong/data/wiki_birdclass_names.txt') as f:
            nouns = [line.rstrip('\n') for line in f] 
    elif 'wikidog' == corpus:
        with open('/disk/work/xhhuang/scd_v1/language_ncd_yandong/data/wiki_dogclass_names.txt') as f:
            nouns = [line.rstrip('\n') for line in f]               
    return nouns

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def assign_name(unique_name_idx, cluster_to_counter, num_common = 4):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    unameidx_to_newidx = {uidx:nidx for nidx, uidx in enumerate(unique_name_idx)}
    unlab_cluster_idx = list(cluster_to_counter.keys())
    D = max(len(unique_name_idx), len(unlab_cluster_idx))
    w = np.zeros((D, D), dtype=int)
    for i in range(len(unlab_cluster_idx)):
        ct = cluster_to_counter[unlab_cluster_idx[i]]
        #imagenet100: 4
        # sdogs: 2
        for k, v in ct.most_common(num_common): 
            w[i, unameidx_to_newidx[k]] += v

    ind = linear_assignment(w.max() - w)

    return ind, w

def assign_name_on_leftover(unique_name_idx, cluster_to_counter, voted_unique_name_idx):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    unameidx_to_newidx = {uidx:nidx for nidx, uidx in enumerate(unique_name_idx)}
    unlab_cluster_idx = list(cluster_to_counter.keys())
    D = max(len(unique_name_idx), len(unlab_cluster_idx))
    w = np.zeros((D, D), dtype=int)
    for i in range(len(unlab_cluster_idx)):
        ct = cluster_to_counter[unlab_cluster_idx[i]]
        for k, v in ct.most_common(5): 
            if k in voted_unique_name_idx:
                continue
            w[i, unameidx_to_newidx[k]] += v

    ind = linear_assignment(w.max() - w)

    return ind, w

def assign_name_logits(unique_name_idx, cluster_to_logitcounter):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    unameidx_to_newidx = {uidx:nidx for nidx, uidx in enumerate(unique_name_idx)}
    unlab_cluster_idx = list(cluster_to_logitcounter.keys())
    D = max(len(unique_name_idx), len(unlab_cluster_idx))
    w = np.zeros((D, D), dtype=int)
    for i in range(len(unlab_cluster_idx)):
        ct = cluster_to_logitcounter[unlab_cluster_idx[i]]
        sorted_ct = sorted(ct.items(), key=lambda kv: kv[1], reverse=True)
        candidates = []
        for name_idx_per_cluster in range(min(4,len(sorted_ct))):
            candidates.append(sorted_ct[name_idx_per_cluster])
        for k, v in candidates: 
            w[i, unameidx_to_newidx[k]] += v

    ind = linear_assignment(w.max() - w)

    return ind, w
