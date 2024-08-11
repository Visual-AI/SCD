from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
# matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
from .linear_assignment import linear_assignment
import random
import os
import argparse

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def Class2Simi(x,mode='hinge',mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1)==n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)
    
    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2 
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis

def save_tsne(embeddings, labels, path='tsne.png'):
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(embeddings)
    label_names = np.unique(labels).astype(int)
    target_ids = range(len(label_names))

    fig=plt.figure(figsize=(8, 6))
    marker_sz = 5 
    vals = np.linspace(0,1,len(label_names))
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.gist_ncar(vals))
    for i, label_name in zip(target_ids, label_names):
        plt.scatter(X_2d[labels == label_names[i], 0], X_2d[labels == label_names[i], 1], c=cmap(i), label=label_name, s=marker_sz)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('equal')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

def save_tsne_wcolor(embeddings, labels, path='tsne.png'):
    plt.rcParams.update({'font.size': 18})
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(embeddings)
    label_names = np.unique(labels).astype(int)
    target_ids = range(len(label_names))

    fig=plt.figure(figsize=(8, 6))
    marker_sz = 10 
    vals = np.linspace(0,1,len(label_names))
    np.random.shuffle(vals)
    #  plt.ylim(0.0, 1.0)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'springgreen', 'orange', 'purple'
    #  cmap = plt.cm.colors.ListedColormap(plt.cm.gist_ncar(vals))
    for i, label_name in zip(target_ids, label_names):
        plt.scatter(X_2d[labels == label_names[i], 0], X_2d[labels == label_names[i], 1], c=colors[i], label=label_name, s=marker_sz)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  plt.legend()
    plt.axis('equal')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_heatmap(data, figsize= (4, 10), path='heatmap.png'):
   fig=plt.figure(figsize=figsize)
   ax = sns.heatmap(data)
   plt.savefig(path, bbox_inches='tight')
   plt.close(fig)

def save_cvi_curves(val_list, k_list, figsize= (4, 10), path='cvi_curves.png'):
    fig=plt.figure(figsize=figsize)
    plt.figure("CVIs")
    plt.xlabel('k', fontsize=14)
    plt.ylabel(r'$CVIs$', fontsize=14)
    plt.title("CVI Curves", fontsize=14)
    #  plt.xlim(0.0, 1.0)
    #  plt.ylim(0.0, 1.0)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    acc_list = [x[0] for x in val_list]
    nmi_list = [x[1] for x in val_list]
    ari_list = [x[2] for x in val_list]
    silh_list = [x[3] for x in val_list]
    dunn_list = [x[4] for x in val_list]
    plt.plot(k_list, acc_list, color=colors[0], marker='.', linewidth=2, markersize=7, label='acc')
    plt.plot(k_list, nmi_list, color=colors[1], marker='.', linewidth=2, markersize=7, label='nmi')
    plt.plot(k_list, ari_list, color=colors[2], marker='.', linewidth=2, markersize=7, label='ari')
    plt.plot(k_list, silh_list, color=colors[3], marker='.', linewidth=2, markersize=7, label='silh')
    plt.plot(k_list, dunn_list, color=colors[4], marker='.', linewidth=2, markersize=7, label='dunn')
    plt.tight_layout()
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
   uniform_data = np.random.rand(100, 30)
   fig=plt.figure(figsize=(4, 10))
   #  fig=plt.figure()
   ax = sns.heatmap(uniform_data)
   plt.savefig('heatmap.png', bbox_inches='tight')
   plt.close(fig)
   #  plt.show()
