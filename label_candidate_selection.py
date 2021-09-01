import copy


import numpy as np
import torch.nn.functional as F
import sys
import os
import torch.optim as optim


debug = False
directory_path = os.getcwd()
sys.path.append(directory_path)
print ("Adding {} to system path".format(sys.path[-1]))
from Lancet.models import image_model, speechcommand_model
from config import svhn_config as config
from Lancet.utils import *

import copy


class DMN_trainer():
    def __init__(self):
        #self.dmn = image_model.Distribution_Matching_Network().cuda()
        self.dmn = image_model.Distribution_Matching_Network().cuda()
        self.dmn_optimizer = optim.Adam(self.dmn.parameters(), lr=config.dis_lr, betas=(0.5, 0.999))

        self.d_criterion = nn.CrossEntropyLoss()

    def train(self, unl_feats, lab_feats, unl_weight, lab_weight, batch_size = 100, max_epoch = 10):
        self.unl_feats = torch.from_numpy(unl_feats).float().cuda()
        self.lab_feats = torch.from_numpy(lab_feats).float().cuda()

        self.batch_size = batch_size
        self.max_epoch = max_epoch

        i1, i2 = 0,0
        epoch = 0

        while epoch < max_epoch:

            unl_weight = Variable(unl_weight,requires_grad=False)

            i1_ = i1 + 100
            i2_ = i2 + 100

            unl_logits = self.dmn(self.unl_feats[i1:i1_])
            lab_logits = self.dmn(self.lab_feats[i2:i2_])

            lab_loss = F.cross_entropy(lab_logits, torch.ones(lab_logits.shape[0]).long().cuda(), reduction='none')
            unl_loss = F.cross_entropy(unl_logits, torch.zeros(unl_logits.shape[0]).long().cuda(), reduction = 'none')
            loss = lab_loss.mean() + (unl_loss*unl_weight[i1:i1_]).mean()

            self.dmn_optimizer.zero_grad()
            loss.backward()
            self.dmn_optimizer.step()

            unl_logits_ = self.dmn(self.unl_feats[i1:i1_])
            unl_loss_ = F.cross_entropy(unl_logits_, torch.zeros(unl_logits.shape[0]).long().cuda(), reduction = "none")
            unl_weight_ = Variable(unl_weight[i1:i1_],requires_grad=True)

            lab_logits_ = self.dmn(self.lab_feats[i2:i2_])
            lab_loss_  = F.cross_entropy(lab_logits, torch.ones(lab_logits.shape[0]).long().cuda(), reduction='none')
            loss_ = torch.mean(unl_loss_*unl_weight_) + lab_loss_.mean()

            grad_weight = torch.autograd.grad(loss_ , unl_weight_ , only_inputs=True)[0]
            unl_weight[i1:i1_] = torch.abs(grad_weight)

            unl_weight[i1:i1_] = torch.clamp(unl_weight[i1:i1_], min = 0)
            norm_weight = torch.mean(unl_weight[i1:i1_])

            if norm_weight > 0:
                unl_weight[i1:i1_] = unl_weight[i1:i1_] / norm_weight

            loss__ = torch.mean(unl_loss_*unl_weight[i1:i1_]) + lab_loss_.mean()

            if i1_ >= len(unl_feats):
                i1 = 0
                epoch += 1
            else:
                i1 += 100

            if i2_ >= len(lab_feats):
                i2 = 0
            else:
                i2 += 100

        print (loss, unl_loss.mean(), lab_loss.mean(), loss_.mean())

        return unl_weight


def get_feats(trainer, data_loader, max_batch = 740):
    feats = []
    pred_certainty = []
    error = 0
    indices_ = []
    for i, (images, labels, indices) in enumerate(data_loader.get_iter()):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        if images.shape[0] != 100: continue

        feats_batch = trainer.dis(images, feat = True)
        pred_logits = trainer.dis(images)
        pred_cert = torch.softmax(pred_logits, 1)
        error += torch.ne(torch.max(pred_logits,1)[1], labels).data.sum()
        if not len(feats): feats = feats_batch.cpu().detach().numpy()
        else: feats = np.append(feats, feats_batch.cpu().detach().numpy(), axis = 0)

        if not len(pred_certainty): pred_certainty = pred_cert.cpu().detach().numpy()
        else: pred_certainty = np.append(pred_certainty, pred_cert.cpu().detach().numpy(),0)

        if not len(indices_): indices_ = copy.deepcopy(indices)
        else: indices_ = np.append(indices_, indices, axis = 0)
        if max_batch is not None and i >= max_batch - 1: break

    return feats, pred_certainty, indices_




def DBAL(trainer, batch_size = 500, num_classes = 100, num_ensumbles = 5):
    model = trainer.dis
    model.train()


    data_loader = trainer.unlabeled_loader
    dropout_pred = torch.zeros(num_ensumbles, len(trainer.dataset), num_classes)

    cnt = 0
    while cnt < num_ensumbles:
        for i, (images, labels, indices) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda())

            pred_logits = model(images).detach().cpu()

            pred_certainty = torch.softmax(pred_logits, 1)

            dropout_pred[cnt][indices] = pred_certainty

        cnt += 1

    overall_pred = torch.mean(dropout_pred, 0)
    assert overall_pred.shape[0] == len(trainer.dataset)

    new_labeled_indices = np.argsort(torch.max(overall_pred,1)[0])[:batch_size]

    print ("New {} labels are inserted".format(len(new_labeled_indices)))

    return new_labeled_indices


def coreset(trainer, batch_size = 500):
    from scipy.spatial import distance_matrix
    def greedy_k_center(labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    unl_feats, pred_certainty, unl_indices = get_feats(trainer, trainer.special_loader)
    lab_feats, _, lab_indices = get_feats(trainer, trainer.labeled_loader)

    print (unl_feats.shape)
    new_labeled_indices = trainer.unlabeled_indices[greedy_k_center(lab_feats, unl_feats, amount = batch_size)]

    print ("New {} labels are inserted".format(len(new_labeled_indices)))

    return new_labeled_indices


def random_sampling(trainer, batch_size =500):
    np.random.seed(1)
    new_labeled_indices = np.random.choice(trainer.unlabeled_indices, size = batch_size, replace = False)
    print ("New {} labels are inserted".format(len(new_labeled_indices)))
    return new_labeled_indices, [], []

def uncertainty_sampling(trainer, batch_size = 500):
    unl_feats, pred_certainty, unl_indices = get_feats(trainer, trainer.unlabeled_loader)
    new_labeled_indices = unl_indices[np.argsort(pred_certainty)[:batch_size]]
    print ("New {} labels are inserted".format(len(new_labeled_indices)))
    return new_labeled_indices


def Entropy_sampling(trainer, batch_size = 500):
    model = trainer.dis
    model.train()

    data_loader = trainer.special_loader

    unlabeled_predictions, unl_indices = [], []

    for i, (images, labels, indices) in enumerate(data_loader.get_iter()):
        images = Variable(images.cuda())

        predictions = model(images).detach().cpu().numpy()

        if not len(unlabeled_predictions):
            unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-4), axis=1)
        else:
            unlabeled_predictions = np.append(unlabeled_predictions, np.sum(predictions * np.log(predictions + 1e-5), axis=1))

        unl_indices += indices.tolist()

    unl_indices = np.array(unl_indices)
    new_labeled_indices = unl_indices[np.argpartition(unlabeled_predictions, batch_size)[:batch_size]]
    return new_labeled_indices


def KMedian(trainer, batch_size = 500):
    unl_feats, pred_certainty, unl_indices = get_feats(trainer, trainer.unlabeled_loader)

    sample_indices = np.random.choice(np.arange(len(unl_indices)), size = len(unl_indices), replace = False)
    sample_feats = unl_feats[sample_indices]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = batch_size)
    cluster_labels = kmeans.fit_predict(sample_feats)
    k_medians = []
    for i in range(batch_size):
        samples_i = sample_feats[cluster_labels == i]
        center_i = kmeans.cluster_centers_[i]
        dist_ = np.sum((samples_i - center_i[np.newaxis, ])**2, axis = 1)**0.5
        centroid = np.argmin(dist_)
        k_medians += [unl_indices[sample_indices[cluster_labels == i][centroid]]]

    new_labeled_indices = k_medians[:]

    print ("New {} labels are inserted".format(len(new_labeled_indices)))

    return new_labeled_indices

def DRAL_ours(trainer, batch_size = 500, threshold = 0.1):
    unl_feats, pred_certainty, unl_indices = get_feats(trainer, trainer.special_loader)
    lab_feats, _, lab_indices = get_feats(trainer, trainer.labeled_loader)

    print ("The number of unlabeled indices is", len(unl_indices))

    dmn_trainer = DMN_trainer()
    
    new_unl_weight = dmn_trainer.train(unl_feats, lab_feats, trainer.weight[unl_indices], trainer.weight[lab_indices])
    if np.sum(new_unl_weight.cpu().detach().numpy() < threshold) < batch_size:
        new_labeled_indices = np.random.choice(unl_indices, batch_size, replace = False)
    else:
        new_labeled_indices = np.random.choice(unl_indices[new_unl_weight.cpu().detach().numpy() < threshold], size = batch_size, replace = False)
    
    print ("The number of low weight instances is", threshold, np.sum(new_unl_weight.cpu().detach().numpy() < threshold))

    new_propagate_indices = unl_indices[new_unl_weight.cpu().detach().numpy() > threshold]

    new_propagate_labels = np.argmax(pred_certainty[new_unl_weight.cpu().detach().numpy() > threshold], 1)

    print ("New {} labels are inserted".format(len(new_labeled_indices)))
    print ("New {} labels are propagated".format(len(new_propagate_indices)))

    np.savetxt('weight_dmn_iter_{}.txt'.format(trainer.iter_cnt), new_unl_weight[np.argsort(unl_indices)].detach().cpu().numpy())
    np.savetxt('certainty_dmn_iter_{}.txt'.format(trainer.iter_cnt), pred_certainty[np.argsort(unl_indices)])

    return new_labeled_indices, new_propagate_indices, new_propagate_labels


