import argparse
import os
import sys
import timeit
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable


import config
from data_loaders import get_loaders
from models import image_model as model
import label_candidate_selection as alm
from utils import *

debug = False

DIRECTORY_PATH = os.getcwd()

class Trainer(object):

    def __init__(self, config, args):
        self.config = config
        if debug: self.config.label_period = 1000
        for k, v in args.__dict__.items():
            setattr(self.config, k, v)

        setattr(self.config, 'size_labeled_data', args.num_labels)
        setattr(self.config, 'method',args.method)
        setattr(self.config, 'save_dir', './cifar_intermidiate_result/ours/'.format(self.config.dataset, self.config.method))
        
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        
        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        sys.stdout.write(disp_str)
        sys.stdout.flush()

        self.labeled_loader, self.unlabeled_loader, self.dev_loader, self.special_loader, self.dataset = get_loaders.get_cifar_loaders(
            config, load_indices = False)

        self.dis = model.Discriminative(config).cuda()
        self.gen = model.Generator(image_size=config.image_size, noise_size=config.noise_size).cuda()
        
        self.dis = nn.DataParallel(self.dis)
        self.gen = nn.DataParallel(self.gen)
        
        #don't need to set indices if load pretrained
        self.load_pretrained()
        # self.set_indices()

        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=config.dis_lr, betas=(0.5, 0.999))
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=config.gen_lr, betas=(0.0, 0.999))

        self.d_criterion = nn.CrossEntropyLoss().cuda()

        cuda = True if torch.cuda.is_available() else False
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        log_path = os.path.join(self.config.save_dir, '{}.FM+VI.{}.txt'.format(self.config.dataset, self.config.suffix))
        self.logger = open(log_path, 'wb')
        self.logger.write(str.encode(disp_str))

        self.temporal_pred = np.zeros((50000, 10))
        
        self.weight = torch.ones(len(self.unlabeled_loader)).cuda()
        self.mask = np.zeros(50000)
        
        self.propagate_flag = False
        self.label_iter = 1

        #self.set_indices()
        
        print(self.dis)

    def _get_vis_images(self, labels):
        labels = labels.data.cpu()
        vis_images = self.special_set.index_select(0, labels)
        return vis_images

    def to_categorical(self, y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y.cpu().detach().numpy().astype(int)] = 1.

        return Variable(self.FloatTensor(y_cat))

    def load_pretrained(self):
        path = "./labeled_indices_init/ours/"
        self.dis.load_state_dict(torch.load(path+"dis_cifar_init_longrun.pth"))
        self.dis.eval()
        
        self.gen.load_state_dict(torch.load(path+"gen_cifar_init_longrun.pth"))
        self.gen.eval()
        
        self.labeled_indices = np.loadtxt(path+"labeled_indices_cifar_500.txt").astype(int)

        mask = np.zeros(len(self.dataset), dtype = bool)
        mask[self.labeled_indices] = True
        self.unlabeled_indices = np.arange(len(self.dataset))[~mask]

        self.labeled_loader = get_loaders.DataLoader(self.config, self.dataset, self.labeled_indices, self.config.train_batch_size)
        self.unlabeled_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset)), self.config.train_batch_size)
        self.special_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset))[~mask], self.config.train_batch_size)
 
        print("The pretrainer model at {} is loaded, and the numbers of unlabeled instances and labeled instance are {}, {}".format(path, len(self.unlabeled_loader), len(self.labeled_loader)))

    def update_indices(self, labeled_indices):
        log_path = self.config.log_path
        method = self.config.method
        intermidiate_path = "./{}/{}/labeled_indices_{}.txt".format(log_path, method, len(labeled_indices))
        np.savetxt(self.intermidiate_path, labeled_indices)

    def set_indices(self):
        
        intermidiate_path = "./{}/{}/labeled_indices_{}.txt".format(self.config.log_path, self.config.method, len(self.labeled_indices))
        
        self.labeled_indices = np.loadtxt(intermidiate_path).astype(int)
        self.mask[self.labeled_indices] = 1
        self.unlabeled_indices = np.arange(50000)[self.mask == 0]
    
    def _train(self, labeled=None, vis=False):
        config = self.config
        self.dis.train()
        self.gen.train()

        ##### train Dis
        lab_images, lab_labels, lab_indices = self.labeled_loader.next()
        lab_images, lab_labels = Variable(lab_images.cuda()), Variable(lab_labels.cuda())

        unl_images, unl_labels, unl_indices = self.unlabeled_loader.next()
        unl_images = Variable(unl_images.cuda())

        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        lab_logits = self.dis(lab_images)
        unl_logits = self.dis(unl_images)
        gen_logits = self.dis(gen_images.detach())

        # Standard classification loss
        lab_loss = self.d_criterion(lab_logits, lab_labels)

        # GAN true-fake loss: sumexp(logits) is seen as the input to the sigmoid
        unl_logsumexp = log_sum_exp(unl_logits)
        gen_logsumexp = log_sum_exp(gen_logits)

        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
        unl_loss = true_loss + fake_loss

        dmn_unl_feats = self.dis(unl_images.detach(), feat=True)
        dmn_lab_feats = self.dis(lab_images.detach(), feat=True)

        dmn_loss = 0
        pred_lab = torch.max(unl_logits, 1)[1]
        cnt = 0
        for c in range(10):
            if sum(lab_labels == c) * sum(pred_lab == c) > 0:
                dmn_loss += torch.mean(torch.abs(torch.mean(dmn_lab_feats[lab_labels == c], 0) - torch.mean(dmn_unl_feats[pred_lab == c], 0)))
                cnt += 1
            else:
                dmn_loss += 0

        alpha = 1.0
        d_loss = lab_loss + unl_loss + alpha * dmn_loss

        self.temporal_pred[unl_indices] = torch.softmax(unl_logits,dim=1).detach().cpu().numpy()
        
        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen
        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        # Feature matching loss
        unl_feat = self.dis(unl_images, feat=True)
        gen_feat = self.dis(gen_images, feat=True)
        fm_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))

        # Generator loss
        g_loss = fm_loss

        self.gen_optimizer.zero_grad()
        g_loss.backward()
        self.gen_optimizer.step()

        monitor_dict = OrderedDict([
            ('lab loss', lab_loss.item()),
            ('unl loss', unl_loss.item()),
            ("dmn_loss", dmn_loss.item()),
            ('fm loss', fm_loss.item())
        ])

        return monitor_dict

    def eval_true_fake(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()

        cnt = 0
        unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
        for i, (images, _, __) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda())
            noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda())

            unl_feat = self.dis(images, feat=True)
            gen_feat = self.dis(self.gen(noise), feat=True)

            unl_logits = self.dis.module.out_net(unl_feat)
            gen_logits = self.dis.module.out_net(gen_feat)

            unl_logsumexp = log_sum_exp(unl_logits)
            gen_logsumexp = log_sum_exp(gen_logits)

            ##### Monitoring (eval mode)
            # true-fake accuracy
            unl_acc += torch.mean(torch.sigmoid(unl_logsumexp).gt(0.5).float()).item()
            gen_acc += torch.mean(torch.sigmoid(gen_logsumexp).gt(0.5).float()).item()
            # top-1 logit compared to 0: to verify Assumption (2) and (3)
            max_unl_acc += torch.mean(unl_logits.max(1)[0].gt(0.0).float()).item()
            max_gen_acc += torch.mean(gen_logits.max(1)[0].gt(0.0).float()).item()

            cnt += 1
            if max_batch is not None and i >= max_batch - 1: break

        return unl_acc / cnt, gen_acc / cnt, max_unl_acc / cnt, max_gen_acc / cnt

    def eval(self, data_loader, max_batch=None, eval_gen=True):
        self.gen.eval()
        self.dis.eval()

        loss, incorrect, cnt = 0, 0, 0

        for i, (images, labels, indices) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            pred_prob = self.dis(images)

            if eval_gen:
                noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda())
                gen_images = self.gen(noise)

            loss += self.d_criterion(pred_prob, labels).item()
            cnt += 1
            incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum()
            if max_batch is not None and i >= max_batch - 1: break
        return loss / cnt, incorrect

    def visualize(self):
        self.gen.eval()
        self.dis.eval()

        vis_size = 100
        noise = Variable(torch.Tensor(vis_size, self.config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        
    def save_result(self):

        intermidiate_path = "./cifar_intermidiate_result/{}/".format(self.config.method)
        torch.save(self.gen.state_dict(),
                   os.path.join(intermidiate_path, 'gen_{}_cifar_{}_{}.pth'.format(self.config.method, len(self.labeled_indices), self.iter_cnt)))
        torch.save(self.dis.state_dict(),
                   os.path.join(intermidiate_path, 'dis_{}_cifar_{}_{}.pth'.format(self.config.method, len(self.labeled_indices), self.iter_cnt)))
        
        np.savetxt("./{}/{}/cifar_pseudo_labels_{}.txt".format(self.config.log_path, self.config.method, len(self.labeled_indices)),self.temporal_pred)

    def param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)

            return func

        lab_images, lab_labels, _ = self.labeled_loader.next()
        lab_images, lab_labels = Variable(lab_images.cuda()), Variable(lab_labels.cuda()).float()

        self.gen.apply(func_gen(True))
        noise = Variable(torch.Tensor(lab_images.size(0), self.config.noise_size).uniform_().cuda()).float()
        self.gen.apply(func_gen(False))

        self.dis.apply(func_gen(True))
        self.dis.apply(func_gen(False))
    
    def train(self):
        time0 = timeit.default_timer()
        config = self.config

        self.iter_cnt = 0
        iter, min_dev_incorrect = 0, 1e6
        monitor = OrderedDict()

        batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
        min_lr = config.min_lr if hasattr(config, 'min_lr') else 0.0
        self.epoch = iter // batch_per_epoch
        while True:
            self.epoch = iter // batch_per_epoch
            if iter % batch_per_epoch == 0:
                epoch_ratio = float(self.epoch) / float(config.max_epochs)
                # use another outer max to prevent any float computation precision problem
                self.dis_optimizer.param_groups[0]['lr'] = max(min_lr, config.dis_lr * min(3. * (1. - epoch_ratio), 1.))
                self.gen_optimizer.param_groups[0]['lr'] = max(min_lr, config.gen_lr * min(3. * (1. - epoch_ratio), 1.))

            iter_vals = self._train()

            for k, v in iter_vals.items():
                if k not in monitor:
                    monitor[k] = 0.
                monitor[k] += v

            if iter % config.vis_period == 0:
                self.visualize()

            if iter % config.eval_period == 0:
                train_loss, train_incorrect = self.eval(self.special_loader)
                dev_loss, dev_incorrect = self.eval(self.dev_loader, eval_gen=False)

                train_incorrect = train_incorrect.float() / (1.0 * len(self.special_loader))
                dev_incorrect = dev_incorrect.float() / (1.0 * len(self.dev_loader))
                min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)

                disp_str = '#{}\ttrain: {:.4f}, {:.4f} | dev: {:.4f}, {:.4f} | best: {:.4f}'.format(
                    iter, train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, (v + 0.0) / config.eval_period)

                disp_str += ' | lr: {:.5f}'.format(self.dis_optimizer.param_groups[0]['lr'])


                propagate_count = 0
                propagate_correct_count = 0
                for i in np.arange(len(self.temporal_pred)):
                    if i not in self.labeled_indices and np.max(self.temporal_pred[i]) > 0.95:
                        propagate_count += 1
                        if self.dataset[i][1] == np.argmax(self.temporal_pred[i]):
                            propagate_correct_count += 1
                disp_str += ' | propagate count: {:.4f}, propagate correct: {:.4f}'.format(propagate_count, propagate_correct_count)
                disp_str += ' | time: {:4f}'.format(int(timeit.default_timer() - time0))
                disp_str += '\n'

                monitor = OrderedDict()

                self.logger.write(str.encode(disp_str))
                sys.stdout.write(disp_str)
                sys.stdout.flush()


            if iter > -1 and iter % config.label_period == 0:
                
                self.save_result()
                
                print ("Start adding labels.")
                if iter == 0: flag = False
                else: flag = True

                if self.label_iter == 1:
                    batch_size = 7000
                elif self.label_iter == 2:
                    batch_size = 15000
                elif self.label_iter == 3:
                    batch_size = 15000
                else:
                    break

                print ("Batch Size:", batch_size)

                self.label_iter += 1
                
                new_labeled_indices, new_propagate_indices, new_propagate_labels = alm.DRAL_ours(self, batch_size = batch_size)
                
                self.labeled_indices = np.append(self.labeled_indices, new_labeled_indices)
                
                intermidiate_path = "./{}/{}/labeled_indices_{}.txt".format(self.config.log_path, self.config.method, len(self.labeled_indices))
                np.savetxt(intermidiate_path, self.labeled_indices)

                mask = np.zeros(len(self.dataset), dtype = bool)
                mask[self.labeled_indices] = 1
                self.unlabeled_indices = np.arange(len(self.dataset))[~mask]
                print (len(self.labeled_indices))
                self.labeled_loader = get_loaders.DataLoader(self.config, self.dataset, self.labeled_indices, config.train_batch_size)
                self.unlabeled_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset)), config.train_batch_size)
                self.special_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset))[~mask], config.train_batch_size)
                
                train_loss, train_incorrect = self.eval(self.special_loader)
                print ("The original accuracy of unlabeled data is:", (train_incorrect+0.0)/len(self.special_loader))

                print ("The data loader is updated, the number of labeled data is {}, and the number of unlabeled data is {}.".format(len(self.labeled_loader), len(self.unlabeled_loader)))

                sys.stdout.flush()

            iter += 1
            self.iter_cnt += 1

            if self.epoch > self.config.max_epochs:
                self.save_result()
                break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar_trainer.py')
    parser.add_argument('-suffix', default='run0', type=str, help="Suffix added to the save images.")
    parser.add_argument('-num_labels', type=int, help="num of labels")
    parser.add_argument('-method', default='ours', type=str, help="Method")

    args = parser.parse_args()

    trainer = Trainer(config.cifar_config(), args)
    trainer.train()


