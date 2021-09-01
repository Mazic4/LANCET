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
#from torch.utils.tensorboard import SummaryWriter


import config
from data_loaders import get_loaders
from models import image_model as model
import label_candidate_selection as alm
from utils import *


class Trainer(object):

    def __init__(self, config, args):
        self.config = config
        for k, v in args.__dict__.items():
            setattr(self.config, k, v)
        if config.sys_label:
            self.config.size_labeled_data = 300
            config.size_labeled_data = 300

        setattr(self.config, 'size_labeled_data', args.num_labels)
        setattr(self.config, 'method',args.method)
        setattr(self.config, 'save_dir', '/home/zhanghuayi01/lancet/LANCET/svhn_intermidiate_result/ours/')

        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        sys.stdout.write(disp_str)
        sys.stdout.flush()
        
        self.labeled_loader, self.unlabeled_loader,  self.dev_loader, self.special_loader, self.dataset = get_loaders.get_svhn_loaders(self.config)
        
        self.weight = torch.ones(len(self.unlabeled_loader)).cuda()
        self.unl_dmn_certainty = torch.ones(len(self.unlabeled_loader)).cuda()

        self.dis = model.Discriminative(config).cuda()
        self.gen = model.Generator(image_size=config.image_size, noise_size=config.noise_size).cuda()
        
        self.dis = nn.DataParallel(self.dis)
        self.gen = nn.DataParallel(self.gen)
       
        #distribtion matching network
        self.dmn = model.Distribution_Matching_Network().cuda()
        self.dmn_optimizer = optim.Adam(self.dmn.parameters(), lr=config.dis_lr, betas=(0.5, 0.999))

        # don't need to set indices if load pretrained
        self.load_pretrained()
        #self.set_indices()

        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=config.dis_lr, betas=(0.5, 0.999)) # 0.0 0.9999
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=config.gen_lr, betas=(0.0, 0.999)) # 0.0 0.9999

        self.d_criterion = nn.CrossEntropyLoss(reduction='none')
        
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        log_path = os.path.join(self.config.save_dir, '{}.FM+PT+ENT.{}.txt'.format(self.config.dataset, self.config.suffix))
        self.logger = open(log_path, 'wb')
        self.logger.write(disp_str.encode())

        self.temporal_pred = np.zeros((len(self.dataset),10))
        self.label_iter = 3
    
    def set_indices(self):
        intermidiate_path = "./{}/{}/svhn_labeled_indices_{}.txt".format(self.config.log_path, self.config.method, self.config.size_labeled_data)
        self.labeled_indices = np.loadtxt(intermidiate_path).astype(int)
        self.unlabeled_indices = np.arange(len(self.unlabeled_loader))

    def _get_vis_images(self, labels):
        labels = labels.data.cpu()
        vis_images = self.special_set.index_select(0, labels)
        return vis_images

    def load_pretrained(self):
        path = "./labeled_indices_init/ours/"
        self.dis.load_state_dict(torch.load(path+"dis_svhn_init_longrun.pth"))
        self.dis.eval()

        self.gen.load_state_dict(torch.load(path+"gen_svhn_init_longrun.pth"))
        self.gen.eval()

        self.labeled_indices = np.loadtxt(path+"labeled_indices_svhn_300.txt").astype(int)

        mask = np.zeros(len(self.dataset), dtype = bool)
        mask[self.labeled_indices] = True
        self.unlabeled_indices = np.arange(len(self.dataset))[~mask]

        self.labeled_loader = get_loaders.DataLoader(self.config, self.dataset, self.labeled_indices, self.config.train_batch_size)
        self.unlabeled_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset)), self.config.train_batch_size)
        self.special_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset))[~mask], self.config.train_batch_size)

        print("The pretrainer model at {} is laoded, and the numbers of unlabeled instances and labeled instance are {}, {}".format(path, len(self.unlabeled_loader), len(self.labeled_loader)))

    def _train(self, labeled=None, vis=False):
        config = self.config
        self.dis.train()
        self.gen.train()
        ##### train Dis
        lab_images, lab_labels, lab_indice = self.labeled_loader.next()
        lab_images, lab_labels = Variable(lab_images.cuda()), Variable(lab_labels.cuda())

        unl_images, _, unl_indices = self.unlabeled_loader.next()
        unl_images = Variable(unl_images.cuda())

        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)
        
        lab_logits = self.dis(lab_images)
        unl_logits = self.dis(unl_images)
        gen_logits = self.dis(gen_images.detach())

        # Standard classification loss
        lab_loss = self.d_criterion(lab_logits, lab_labels).mean()

        # GAN true-fake loss: sumexp(logits) is seen as the input to the sigmoid
        unl_logsumexp = log_sum_exp(unl_logits)
        gen_logsumexp = log_sum_exp(gen_logits)

        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
        unl_loss = true_loss + fake_loss
        
        beta = 1 - np.exp(-self.iter_cnt/7300)
        pseudo_labels = torch.max(unl_logits.softmax(1),1)[1]
        entropy_loss = F.cross_entropy(unl_logits, pseudo_labels)
        
        dmn_unl_feats = self.dis(unl_images.detach(), feat=True)
        dmn_lab_feats = self.dis(lab_images.detach(), feat=True)
        
        dmn_loss = 0
        pred_lab = torch.max(unl_logits, 1)[1]
        for c in range(10):
            if sum(lab_labels == c) * sum(pred_lab == c) > 0:
                dmn_loss += torch.mean(torch.abs(torch.mean(dmn_lab_feats[lab_labels == c], 0) - torch.mean(dmn_unl_feats[pred_lab == c], 0)))
            else:
                dmn_loss += 0
        
        d_loss = lab_loss + unl_loss + dmn_loss

        ##### Monitoring (train mode)
        # true-fake accuracy
        unl_acc = torch.mean(nn.functional.sigmoid(unl_logsumexp.detach()).gt(0.5).float())
        gen_acc = torch.mean(nn.functional.sigmoid(gen_logsumexp.detach()).gt(0.5).float())
        # top-1 logit compared to 0: to verify Assumption (2) and (3)
        max_unl_acc = torch.mean(unl_logits.max(1)[0].detach().gt(0.0).float())
        max_gen_acc = torch.mean(gen_logits.max(1)[0].detach().gt(0.0).float())

        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen and Enc
        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        # Feature matching loss
        unl_feat = self.dis(unl_images, feat=True)
        gen_feat = self.dis(gen_images, feat=True)
        fm_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))
        
        # Entropy loss via feature pull-away term

        # Generator loss
        g_loss = fm_loss
         
        self.gen_optimizer.zero_grad()
        g_loss.backward()
        self.gen_optimizer.step()

        self.temporal_pred[unl_indices] = torch.softmax(unl_logits,dim=1).detach().cpu().numpy()

        monitor_dict = OrderedDict([
                       ('unl acc' , unl_acc.item()), 
                       ('gen acc' , gen_acc.item()), 
                       ('max unl acc' , max_unl_acc.item()), 
                       ('max gen acc' , max_gen_acc.item()), 
                       ('lab loss' , lab_loss.item()),
                       ('unl loss' , unl_loss.item()),
                       #('ent loss' , ent_loss.data[0]),
                       ('fm loss' , fm_loss.item()),
                       ("entropy loss", entropy_loss.item()),
                       ('dmn_loss', dmn_loss.item())
                   ])
                
        return monitor_dict
    
    def train_dmn(self, lab_feats, unl_feats, lab_indices):
        for _ in range(10):
            dmn_unl_logits = self.dmn(unl_feats.detach())
            dmn_lab_logits = self.dmn(lab_feats.detach())

            dmn_unl_loss = self.d_criterion(dmn_unl_logits, torch.zeros(dmn_unl_logits.shape[0]).long().cuda())

            dmn_loss_raw = F.cross_entropy(dmn_lab_logits, torch.ones(dmn_lab_logits.shape[0]).long().cuda(), reduction = 'none')
            dmn_loss = torch.mean(dmn_loss_raw * self.weight[lab_indices]) + dmn_unl_loss.mean()

            self.dmn_optimizer.zero_grad()
            dmn_loss.backward()
            self.dmn_optimizer.step()
            
            lab_certainty = torch.softmax(dmn_lab_logits, dim=1)[:,1]
            
            alpha = 0.1
            self.weight[lab_indices] -= alpha * (lab_certainty.detach()-0.5)
            self.weight[lab_indices] = torch.clamp(self.weight[lab_indices], 0, 10)
            self.weight[lab_indices] /= torch.mean(self.weight[lab_indices]).item()

        lab_certainty = torch.softmax(dmn_lab_logits, dim=1)[:,1]
        unl_certainty = torch.softmax(dmn_unl_logits, dim=1)[:,0]

        dmn_unl_logits = self.dmn(unl_feats)
        dmn_lab_logits = self.dmn(lab_feats)

        dmn_unl_loss = self.d_criterion(dmn_unl_logits, torch.zeros(dmn_unl_logits.shape[0]).long().cuda())

        dmn_loss_raw = F.cross_entropy(dmn_lab_logits, torch.ones(dmn_lab_logits.shape[0]).long().cuda(), reduction = 'none')
        dmn_loss = torch.mean(dmn_loss_raw * self.weight[lab_indices]) + dmn_unl_loss.mean()
        
        return dmn_loss, lab_certainty, unl_certainty

    def eval_true_fake(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        
        cnt = 0
        unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
        for i, (images, _, indices) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda())
            noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda(), volatile=True)

            unl_feat = self.dis(images, feat=True)
            gen_feat = self.dis(self.gen(noise), feat=True)

            unl_logits = self.dis.module.out_net(unl_feat)
            gen_logits = self.dis.module.out_net(gen_feat)

            unl_logsumexp = log_sum_exp(unl_logits)
            gen_logsumexp = log_sum_exp(gen_logits)

            ##### Monitoring (eval mode)
            # true-fake accuracy
            unl_acc += torch.mean(nn.functional.sigmoid(unl_logsumexp).gt(0.5).float()).item()
            gen_acc += torch.mean(nn.functional.sigmoid(gen_logsumexp).gt(0.5).float()).item()
            # top-1 logit compared to 0: to verify Assumption (2) and (3)
            max_unl_acc += torch.mean(unl_logits.max(1)[0].gt(0.0).float()).item()
            max_gen_acc += torch.mean(gen_logits.max(1)[0].gt(0.0).float()).item()

            cnt += 1
            if max_batch is not None and i >= max_batch - 1: break

        return unl_acc / cnt, gen_acc / cnt, max_unl_acc / cnt, max_gen_acc / cnt

    def eval(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()

        loss, incorrect, cnt = 0, 0, 0
        for i, (images, labels, indices) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            pred_prob = self.dis(images)
            loss += torch.mean(self.weight[indices]*F.cross_entropy(pred_prob, labels, reduction='none')).item()
            #loss += torch.mean(self.weight[indices]*self.d_criterion(pred_prob, labels)).item()
            cnt += 1
            incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum().item()
            if max_batch is not None and i >= max_batch - 1: break
        return loss / cnt, incorrect


    def visualize(self):
        self.gen.eval()
        self.dis.eval()

        vis_size = 100
        noise = Variable(torch.Tensor(vis_size, self.config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

    def param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(500 // self.config.train_batch_size):
            lab_images, _, indices= self.labeled_loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        self.gen.apply(func_gen(True))
        noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)
        self.gen.apply(func_gen(False))

        self.dis.apply(func_gen(True))
        logits = self.dis(Variable(images.cuda()))
        self.dis.apply(func_gen(False))

    def save_result(self):

        intermidiate_path = "/home/zhanghuayi01/lancet/LANCET/svhn_intermidiate_result/ours/"

        save_path = os.path.join(intermidiate_path, '{}.FM+VI.{}.png'.format(self.config.dataset, self.iter_cnt))
        #vutils.save_image(gen_images.data.cpu(), save_path, normalize=True, range=(-1, 1), nrow=10)
        torch.save(self.gen.state_dict(),
                   os.path.join(intermidiate_path, 'gen_{}_svhn_{}_{}.pth'.format(self.config.method, len(self.labeled_indices), self.iter_cnt)))
        torch.save(self.dis.state_dict(),
                   os.path.join(intermidiate_path, 'dis_{}_svhn_{}_{}.pth'.format(self.config.method, len(self.labeled_indices), self.iter_cnt)))

        np.savetxt(intermidiate_path + "/svhn_pseudo_labels_{}.txt".format(len(self.labeled_indices)),self.temporal_pred)

    def train(self):
        config = self.config
        self.param_init()

        self.iter_cnt = 0
        iter, min_dev_incorrect = 0, 1e6
        monitor = OrderedDict()
        
        batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
        self.epoch = iter // batch_per_epoch
        min_lr = config.min_lr if hasattr(config, 'min_lr') else 0.0
        while True:

            if iter % batch_per_epoch == 0:
                epoch = iter / batch_per_epoch
                if config.dataset != 'svhn' and epoch >= config.max_epochs:
                    break
                epoch_ratio = float(epoch) / float(config.max_epochs)
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

            if iter> 1 and iter % config.eval_period == 0:
                train_loss, train_incorrect = self.eval(self.unlabeled_loader)
                dev_loss, dev_incorrect = self.eval(self.dev_loader)

                unl_acc, gen_acc, max_unl_acc, max_gen_acc = self.eval_true_fake(self.dev_loader, 10)

                train_incorrect /= 1.0 * len(self.unlabeled_loader)
                dev_incorrect /= 1.0 * len(self.dev_loader)
                min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)

                disp_str = '#{}\ttrain: {:.4f}, {:.4f} | dev: {:.4f}, {:.4f} | best: {:.4f}'.format(
                    iter, train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / config.eval_period)
                
                disp_str += ' | [Eval] unl acc: {:.4f}, gen acc: {:.4f}, max unl acc: {:.4f}, max gen acc: {:.4f}'.format(unl_acc, gen_acc, max_unl_acc, max_gen_acc)
                disp_str += ' | lr: {:.5f}'.format(self.dis_optimizer.param_groups[0]['lr'])
                disp_str += '\n'

                monitor = OrderedDict()

                self.logger.write(disp_str.encode())
                sys.stdout.write(disp_str)
                sys.stdout.flush()

            if self.config.sys_label == True and iter > -1 and iter % self.config.label_period == 0:

                print("Start adding labels.")
                #if iter == 0:
                #    flag = False
                #else:
                #    flag = True

                if self.label_iter == 1:
                    batch_size = 14651 - 700
                elif self.label_iter == 2:
                    batch_size = 21976
                elif self.label_iter == 3:
                    batch_size = 21900
                else:
                    break

                self.label_iter += 1

                #new_labeled_indices = alm.random_sampling(trainer, batch_size = 300)
                new_labeled_indices, new_propagate_indices, new_propagate_labels = alm.DRAL_ours(trainer, batch_size = batch_size)
                self.labeled_indices = np.append(self.labeled_indices, new_labeled_indices)
                intermidiate_path = "./{}/{}/labeled_indices_{}.txt".format(self.config.log_path, self.config.method,
                                                                            len(self.labeled_indices))
                mask = np.zeros(len(self.dataset), dtype = bool)
                mask[self.labeled_indices] = 1
                self.unlabeled_indices = np.arange(len(self.dataset))[~mask]

                print (len(self.unlabeled_indices))

                np.savetxt(intermidiate_path, self.labeled_indices)
                self.labeled_loader = get_loaders.DataLoader(self.config, self.dataset, self.labeled_indices, config.train_batch_size)
                self.unlabeled_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset)), config.train_batch_size)
                self.special_loader = get_loaders.DataLoader(self.config, self.dataset, np.arange(len(self.dataset))[~mask], config.train_batch_size)

                self.save_result()
                
            iter += 1
            self.iter_cnt += 1

            if self.epoch > self.config.max_epochs:
                self.save_result()
                break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='svhn_trainer.py')
    parser.add_argument('-suffix', default='run0', type=str, help="Suffix added to the save images.")
    parser.add_argument('-num_labels', type=int, help="num of labels")
    parser.add_argument('-method', default='ours', type=str, help="Method")

    args = parser.parse_args()

    trainer = Trainer(config.svhn_config(), args)
    trainer.train()
