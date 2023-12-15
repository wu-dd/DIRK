import copy
import os
import time
import argparse
import torch
import random
import logging
from torch.backends import cudnn
import torch.nn.functional as F
import torch.nn as nn

from dataset import *
from model import *
from resnet import *
from utils.utils_loss import *
from utils.util import *


parser = argparse.ArgumentParser(description='DIRK')
# global set
parser.add_argument('--dataset', default='cifar10',choices=['cifar10','kmnist','fmnist','mnist','cifar100'], type=str)
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--method', default='DIRK', type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--seed', default=3407, type=int)
# optimization
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--m', type=float, default=0.99)
# PLL setting
parser.add_argument('--rate', default=1.0, type=float)
parser.add_argument('--weight',type=float,default=1.0)
# REF setting
parser.add_argument('--queue',type=int,default=4096)
parser.add_argument('--dist_temp',type=float,default=0.4)
parser.add_argument('--feat_temp',type=float,default=0.07)
parser.add_argument('--prot_start',type=int,default=1)

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])
torch.set_printoptions(linewidth=2000)

def main():
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(args.__dict__)
    main_worker(args)

def main_worker(args):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load dataloader
    logging.info("=> creating loader '{}'".format(args.dataset))
    train_loader, test_loader, num_class = get_loader(args)
    args.num_class=num_class
    logging.info('=> Average number of partial labels: {}'.format(
        train_loader.dataset.given_partial_label_matrix.sum() / len(train_loader.dataset)))

    # load model
    logging.info("=> Creating model '{}'".format(args.arch))
    Teacher = conTea(args,SupConResNet).cuda()
    Teacher.encoder=torch.nn.DataParallel(Teacher.encoder)
    Student = conStu(args, SupConResNet).cuda()
    Student.encoder = torch.nn.DataParallel(Student.encoder)
    S_optimizer = torch.optim.SGD(Student.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)

    # set contrastive loss function
    loss_cont_fn = WeightedConLoss(temperature=args.feat_temp,dist_temprature=args.dist_temp)

    logging.info('=> Start Training')
    best_acc1 = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(args, S_optimizer, epoch)
        start_upd_prot = epoch >= args.prot_start
        teach_loss, cont_loss=train(train_loader,Teacher, Student, S_optimizer,loss_cont_fn,epoch,args,start_upd_prot)
        logging.info("[Training-Epoch {}]:teach_loss:{}\tcont_loss:{}".format(epoch,teach_loss,cont_loss))
        val_acc = test(args, epoch, test_loader, Student)
        best_acc1 = max(best_acc1, val_acc)
        logging.info("[Testing-Epoch {}]:val_acc: {}".format(epoch, val_acc))


def train(train_loader,Teacher, Student, S_optimizer,loss_cont_fn,epoch,args, start_upd_prot=False):
    teach_losses=AverageMeter('Teaching_Loss',':.2f')
    con_losses=AverageMeter('Con_Loss',':.2f')

    # switch to train mode
    Student.train()
    Teacher.train()

    for i,(img_w,img_s,img_distill,partY,target,index) in enumerate(train_loader):
        img_w, img_s, img_distill, partY, target, index = img_w.cuda(), img_s.cuda(), img_distill.cuda(), partY.cuda(), target.cuda(), index.cuda()
        # obtain pools
        features, partYs, dists, rec_conf_t=Teacher(img_w,img_s,img_distill,partY,target=target.unsqueeze(1))
        # obtain Student's output and feature
        output_s,feat_s=Student(img_s,img_distill)
        # bind features and partial distribution
        features_cont = torch.cat((feat_s, features), dim=0)
        partY_cont = torch.cat((partY, partYs), dim=0)
        dist_cont = torch.cat((rec_conf_t, dists), dim=0)

        batch_size=output_s.shape[0]
        mask_partial = torch.matmul(partY_cont[:batch_size], partY_cont.T) #
        mask_partial[mask_partial != 0] = 1
        _, pseudo_target = torch.max(dist_cont, dim=1) #
        pseudo_target = pseudo_target.contiguous().view(-1, 1)
        mask_pseudo_target = torch.eq(pseudo_target[:batch_size], pseudo_target.T).float() #

        if start_upd_prot:
            mask = mask_partial * mask_pseudo_target
        else:
            mask = None

        # contrastive loss
        if args.weight!=0:
            loss_cont = loss_cont_fn(features=features_cont, dist=dist_cont, partY=partY_cont,mask=mask,epoch=epoch, args=args,batch_size=partY.shape[0])
        else:
            loss_cont=torch.tensor(0.0).cuda()
        # teaching loss
        loss_teach=CE_loss(output_s,rec_conf_t)
        # total loss
        loss=loss_teach+args.weight*loss_cont

        teach_losses.update(loss_teach.item(),partY.size(0))
        con_losses.update(loss_cont.item(),partY.size(0))

        # compute gradient and do SGD step
        S_optimizer.zero_grad()
        loss.backward()
        S_optimizer.step()

        momentum_model(Teacher, Student, args.m)

    return teach_losses.avg, con_losses.avg

def momentum_model(model_tea, model_stu, momentum=0.5):
    for param_tea, param_stu in zip(model_tea.parameters(), model_stu.parameters()):
        param_tea.data = param_tea.data * momentum + param_stu.data * (1 - momentum)

def adjust_learning_rate(args, optimizer, epoch):
    import math
    lr = args.lr
    eta_min=lr * (args.lr_decay_rate**3)
    lr=eta_min+(lr-eta_min)*(
        1+math.cos(math.pi*epoch/args.epochs))/2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    logging.info('LR: {}'.format(lr))


if __name__ == '__main__':
    main()

