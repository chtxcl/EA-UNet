import os
import argparse
import time
from glob import glob
from collections import OrderedDict
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import Dataset
from metrics import iou_score
import losses
from utils import str2bool, count_params
import pandas as pd
import torch.nn.functional as F
from Nets import EA_UNet
from torchstat import stat


arch_names = list(EA_UNet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')#

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='EA_UNet',
                        choices=arch_names,\

                        help='model architecture: ' +
                            ' | '.join(arch_names) +

                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="XXX_2019_Data_2",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3,  type=float,     metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_gamma',       default=0.99,         type=float,     help='learning rate gamma')
    parser.add_argument('--momentum',       default=0.9,          type=float,     help='momentum')
    parser.add_argument('--weight-decay',   default=1e-4,         type=float,     help='weight decay')
    parser.add_argument('--nesterov',       default=False,        type=str2bool,  help='nesterov')

    args = parser.parse_args()

    return args

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


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()

        target = target.cuda()

        # compute output
        if args.deepsupervision:#深层监督
            outputs = model(input)
            loss = 0
            for output in outputs:
             loss += criterion(output, target)
             loss /= len(outputs)
             iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log

def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()

            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)

                loss = criterion(output, target)
                iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()


    # Data loading code
    img_paths = glob(r'D:/BraTs/2_BrainData/trainImage/*')
    mask_paths = glob(r'D:/BraTs/2_BrainData/trainMask/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))

    # create model

    print("=> creating model %s" %args.arch)
    print(args.arch)

    model = EA_UNet.__dict__[args.arch](args)
    model = model.cuda()

    print(count_params(model))

    # print(parameter_count_table(model))

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)
        print('parameter: %s' %count_params(model), file=f)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_gamma)

    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_iou = 0
    trigger = 0
    val_loss = 1

    #Total_start_time=time.clock()
    for epoch in range(args.epochs):

        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        print("train start:")
        train_start = time.perf_counter()
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        train_end = time.perf_counter()
        print('Train time: %s s'%(train_end-train_start))

        print("------------------------------------------------------")
        print('本轮learning rate:%.6f'%(optimizer.state_dict()['param_groups'][0]['lr']))
        print("------------------------------------------------------")

        lr_scheduler.step()
        # evaluate on validation set
        print("validate start:")
        validation_start = time.perf_counter()
        val_log = validate(args, val_loader, model, criterion)
        validation_end = time.perf_counter()
        print('Validate time: %s s' % (validation_end - validation_start))

        print('loss %.4f ------ iou %.4f ------- val_loss %.4f -------- val_iou %.4f'
            %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            optimizer.state_dict()['param_groups'][0]['lr'],
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log._append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)
        trigger += 1

        epoch= epoch+1
        if val_log['loss'] < val_loss:
            torch.save(model.state_dict(), 'models/%s/Epoch-%d----Train_Loss--%.4f----Val_Loss--%.4f.pth' %(args.name,epoch,train_log['loss'],val_log['loss']))
            val_loss = val_log['loss']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()

    log_csv_path = 'models/%s/log.csv' % args.name
    draw_train_loss = []
    draw_val_loss = []

    # 读取log.csv的文件信息
    with open(log_csv_path)as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            draw_train_loss.append(row[2])

    with open(log_csv_path)as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            draw_val_loss.append(row[4])

    draw_train_loss.remove(draw_train_loss[0])
    draw_val_loss.remove(draw_val_loss[0])

    New_train_loss = []
    New_val_loss = []

    for i in range(len(draw_train_loss)):
        z = float(draw_train_loss[i])
        New_train_loss.append(z)

    for i in range(len(draw_val_loss)):
        z = float(draw_val_loss[i])
        New_val_loss.append(z)

    plt.plot(New_train_loss, label='Train_Loss')
    plt.plot(New_val_loss, label='Val_Loss')
    plt.title('%s' % args.arch)
    plt.legend()
    plt.savefig('models/%s/Loss' % args.name)


if __name__ == '__main__':
    main()

