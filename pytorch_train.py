import matplotlib
import time
import sys
from glob import glob
from os.path import abspath, dirname, isdir, isfile, join
from utils import bifloss as LOSS2
from os import makedirs, fsync
import os
from models import MODELS
from utils.extract_patches import get_data_training
import torch.nn as nn
import numpy as np
from torchsummary import summary
import torch.optim as optim
from utils.CL_pixel_loss import ConLoss
sys.path.insert(0, './utils/')
from utils.Data_loader import Retina_loader
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.cuda import empty_cache

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

matplotlib.use("Agg")
import configparser

import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="nasopharyngeal training")
parser.add_argument('--mode', default='gpu', type=str, metavar='train on gpu or cpu',
                    help='train on gpu or cpu(default gpu)')
parser.add_argument('--gpu', default=0, type=int, help='gpu number')
parser.add_argument('--optimizer', default='Adam',
                    choices=['Adam', 'SGD'],
                    help='loss: ' +
                         ' | '.join(['Adam', 'SGD']) +
                         ' (default: Adam)')
parser.add_argument('--finetuning', type=str2bool, default=False,
                    help='is fine tuning')
parser.add_argument('--decay', type=float, default=1e-6,
                    help='decay of learning process')
parser.add_argument('--printfreq', type=int, default=1,
                    help='printfreq show training loss')
parser.add_argument('--itersize', type=int, default=100,
                    help='itersize of learning process')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--checkpoint-interval', type=int, default=10,
                    help='checkpoint interval')
args = parser.parse_args()

gpuid = args.gpu
mode = args.mode
decay = args.decay
itersize = args.itersize
printfreq = args.printfreq
checkpoint_interval = args.checkpoint_interval
finetuning = args.finetuning

config = configparser.ConfigParser()
# ========= Load settings from Config file
config.read('configuration.txt')
algorithm = config.get('experiment name', 'name')
dataset = config.get('data attributes', 'dataset')
log_path_experiment = './log/experiments/' + algorithm + '/' + dataset + '/'

# ========= Load settings from Config file
path_data = './' + dataset + config.get('data paths', 'path_local')
model_path = config.get('data paths', 'model_path')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
inp_shape = (int(config.get('data attributes', 'patch_width')), int(config.get('data attributes', 'patch_height')), 1)
lr = float(config.get('training settings', 'lr'))
cl_loss = str2bool(config.get('training settings', 'CL_loss'))

THIS_DIR = abspath(dirname(log_path_experiment))
TMP_DIR = log_path_experiment
if not isdir(TMP_DIR):
    makedirs(TMP_DIR)

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
log = Logger(join(TMP_DIR, algorithm + '-log.txt'))
# log
sys.stdout = log
print('[i] Data name:            ', dataset)
print('[i] epochs:               ', N_epochs)
print('[i] Batch size:           ', batch_size)
print('[i] algoritm:             ', algorithm)
print('[i] gpu:                  ', args.gpu)
print('[i] mode:                 ', args.mode)
print('[i] learning rate:        ', lr)
print('[i] optimizer:            ', args.optimizer)
print('[i] finetuning:           ', finetuning)
fcn = True

tensorboardPath = TMP_DIR + "/tensorboard"

def to_cuda(t, mode='gpu'):
    if mode == 'gpu':
        return t.cuda()
    return t


def main():
    torch.manual_seed(0)

    if algorithm == 'UNet' or algorithm == 'DUNet':
        model = MODELS[algorithm](n_channels=1, n_classes=1)
    elif algorithm == 'AttUNet':
        model = MODELS[algorithm](img_ch=1, output_ch=1)
    elif algorithm == 'UKAN':
        model = MODELS[algorithm](in_chans=1, num_classes=1, img_size=inp_shape[0])
    elif algorithm == 'DSCNet':
        model = MODELS[algorithm](n_channels=1, n_classes=1, kernel_size=5, extend_scope=1, if_offset=True
                                  , device='cuda', number=4, dim=1)
    elif algorithm == 'AttUKAN':
        model = MODELS[algorithm](in_chans=1, num_classes=1, img_size=inp_shape[0])
    elif algorithm == 'RollingUNet':
        model = MODELS[algorithm](input_channels = 1,num_classes = 1)
    elif algorithm == 'mambaUNet':
        model = MODELS[algorithm](input_nc=1, num_classes=1)
    elif algorithm == 'CTFNet':
        model = MODELS[algorithm](num_classes=1,inplanes=1)
    elif algorithm == 'BCDUNet':
        model = MODELS[algorithm](input_dim=1, output_dim=1)
    elif algorithm == 'IterNet':
        model = MODELS[algorithm](n_channels=1, n_classes=1)
    elif algorithm == 'unet++':
        model = MODELS[algorithm](in_channel=1,out_channel=1)


    if finetuning:
        #weight_files = sorted(glob(join(TMP_DIR, 'checkpoint_epoch_*.pth')), reverse=True)
        log_path= './log/experiments/' + algorithm + '/'
        pretrain = config.get('data attributes', 'pretrain')
        weight_path = log_path + pretrain +'/'
        weight_files = join(weight_path, 'best.pth')
        print("loaded:" + weight_files)
        model, _ = load_pretrained(model, weight_files)
        #global lr
        #lr = 1e-4

    print('lr:' + str(lr))
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              weight_decay=decay, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()

    if mode == 'gpu':
        torch.cuda.set_device(gpuid)
        torch.cuda.manual_seed(0)
        model.cuda()
        #summary(model, input_size=(1, inp_shape[0], inp_shape[1]))

    summary_writer = SummaryWriter(tensorboardPath)

    path_data = './' + dataset + config.get('data paths', 'path_local')
    patches_imgs_train, patches_masks_train = get_data_training(
        train_imgs_original=path_data + dataset + config.get('data paths', 'train_imgs_original'),
        train_groudTruth=path_data + dataset + config.get('data paths', 'train_groundTruth'),  # masks
        patch_height=inp_shape[0],
        patch_width=inp_shape[1],
        N_subimgs=int(config.get('training settings', 'N_subimgs')),
        inside_FOV=config.getboolean('training settings', 'inside_FOV'),
        fcn=fcn
    )

    patches_imgs_train = np.transpose(patches_imgs_train, (0, 2, 3, 1))

    if fcn:
        patches_masks_train = np.transpose(patches_masks_train, (0, 2, 3, 1))
        train_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.8, split='train')
        test_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.8, split='test')
    else:
        train_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.8, split='train', fcn=False)
        test_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.8, split='test', fcn=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    min_val = np.inf
    for epoch in range(N_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, epoch, cl_loss, algorithm)
        test_loss, test_acc, testval = test(model, test_loader, epoch, cl_loss, algorithm)
        if testval < min_val:
            min_val = testval
            save_file = "{}/best.pth".format(TMP_DIR)
            save_file = "{}/checkpoint_epoch_{}_{:.4f}_dice.pth".format(TMP_DIR, epoch+1, min_val)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}, filename=save_file)
        save_file_final = "{}/checkpoint_epoch_{}.pth".format(TMP_DIR, epoch + 1)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}, filename=save_file_final)
        summary_writer.add_scalar(
            'train/loss', train_loss, epoch)
        summary_writer.add_scalar(
            'train/acc', train_acc, epoch)
        scheduler.step(test_loss)
        summary_writer.add_scalar(
            'vali/loss', test_loss, epoch)
        summary_writer.add_scalar(
            'vali/acc', test_acc, epoch)
        log.flush()  # write log
    summary_writer.close()


def train(model, train_loader, optimizer, epoch, cl_loss, algorithm):
    model.train()
    if mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
    global net_vis
    end = time.time()
    pend = time.time()
    batch_time = Averagvalue()
    printfreq_time = Averagvalue()
    losses = Averagvalue()
    acc = Averagvalue()
    dicemetric = SoftDiceLoss()
    jsloss = SoftIoULoss(n_classes=1)
    cldiceloss = clDiceLoss()

    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_loader):

        image = dtype_float(to_cuda(image.float(), mode)).requires_grad_(False)
        label = to_cuda(label, mode).requires_grad_(False)

        if cl_loss:
            cl_feature_all = []
            label_all = []
            pre_label, cl_feature = model(image)

            label_all.append(label)
            cl_feature_all.append(cl_feature)
        else:
            pre_label = model(image)

        if fcn:
            if algorithm == 'UNet' or algorithm == 'DUNet':
                loss = to_cuda(BCELoss(pre_label, label),'gpu')
            elif algorithm == 'AttUNet':
                loss = to_cuda(BCELoss(pre_label, label),'gpu')
            elif algorithm == 'UKAN':
                loss = to_cuda(BCELoss(pre_label, label),'gpu') * 0.8 + to_cuda(dicemetric(pre_label, label),'gpu') * 1.0 + to_cuda(jsloss(pre_label, label), 'gpu') * 0.2
            elif algorithm == 'DSCNet':
                loss = to_cuda(cldiceloss(pre_label, label),'gpu')
            elif algorithm == 'AttUKAN':
                loss = to_cuda(BCELoss(pre_label, label),'gpu') * 0.8 
                loss += to_cuda(dicemetric(pre_label, label),'gpu') * 1.0 
                loss += to_cuda(jsloss(pre_label, label), 'gpu') * 0.2
            elif algorithm == 'RollingUNet':
                loss = to_cuda(dicemetric(pre_label, label),'gpu') + to_cuda(BCELoss(pre_label, label),'gpu')
            elif algorithm == 'mambaUNet':
                loss = to_cuda(dicemetric(pre_label, label),'gpu') + to_cuda(BCELoss(pre_label, label),'gpu')
            elif algorithm == 'CTFNet':
                loss = to_cuda(BCELoss(pre_label, label),'gpu')
            elif algorithm == 'IterNet':
                loss = to_cuda(BCELoss(pre_label, label),'gpu')
            elif algorithm == 'UNet++':
                loss = to_cuda(dicemetric(pre_label, label),'gpu') + to_cuda(BCELoss(pre_label, label),'gpu')
            elif algorithm == 'BCDUNet':
                loss = to_cuda(BCELoss(pre_label, label),'gpu') # to_cuda(BCELoss())

            if cl_loss:
                cl_feature_all = torch.cat(cl_feature_all, dim=0)
                label_all = torch.cat(label_all, dim=0)
                loss += to_cuda(ConLoss(cl_feature_all, label_all) * 0.3,'gpu')
            prec1 = accuracy_check(pre_label, label)
            acc.update(prec1, 1)
        else:
            loss = CELoss(pre_label, label)
            prec1 = accuracy(pre_label, label)
            acc.update(prec1[0].item(), image.size(0))
        losses.update(loss.item(), image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % (int(len(train_loader) / printfreq)) == 0:
            printfreq_time.update(time.time() - pend)
            pend = time.time()
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, N_epochs, i, len(train_loader)) + \
                   'printfreq time {printfreq_time.val:.3f} (avg:{printfreq_time.avg:.3f}) '.format(
                       printfreq_time=printfreq_time)
            print(info)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    empty_cache()
    return losses.avg, acc.avg


def test(model, test_loader, epoch, cl_loss, algorithm):
    model.eval()
    epoch_time = Averagvalue()
    losses = Averagvalue()
    acc = Averagvalue()
    test_val = Averagvalue()
    end = time.time()
    dicemetric = SoftDiceLoss()
    jsloss = SoftIoULoss(n_classes=1)
    cldiceloss = clDiceLoss()

    if mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = dtype_float(to_cuda(image.float(), mode)).requires_grad_(False)
            label = to_cuda(label, mode).requires_grad_(False)
            if cl_loss:
                cl_feature_all = []
                label_all = []
                pre_label, cl_feature = model(image)

                label_all.append(label)
                cl_feature_all.append(cl_feature)
            else:
                pre_label = model(image)
                
            if fcn:
                if algorithm == 'UNet' or algorithm == 'DUNet':
                    loss = to_cuda(BCELoss(pre_label, label),'gpu')
                elif algorithm == 'AttUNet':
                    loss = to_cuda(BCELoss(pre_label, label),'gpu')
                elif algorithm == 'UKAN':
                    loss = to_cuda(BCELoss(pre_label, label),'gpu') * 0.8 + to_cuda(dicemetric(pre_label, label),'gpu') * 1.0 + to_cuda(jsloss(pre_label, label), 'gpu') * 0.2
                elif algorithm == 'DSCNet':
                    loss = to_cuda(cldiceloss(pre_label, label),'gpu')
                elif algorithm == 'AttUKAN' :
                    loss = to_cuda(BCELoss(pre_label, label),'gpu') * 0.8 
                    loss += to_cuda(dicemetric(pre_label, label),'gpu') * 1.0 
                    loss += to_cuda(jsloss(pre_label, label), 'gpu') * 0.2
                elif algorithm == 'RollingUNet':
                    loss =  to_cuda(dicemetric(pre_label, label),'gpu') + to_cuda(BCELoss(pre_label, label),'gpu')
                elif algorithm == 'mambaUNet':
                    loss =  to_cuda(dicemetric(pre_label, label),'gpu') + to_cuda(BCELoss(pre_label, label),'gpu')
                elif algorithm == 'CTFNet':
                    loss = to_cuda(BCELoss(pre_label, label),'gpu')+ to_cuda(dicemetric(pre_label, label),'gpu')
                elif algorithm == 'IterNet':
                    loss = to_cuda(BCELoss(pre_label, label),'gpu')
                elif algorithm == 'UNet++':
                    loss =  to_cuda(dicemetric(pre_label, label),'gpu')
                elif algorithm == 'BCDUNet':
                    loss = to_cuda(BCELoss(pre_label, label),'gpu')

                if cl_loss :
                    cl_feature_all = torch.cat(cl_feature_all, dim=0)
                    label_all = torch.cat(label_all, dim=0)
                    loss += to_cuda(ConLoss(cl_feature_all, label_all) * 0.3, 'gpu')
                prec1 = accuracy_check(pre_label, label)
                acc.update(prec1, 1)
            else:
                loss = CELoss(pre_label, label)
                prec1 = accuracy(pre_label, label)
                acc.update(prec1[0].item(), image.size(0))
            losses.update(loss.item(), image.size(0))
            test_val.update(dicemetric(pre_label, label).item(), image.size(0))

        empty_cache()
        # measure elapsed time
    epoch_time.update(time.time() - end)
    info = 'TEST Epoch: [{0}/{1}]'.format(epoch, N_epochs) + \
           'Test Epoch Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=epoch_time) + \
           'Acc {acc.val:f} (avg:{acc.avg:f}) '.format(acc=acc) + \
           'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
           'Dice {loss.val:f} (avg:{loss.avg:f}) '.format(loss=test_val)
    print(info)
    return losses.avg, acc.avg, test_val.avg




def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.cpu().detach().numpy()
        np_ims.append(item)
    compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.div(batch_size))
    return res




class Averagvalue(object):
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


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        # target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target - (pred * target)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


class clDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth

    def soft_cldice_loss(self, pred, target, target_skeleton=None):
        '''
        inputs shape  (batch, channel, height, width).
        calculate clDice loss
        Because pred and target at moment of loss calculation will be a torch tensors
        it is preferable to calculate target_skeleton on the step of batch forming,
        when it will be in numpy array format by means of opencv
        '''
        cl_pred = self.soft_skeletonize(pred)
        if target_skeleton is None:
            target_skeleton = self.soft_skeletonize(target)
        iflat = self.norm_intersection(cl_pred, target)
        tflat = self.norm_intersection(target_skeleton, pred)
        intersection = (iflat * tflat).sum()
        return 1. - (2. * intersection) / (iflat + tflat).sum()

    def dice_loss(self, pred, target):
        '''
        inputs shape  (batch, channel, height, width).
        calculate dice loss per batch and channel of sample.
        E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
        '''
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

    def soft_skeletonize(self, x, thresh_width=10):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        '''
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def norm_intersection(self, center_line, vessel):
        '''
        inputs shape  (batch, channel, height, width)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        '''
        smooth = 1.
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)

    def forward(self, pred, target):
        return 0.8 * self.dice_loss(pred, target) + 0.2 * self.soft_cldice_loss(pred, target)


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        # probs = F.sigmoid(logits)
        probs = logits
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

    def dice_loss(self, pred, target):
        '''
        inputs shape  (batch, channel, height, width).
        calculate dice loss per batch and channel of sample.
        E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
        '''
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

    def soft_skeletonize(self, x, thresh_width=10):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        '''
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def norm_intersection(self, center_line, vessel):
        '''
        inputs shape  (batch, channel, height, width)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        '''
        smooth = 1.
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)

    def forward(self, pred, target):
        return 0.8 * self.dice_loss(pred, target) + 0.2 * self.soft_cldice_loss(pred, target)


def CELoss(y, label):
    loss = nn.CrossEntropyLoss()
    return loss(y, label)


def BCELoss(prediction, label):
    masks_probs_flat = prediction.view(-1)
    true_masks_flat = label.float().view(-1)
    loss = nn.BCELoss()(masks_probs_flat, true_masks_flat)
    return loss


def D25_Loss(x,y):
    n,c,h,w=x.shape
    #x = torch.tensor([item.cpu().detach().numpy() for item in x]).cuda()
    #y = torch.tensor([item.cpu().detach().numpy() for item in y]).cuda()
    #x = float(x)
    #y = float(y)
    x = torch.sigmoid(x.float())
    y = torch.sigmoid(y.float())
    one=torch.ones_like(x)
    zero=torch.zeros_like(x)
    D25Mat=to_cuda(torch.ones(c,1,5,5),'gpu')#suppose x is n*c*h*w
    #get region connectivity
    #x:predict label y:true label
    C_x=torch.max(torch.min(0.0173*F.conv2d(x,D25Mat,padding=2,groups=1)**1.9956-0.0258,one),zero)
    C_x = C_x ** 2 * x
    C_y=torch.max(torch.min(0.0173*F.conv2d(y,D25Mat,padding=2,groups=1)**1.9956-0.0258,one),zero)
    C_y = C_y ** 2 * y
    result= (-1) * (torch.exp(1 - C_y) * y * torch.log(x + 1e-12) + torch.exp(1 - C_x) * (1 - y) * torch.log(1 - x + 1e-12))
    return torch.mean(result)


if __name__ == '__main__':
    main()


