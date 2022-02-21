# -*- coding: utf-8 -*-
'''
        Project: Product Aesthetic Design: A Machine Learning Augmentation
        Authors: Alex Burnap, Yale University
        Email: alex.burnap@yale.edu

        License: MIT License

        OSS Code Attribution (see Licensing Inheritance):
        Portions of Code From or Modified from Open Source Projects:
        https://github.com/tkarras/progressive_growing_of_gans
        https://github.com/AaltoVision/pioneer
        https://github.com/DmitryUlyanov/AGE
        https://github.com/akanimax/attn_gan_pytorch/
'''
import random

import torch
import torch.nn
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import normalize
import numpy as np
from config import c

break_string = '######################################################################'

def init_linear(linear):
    nn.init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    nn.init.kaiming_normal_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

def freeze_model(model, flag=True):
    flag = not flag
    for p in model.parameters():
        p.requires_grad = flag

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0, p1) / np.sqrt(np.dot(p0, p0)) / np.sqrt(np.dot(p1, p1)))
    k1 = np.sin((1 - t) * omega) / np.sin(omega)
    k2 = np.sin(t * omega) / np.sin(omega)
    return k1 * p0 + k2 * p1

def split_attributes_out(z):
    if c.conditional_model:
        label = z[:, -c.n_label:]
    else:
        #label = torch.unsqueeze(z[:, -c.n_label:], dim=1)
        label = z[:, -c.n_label:]
    return z[:, :c.nz], label

def split_mask_out_of_generated(x_hat):
    if x_hat.size(1) == 4:
        image = x_hat[:, :3, :, :]
        mask = x_hat[:, 3, :, :]
    elif x_hat.size(1) == 2:
        image = x_hat[:, 0, :, :].unsqueeze(1)
        mask = x_hat[:, 1, :, :].unsqueeze(1)
    else:
        raise ValueError
    return image, mask


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

# def normalize(x, p=2, dim=1, eps=1e-9):
#     '''
#     Projects points to a sphere.
#     '''
#     zn = x.norm(2, dim=dim).add(eps)
#     zn = zn.unsqueeze(1)
#     return x.div(zn).expand_as(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        val = float(val)
        n= float(n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def untransform(images):
    '''
    Transform from floating point BGR to uint8 RGB for visualization.
    '''
    images = images.transpose(1, 2, 0)
    images = (images + 1.0) * 127.5
    return images.astype('uint8')

def renormalize(images):
    '''
    Using Numpy
    Transform from floating point BGR to uint8 RGB for visualization.
    '''
    #     t = np.array((t/t.max()+1)* 127.5).astype(np.uint8)
    images = np.array((images + 1.0) * 127.5).astype(np.uint8)
    #     t = t/t.max()
    images = images.swapaxes(3, 1).swapaxes(2, 1)
    return images

def renormalize_single(image):
    #     t = np.array((t/t.max()+1)* 127.5).astype(np.uint8)
    image = np.array((image + 1.0) * 127.5).astype(np.uint8)
    #     t = t/t.max()
    image = image.transpose(1, 2, 0)
    return image

# class ImagePool():
#     def __init__(self, pool_size):
#         self.pool_size = pool_size
#         if self.pool_size > 0:
#             self.num_imgs = 0
#             self.images = []
#
#     def query(self, images):
#         if self.pool_size == 0:
#             return images
#         return_images = []
#         for image in images:
#             image = torch.unsqueeze(image, 0)
#             if self.num_imgs < self.pool_size:
#                 self.num_imgs = self.num_imgs + 1
#                 self.images.append(image)
#                 return_images.append(image)
#             else:
#                 p = random.uniform(0, 1)
#                 if p < c.generated_image_pool_threshold:
#                     random_id = random.randint(0, self.pool_size - 1)
#                     tmp = self.images[random_id].clone()
#                     self.images[random_id] = image
#                     return_images.append(tmp)
#                 else:
#                     return_images.append(image)
#         return_images = Variable(torch.cat(return_images, 0))
#         return return_images
#
# def populate_z(z, nz, noise, batch_size):
#     '''
#     Fills noise variable `z` with noise U(S^M) [from https://github.com/DmitryUlyanov/AGE ]
#     '''
#     with torch.no_grad():
#         z.resize_(batch_size, nz)  # , 1, 1)
#         z.data.normal_(0, 1)
#         if noise == 'sphere':
#             normalize_(z.data)
#

#
# def sample_h_prior(h, noise, batch_size):
#     # h = h.resize_(batch_size, num_h)  # , 1, 1)
#     # h = torch.rand_like(h)
#     h.data = h.data.normal_(0, 1)
#     if noise == 'sphere':
#         h = normalize_(h, p=2, dim=1)
#     # return h



# def calc_samples_var(x, dim=0):
#     '''
#     Calculates variance. [from https://github.com/DmitryUlyanov/AGE ]
#     '''
#     x_zero_meaned = x - x.mean(dim).expand_as(x)
#     return x_zero_meaned.pow(2).mean(dim)


# def switch_grad_updates_to_first_of(a, b):
#     freeze_model(a, True)
#     freeze_model(b, False)


# def normalize_(x, dim=1):
#     '''
#     Projects points to a sphere inplace.
#     '''
#     zn = x.norm(2, dim=dim).add(c.norm_epsilon)
#     zn = zn.unsqueeze(1)
#     x = x.div_(zn)
#     x.expand_as(x)

# # Sample from the Gumbel-Softmax distribution and optionally discretize.
# class GumbelSoftmax(nn.Module):
#     '''
#     https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Layers.py
#     '''
#     def __init__(self):
#         super(GumbelSoftmax, self).__init__()
#         # self.logits = nn.Linear(f_dim, c_dim)
#         # self.f_dim = f_dim
#         # self.c_dim = c_dim
#
#     def sample_gumbel(self, shape, is_cuda=True, eps=1e-20):
#         U = torch.rand(shape)
#         if is_cuda:
#             U = U.cuda()
#         return -torch.log(-torch.log(U + eps) + eps)
#
#     def gumbel_softmax_sample(self, logits, temperature):
#         y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
#         return F.softmax(y / temperature, dim=-1)
#
#     def gumbel_softmax(self, logits, temperature, hard=False):
#         """
#         ST-gumple-softmax
#         input: [*, n_class]
#         return: flatten --> [*, n_class] an one-hot vector
#         """
#         # categorical_dim = 10
#         y = self.gumbel_softmax_sample(logits, temperature)
#
#         if not hard:
#             return y
#
#         shape = y.size()
#         _, ind = y.max(dim=-1)
#         y_hard = torch.zeros_like(y).view(-1, shape[-1])
#         y_hard.scatter_(1, ind.view(-1, 1), 1)
#         y_hard = y_hard.view(*shape)
#         # Set gradients w.r.t. y_hard gradients w.r.t. y
#         y_hard = (y_hard - y).detach() + y
#         return y_hard
#
#     def forward(self, x, temperature=1.0, hard=False):
#         logits = self.logits(x).view(-1, self.c_dim)
#         prob = F.softmax(logits, dim=-1)
#         y = self.gumbel_softmax(logits, temperature, hard)
#         return logits, prob, y


# class ResBlock(nn.Module):
#     def __init__(self, channel_num):
#         super(ResBlock, self).__init__()
#         self.block = nn.Sequential(
#             conv3x3(channel_num, channel_num),
#             nn.BatchNorm2d(channel_num),
#             nn.ReLU(True),
#             conv3x3(channel_num, channel_num),
#             nn.BatchNorm2d(channel_num))
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = x
#         out = self.block(x)
#         out += residual
#         out = self.relu(out)
#         return out
