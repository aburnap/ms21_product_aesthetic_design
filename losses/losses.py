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
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import config
c = config.c

class RatingLoss(torch.nn.Module):
    def __init__(self):
        super(RatingLoss, self).__init__()
        self.criterion_train = nn.L1Loss().cuda() if c.match_y_metric == 'L1' else nn.L2Loss().cuda()
        # self.criterion_evaluation = nn.L1Loss().cuda() if self.c.evaluation_criteria == 'l1' else nn.L2Loss().cuda()

    def forward(self, x1, x2):
        return self.criterion_train(x1, x2)

def calc_loss(x1, x2, dist):
    if dist == 'L2':
        return (x1 - x2).pow(2).mean(dim=1).mean()
    elif dist == 'L1':
        return (x1 - x2).abs().mean(dim=1).mean()
    elif dist == 'cos':
        cosine_distance = 1.0 - torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=c.epsilon)
        return cosine_distance.mean()

def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy

def cross_entropy(logits, targets):
    """Cross-Entropy loss
        loss = (1/n) * -Σ targets*log(predicted)
    Args:
        logits: (array) corresponding array containing the logits of the categorical variable
        real: (array) corresponding array containing the true labels

    Returns:
        output: (array/float) depending on average parameters the result will be the mean
                              of all the sample losses or an array with the losses per sample
    """
    log_q = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(targets * log_q, dim=-1))


class DensityRatioApproxKL(torch.nn.Module):

    def __init__(self):
        super(DensityRatioApproxKL, self).__init__()

    def empirical_var(self, x, dim=0):
        x_centered = x - x.mean(dim).expand_as(x)
        return x_centered.pow(2).mean(dim)

    def forward(self, samples, return_mu_and_std=True):

        assert samples.nelement() == samples.size(1) * samples.size(0)

        samples = samples.view(samples.size(0), -1)

        samples_var = self.empirical_var(samples)
        samples_logvar = samples_var.log()
        samples_mean = samples.mean(0)

        KL = -0.5 * torch.mean(1 + samples_logvar - samples_mean.pow(2) - samples_var)

        if return_mu_and_std:
            return KL, samples_mean, samples_logvar
        else:
            return KL


class ChairAttributeLoss(torch.nn.Module):
    '''
    Chair loss is just for 3 attributes for viewpoint in spherical coordinates
    The angles are specified by three values: (theta, phi, rho)
    Paper Link: https://www.di.ens.fr/willow/research/seeing3Dchairs/texts/Aubry14.pdf

    We will primarily penalize theta since it is the X-Y plane rotation viewpoint (i.e., around Z-axis)
    Phi only takes two values in the dataset, 20 and 30 degrees, so we will penalize less
    Rho is same radius for all chairs in dataset, so we will not penalize.

    Also, I want to make sure we don't penalize close angles but parametrized incorrectly, so will use sin and cosine to account for angular difference
    '''

    def __init__(self):
        super(ChairAttributeLoss, self).__init__()
        self.msqe_loss = nn.MSELoss()

    def forward(self, label_hat, label_true):
        phi_loss = self.msqe_loss(label_hat[:, 0], label_true[:, 0])
        # theta_loss = self.msqe_loss(label_hat[:, 1], label_true[:, 1])
        rho_loss = self.msqe_loss(label_hat[:, 2] * math.pi / 180.0, label_true[:, 2] * math.pi / 180.0)  # since this is in angles while the other is in radians

        theta_loss_sin = self.msqe_loss(torch.sin(label_hat[:, 1] * math.pi / 180.0), torch.sin(label_true[:, 1]) * math.pi / 180.0)
        theta_loss_cos = self.msqe_loss(torch.cos(label_hat[:, 1] * math.pi / 180.0), torch.cos(label_true[:, 1]) * math.pi / 180.0)

        theta_loss = theta_loss_sin + theta_loss_cos

        # Combine according to weight rotation viewpoint more heavily
        total_loss = 0.15 * phi_loss + 0.85 * theta_loss + 0.0 * rho_loss
        total_loss /= 10.0  # to keep consistent with vehicle attribute loss, we also scale it lower in config file
        return total_loss


class MaskLoss(torch.nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, mask_hat, mask_true):
        if len(mask_true.size()) == 4:
            mask_true = mask_true.squeeze(1)

        mask_hat = torch.clamp(mask_hat, 0, 1)  # TODO Alex added Jan 22, 2022 for A100 GPUs - device-side CUDA runtime assert
        loss = self.bce_loss(mask_hat, mask_true)
        if loss > c.epsilon:
            return loss
        else:
            return 0.0 * loss + c.epsilon


class AttributeLossWithGumbelSoftmax(torch.nn.Module):

    def __init__(self):
        super(AttributeLossWithGumbelSoftmax, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.msqe_loss = nn.MSELoss()
        self.temperature = c.gumbel_softmax_temperature

        # years = data_labels['arr_0']  # 15 years
        # makes = data_labels['arr_1']  # 48 makes
        # models = data_labels['arr_2']  # 23 models
        # bodytypes = data_labels['arr_3']  # 20 bodytypes
        # views = data_labels['arr_4']  # 2 variables for viewpoint
        # design_ids = data_labels['arr_5']  # 2 variables for viewpoint
        # colors = data_labels['arr_6']  # 3 variables for viewpoint
        self.label_inds = [15, 48, 23, 20, 2, 3]
        # 15, 63, 86, 106, 108, 111

    def sample_gumbel(self, shape, is_cuda=True, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / self.temperature, dim=1)

    def forward(self, label_hat, label_true):

        year_logits = label_hat[:, :15]
        make_logits = label_hat[:, 15:63]
        model_logits = label_hat[:, 63:86]
        bodytype_logits = label_hat[:, 86:106]
        view_logits = label_hat[:, 106:108]
        color_logits = label_hat[:, 108:]

        year_probs = self.gumbel_softmax_sample(year_logits[:, :15])
        make_probs = self.gumbel_softmax_sample(make_logits[:, 15:63])
        model_probs = self.gumbel_softmax_sample(model_logits[:, 63:86])
        bodytype_probs = self.gumbel_softmax_sample(bodytype_logits[:, 86:106])

        # year_entropy = log_q = F.log_softmax(year_logits, dim=-1)
        # -torch.mean(torch.sum(targets * log_q, dim=-1))
        # year_entropy =
        # make_entropy =
        # model_entropy =
        # bodytype_entropy =


        year_loss = self.cross_entropy_loss(year_probs, label_true[:, :15].argmax(dim=1))
        make_loss = self.cross_entropy_loss(make_probs, label_true[:, 15:63].argmax(dim=1))
        model_loss = self.cross_entropy_loss(model_probs, label_true[:, 63:86].argmax(dim=1))
        bodytype_loss = self.cross_entropy_loss(bodytype_probs, label_true[:, 86:106].argmax(dim=1))

        view_loss = self.msqe_loss(view_logits, label_true[:, 106:108])
        color_loss = self.msqe_loss(color_logits, label_true[:, 108:])

        # Combine according to what is likely important
        total_loss = 0.5 * year_loss + 0.5 * make_loss + 0.05 * model_loss + 2.0 * bodytype_loss + 5.0 * view_loss + 2.0 * color_loss
        total_loss /= 10.0
        return total_loss

class VehicleAttributeLoss(torch.nn.Module):

    def __init__(self):
        super(VehicleAttributeLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.msqe_loss = nn.MSELoss()

        # years = data_labels['arr_0']  # 15 years
        # makes = data_labels['arr_1']  # 48 makes
        # models = data_labels['arr_2']  # 23 models
        # bodytypes = data_labels['arr_3']  # 20 bodytypes
        # views = data_labels['arr_4']  # 2 variables for viewpoint
        # design_ids = data_labels['arr_5']  # 2 variables for viewpoint
        # colors = data_labels['arr_6']  # 3 variables for viewpoint
        self.label_inds = [15, 48, 23, 20, 2, 3]
        # 15, 63, 86, 106, 108, 111
    def forward(self, label_hat, label_true):

        year_loss = self.cross_entropy_loss(label_hat[:,:15], label_true[:, :15].argmax(dim=1))
        make_loss = self.cross_entropy_loss(label_hat[:, 15:63], label_true[:, 15:63].argmax(dim=1))
        model_loss = self.cross_entropy_loss(label_hat[:, 63:86], label_true[:, 63:86].argmax(dim=1))
        bodytype_loss = self.cross_entropy_loss(label_hat[:, 86:106], label_true[:, 86:106].argmax(dim=1))
        view_loss = self.msqe_loss(label_hat[:, 106:108], label_true[:, 106:108])
        color_loss = self.msqe_loss(label_hat[:, 108:], label_true[:, 108:])

        total_loss = 0.5 * year_loss + 0.5 * make_loss + 0.05 * model_loss + 2.0 * bodytype_loss + 5.0 * view_loss + 2.0 * color_loss
        total_loss /= 10.0
        return total_loss
