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

import pprint
import random
import numpy as np
import torch
from torch import nn, optim
from config import batch_size_dict
from models.model import Generator, Encoder, PredictiveModelCombinedModel
from utils.logging_utils import summarize_architecture
import torchvision.models as pretrained_models
from models.baselines_pretrained import PretrainedPredictiveModel


class PretrainedSession(object):
    def __init__(self,
                 c=None,
                 model=None):

        assert c is not None

        self.c = c
        self.model = model
        self.alpha = 1.0
        self.phase = int(np.log2(self.c.image_size / 4))
        self.cur_resolution = 4 * 2 ** self.phase
        self.batch_size = batch_size_dict(self.cur_resolution)

        self.criterion_train = nn.L1Loss().cuda() if self.c.training_criteria == 'l1' else nn.L2Loss().cuda()
        self.criterion_evaluation = nn.L1Loss().cuda() if self.c.evaluation_criteria == 'l1' else nn.L2Loss().cuda()

        # Create model
        if self.c.experiment_type == 'pretrain_only':

            print("Creating full model from pre-trained model '{}'".format(
                self.c.arch))
            pretrained_model = pretrained_models.__dict__[self.c.arch](pretrained=True)

        # elif self.c.experiment_type == 'encoder_only':
        #     pretrained_model = nn.DataParallel(Encoder(num_latents=self.c.nz, n_label=self.c.n_label, c=self.c).cuda())
        #     pretrained_model = self.load_encoder_weights(self.c.load_checkpoint_dir, pretrained_model)
        else:
            raise ValueError("Need to specify either pretrained model or checkpointed combined model")

        self.model = PretrainedPredictiveModel(pretrained_model,
                                               # self.c.arch,
                                               # self.c.experiment_type,
                                               c=self.c)
        self.model = self.model.cuda()

        if self.c.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.regressor.parameters()),
                self.c.lr_pretrained,
                momentum=self.c.momentum,
                weight_decay=self.c.weight_decay)
        elif self.c.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.regressor.parameters()),
                self.c.lr_pretrained,
                betas=self.c.adam_optim_betas_pretrain)

        else:
            raise ValueError("Optimizer not specified.")

        if self.c.summarize_architecture:
            print("Encoder Network Structure")
            max_reso = 4 * (2 ** c.max_phase)
            summarize_architecture(self.encoder, input_size=(3, max_reso, max_reso), phase=c.max_phase)
            print("\n\nGenerator Network Structure")
            self.network_summary = summarize_architecture(self.generator, input_size=[(c.nz,), (c.n_label,)], phase=c.max_phase)

        print('Session created.')

        pprint.pprint(c)

    def load_encoder_weights(self, path, pretrained_model):
        checkpoint = torch.load(path)

        print('loading Encoder model from %s' % path)
        state_dict = checkpoint['D_state_dict']
        pretrained_model.load_state_dict(state_dict)

        # We freeze these weights
        if self.c.freeze_encoder_weights:
            for param in pretrained_model.parameters():
                param.freeze_model = False
        print("Done loading weights...")
        return pretrained_model

    def setup(self):
        # if self.c.save_output_log:
        #     self.log_config()
        random.seed(self.c.random_seed)
        torch.manual_seed(self.c.random_seed)
        torch.cuda.manual_seed_all(self.c.random_seed)


class CombinedTrainSession(object):
    def __init__(self,
                 c=None,
                 use_default_config=True):

        assert c is not None

        self.c = c
        self.init_combined_session()

        if self.c.load_checkpoint:
            self.load(self.c.load_checkpoint_dir)
            # self.reset_opt()


        if self.c.summarize_architecture:
            print("Encoder Network Structure")
            max_reso = 4 * (2 ** c.max_phase)
            summarize_architecture(self.encoder, input_size=(3, max_reso, max_reso), phase=c.max_phase)
            print("\n\nGenerator Network Structure")
            self.network_summary = summarize_architecture(self.generator, input_size=[(c.nz,), (c.n_label,)], phase=c.max_phase)

        print('Session created.')

        pprint.pprint(c)

    def init_combined_session(self):

        self.alpha = -1
        self.phase = self.c.start_phase
        self.sample_i = self.phase * self.c.images_per_stage
        self.cur_resolution = 4 * 2 ** self.phase
        self.batch_size = batch_size_dict(self.cur_resolution)

        self.generator = nn.DataParallel(Generator(self.c.nz, self.c.n_label, c=self.c).cuda())
        self.encoder = nn.DataParallel(Encoder(num_latents=self.c.nz, n_label=self.c.n_label, c=self.c).cuda())
        self.predictor = nn.DataParallel(PredictiveModelCombinedModel(c=self.c)).cuda()

        self.reset_optimizers()

    def reset_optimizers(self):
        self.optimizerG = optim.Adam(self.generator.parameters(), self.c.g_lr, betas=self.c.opt_betas, eps=self.c.opt_epsilon)
        self.optimizerE = optim.Adam(self.encoder.parameters(), self.c.e_lr, betas=self.c.opt_betas, eps=self.c.opt_epsilon)  # includes all the encoder parameters...
        # self.optimizerP = optim.Adam(self.predictor.parameters(), self.c.p_lr, betas=self.c.opt_betas_predictor, eps=self.c.opt_epsilon)  # includes all the encoder parameters...
        # leaving off epsilon term
        self.optimizerP = optim.Adam(self.predictor.parameters(), self.c.p_lr, betas=self.c.opt_betas_predictor)  # includes all the encoder parameters...

        if self.c.p_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.predictor.parameters(),
                self.c.p_lr,
                momentum=self.c.momentum,
                weight_decay=self.c.weight_decay)
        elif self.c.p_optimizer == 'adam':
            self.optimizer = self.optimizerP = optim.Adam(self.predictor.parameters(), self.c.p_lr, betas=self.c.opt_betas_predictor)  # includes all the encoder parameters...

    def setup(self):
        # if self.c.save_output_log:
        #     self.log_config()
        random.seed(self.c.random_seed)
        torch.manual_seed(self.c.random_seed)
        torch.cuda.manual_seed_all(self.c.random_seed)

    def save_all(self, path):
        if not self.c.use_cometML == True:
            self.c.comet_experiment_id = None
        else:
            # TODO: get experiment id for continuous logging
            self.c.comet_experiment_id = None

        torch.save({'G_state_dict': self.generator.state_dict(),
                    'D_state_dict': self.encoder.state_dict(),
                    'P_state_dict': self.predictor.state_dict(),

                    'optimizerE': self.optimizerE.state_dict(),
                    'optimizerG': self.optimizerG.state_dict(),
                    'optimizerP': self.optimizerP.state_dict(),
                    'iteration': self.sample_i,
                    'phase': self.phase,
                    'alpha': self.alpha,
                    'comet_key': self.c.comet_experiment_id},
        path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.sample_i = int(checkpoint['iteration'])
        self.alpha = checkpoint['alpha']
        self.phase = int(checkpoint['phase'])
        if not self.c.use_checkpoint_sample_i:  # If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
            # self.phase = min(self.c.start_phase, self.phase)
            if self.c.start_phase != self.phase:
                self.phase = self.c.start_phase
            if self.c.force_sample_i != 0:
                self.sample_i = self.c.force_sample_i
            else:
                self.sample_i = int(self.phase * self.c.images_per_stage)

        print("Starting phase: {}".format(self.phase))
        print('Starting from iteration {}'.format(self.sample_i))

        if self.phase > self.c.max_phase:
            print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, self.c.max_phase))
            self.phase = self.c.max_phase

        self.cur_resolution = 4 * 2 ** self.phase

        self.generator.load_state_dict(checkpoint['G_state_dict'], strict=self.c.load_state_dict_strict)
        self.encoder.load_state_dict(checkpoint['D_state_dict'], strict=self.c.load_state_dict_strict)
        try:
            self.predictor.load_state_dict(checkpoint['P_state_dict'], strict=self.c.load_state_dict_strict)
        except:
            pass

        if not self.c.reset_optimizers:
            self.optimizerE.load_state_dict(checkpoint['optimizerE'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            self.optimizerP.load_state_dict(checkpoint['optimizerP'])
            print("Using optimizers from checkpoint.")
        else:
            print("Using new optimizers.")

    def update_hyperparameters(self):
        if self.phase >= 1:
            self.c.match_x_scale = 4.0
            self.c.match_x_fade_in_scale = 4.0
