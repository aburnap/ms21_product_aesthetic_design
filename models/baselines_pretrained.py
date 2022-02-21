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
from __future__ import print_function
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
from models.model import Encoder
from utils.training_utils import freeze_model


class Feature_Extractor_Model(nn.Module):
    '''
    Output features from encoder.
    '''

    def __init__(self, pretrained_weight_dir=None, encoder=None, c=None):
        super(Feature_Extractor_Model, self).__init__()

        assert c is not None
        self.c = c

        assert pretrained_weight_dir is not None
        self.encoder_parameters_dir = pretrained_weight_dir
        if encoder is None:
            self.encoder = Encoder(nz=self.c.nz, n_label=self.c.n_label)
        else:
            self.encoder = encoder
        self.use_cuda = True
        self.load_discriminator_weights(self.encoder_parameters_dir)
        self.encoder.cuda()

    def load_discriminator_weights(self, path):
        checkpoint = torch.load(path)
        # TODO: This is a hack to remove module from the name

        print('loading Encoder model from %s' % path)
        state_dict = checkpoint['D_state_dict']
        state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            state_dict_rename[name] = v
        self.encoder.load_state_dict(state_dict_rename)

        # We freeze these weights
        if self.c.freeze_encoder_weights:
            for param in self.encoder.parameters():
                param.freeze_model = False
        print("Done loading weights...")
        # if self.c.encoder_train_mode: # NOTE: This was fixed in newest Pytorch release...
        #     self.D.train_combined_model()
        # else:
        #     self.D.eval()

    def _numpy2var(self, x):
        x = x.astype(np.float32)
        var = Variable(torch.from_numpy(x))
        var = var.cuda()
        return var

    def _var2numpy(self, var):
        return var.cpu().data.numpy()

    def forward(self, image):
        # image = self._numpy2var(image)
        image_features = self.encoder(image,
                                      phase=self.c.encoder_phase,
                                      alpha=self.c.encoder_alpha,
                                      labels=None,
                                      use_labels=False,
                                      return_label_hat=False,
                                      return_feature_layer=self.c.return_feature_layer)
        return image_features


class PretrainedPredictiveModel(nn.Module):
    def __init__(self,
                 pretrained_model,
                 c=None):

        super(PretrainedPredictiveModel, self).__init__()
        assert c is not None
        self.c = c

        if pretrained_model and self.c.experiment_type == 'pretrain_only':
            self.pretrained_model = pretrained_model.features

            if self.c.freeze_encoder_weights:
                freeze_model(self.pretrained_model)
        else:
            raise ValueError

        if self.c.freeze_pretrained_weights and pretrained_model:
            # Freeze pretrained weights
            # for p in self.pretrained_model.parameters():
            #     p.requires_grad = False
            freeze_model(self.pretrained_model)

        if self.c.experiment_type == 'pretrain_only':
            if self.c.image_size == 512:
                num_features = 131072
            elif self.c.image_size == 256:
                num_features = 32768
            elif self.c.image_size == 128:
                num_features = 8192
            elif self.c.image_size == 64:
                num_features = 2048
            else:
                raise ValueError

            if self.c.pretrained_model_use_attributes:
                num_features += self.c.n_label

            if self.c.pretrained_tower_batchnorm:
                self.regressor = nn.Sequential(
                    nn.Linear(num_features, self.c.num_pretrained_fc_units),
                    nn.BatchNorm1d(self.c.num_pretrained_fc_units),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.c.num_pretrained_fc_units, self.c.num_pretrained_fc_units),
                    nn.BatchNorm1d(self.c.num_pretrained_fc_units),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.c.num_pretrained_fc_units, 1),
                )
            else:
                self.regressor = nn.Sequential(
                    nn.Linear(num_features, self.c.num_pretrained_fc_units),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.c.num_pretrained_fc_units, self.c.num_pretrained_fc_units),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.c.num_pretrained_fc_units, 1),
                )

        else:
            raise ValueError('Experiment type not found.')

    def forward(self, x, batch_attributes=None):
        if self.c.experiment_type == "pretrain_only":
            f = self.pretrained_model(x)
            f = f.view(f.size(0), -1)
        elif self.c.experiment_type == "encoder_only":
            if self.c.conditional_model:
                # real_z = encoder(batch_x_var, self.phase, self.alpha, labels=batch_y_var)
                if self.c.std_vae_loss:
                    real_h, label_hat, real_mu, real_logvar = self.pretrained_model(x,
                                                                             self.phase,
                                                                             self.alpha,
                                                                             attributes=batch_attributes,
                                                                             return_attributes_hat=True)
                else:
                    # real_h, label_hat = encoder(batch_x_var,
                    #                             self.phase,
                    #                             self.alpha,
                    #                             attributes=batch_attributes,
                    #                             return_attributes_hat=True)
                    if self.c.cond_h_on_y:
                        _, _, f = self.pretrained_model(x,
                                                                            self.phase,
                                                                            self.alpha,
                                                                            attributes=batch_attributes,
                                                                            return_attributes_hat=True,
                                                                            return_concat_x_a=True)

                    else:
                        f, _ = self.pretrained_model(x,
                                                                self.phase,
                                                                self.alpha,
                                                                attributes=batch_attributes,
                                                                return_attributes_hat=True)
        if self.c.pretrained_model_use_attributes:
            f = torch.cat((f, batch_attributes), dim=1)
        # y = self.regressor(f) + 3.0
        y = self.regressor(f)
        return y