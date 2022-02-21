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
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
from torch.nn import functional as F
from utils import training_utils
from models.layers import AttributeLogitsGumbelSample, ConvBlock

class Generator(nn.Module):
    def __init__(self,
                 num_latents,
                 n_label=None,
                 c=None):
        super().__init__()

        num_latents = num_latents + n_label
        # TODO - change this to just num_latents from Contstructor call
        
        assert c is not None
        assert n_label is not None
        self.c = c

        self.progression = nn.ModuleList([ConvBlock(num_latents, num_latents, 4, 3, 3, 1,
                                                    use_residual=False), #4

                                          ConvBlock(num_latents, num_latents, 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                                    squeeze_and_excite=self.c.gen_use_squeeze_and_excite), # 8

                                          ConvBlock(num_latents, num_latents, 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                                    squeeze_and_excite=self.c.gen_use_squeeze_and_excite), #16

                                          ConvBlock(num_latents, num_latents, 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                             squeeze_and_excite=self.c.gen_use_squeeze_and_excite,
                                                    self_attention=self.c.use_self_attention), # 32
                                          # 64
                                          ConvBlock(num_latents, int(num_latents / 2), 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                                    squeeze_and_excite=self.c.gen_use_squeeze_and_excite),
                                          # 128
                                          ConvBlock(int(num_latents / 2), int(num_latents / 4), 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                                    squeeze_and_excite=self.c.gen_use_squeeze_and_excite),
                                          # 256
                                          ConvBlock(int(num_latents / 4), int(num_latents / 8), 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                                    squeeze_and_excite=self.c.gen_use_squeeze_and_excite),

                                          ConvBlock(int(num_latents / 8), int(num_latents / 16), 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                                    squeeze_and_excite=self.c.gen_use_squeeze_and_excite),

                                          ConvBlock(int(num_latents / 16), int(num_latents / 32), 3, 1,
                                                    use_residual=self.c.gen_use_residual,
                                                    squeeze_and_excite=self.c.gen_use_squeeze_and_excite)])

        # num_channels = 4 if self.c.use_masks else 3
        num_channels = self.c.nc + 1 if self.c.use_masks else self.c.nc

        self.to_rgb = nn.ModuleList(
            [nn.Conv2d(num_latents, num_channels, 1),  # Each has 3 out channels and kernel size 1x1 for rgb pixels!
             nn.Conv2d(num_latents, num_channels, 1),
             nn.Conv2d(num_latents, num_channels, 1),
             nn.Conv2d(num_latents, num_channels, 1),
             nn.Conv2d(int(num_latents / 2), num_channels, 1),
             nn.Conv2d(int(num_latents / 4), num_channels, 1),
             nn.Conv2d(int(num_latents / 8), num_channels, 1),
             nn.Conv2d(int(num_latents / 16), num_channels, 1),
             nn.Conv2d(int(num_latents / 32), num_channels, 1)])

        if self.c.dataset == "vehicles":
            self.label_embedding = nn.Sequential(nn.Linear(self.c.n_label, self.c.n_label),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(self.c.n_label, self.c.n_label)
                                             )
        elif self.c.dataset == "chairs":
            self.label_embedding = nn.Sequential(nn.Linear(self.c.n_label, self.c.n_label, bias=True),
                                                 nn.LeakyReLU(0.2),
                                                 )
        else:
            raise ValueError

        self.sigmoid = nn.Sigmoid()

        if self.c.use_generator_output_tanh:
            self.out_act = nn.Tanh()
        else:
            self.out_act = lambda x: x

    def forward(self,
                inputs,
                phase,
                alpha=-1,
                attributes=None,
                transform_onehot_attributes=True,
                return_mask=False,
                return_transformed_attributes=False):

        # TODO: make it so the model works without label_embedding and just z
        if transform_onehot_attributes:
            attributes = self.label_embedding(attributes)

        out = torch.cat([inputs, attributes], 1)
        if self.c.noise == 'sphere':
            out = training_utils.normalize(out, eps=self.c.epsilon)

        out = out.unsqueeze(2).unsqueeze(3)

        # Count forward when iterating layers
        for cur_phase, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if cur_phase > 0 and phase > 0:
                # This is what upscales/doubles resolution
                upsample = F.upsample(out, scale_factor=2)
                out = conv(upsample)
                # if self.c.use_self_attention and phase == self.c.self_attention_layer:

            else:
                out = conv(out)

            if cur_phase == phase:  # The final layer is ALWAYS either to_rgb layer, or a mixture of 2 to-rgb_layers!
                out = self.out_act(to_rgb(out))

                if cur_phase > 0 and 0 <= alpha < 1:
                    skip_rgb = self.out_act(self.to_rgb[cur_phase - 1](upsample))
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        if (out.size(1) == 4) or out.size(1) == 2:
            out, mask = training_utils.split_mask_out_of_generated(out)
            # mask = self.sigmoid(mask)
            mask = mask.add(1.0).div(2.0)
            has_mask = True
        else:
            has_mask = False

        if return_mask and return_transformed_attributes:
            assert has_mask
            return out, mask, attributes
        elif return_mask and not return_transformed_attributes:
            assert has_mask
            return out, mask
        elif not return_mask and return_transformed_attributes:
            return out, attributes
        elif not return_mask and not return_transformed_attributes:
            return out
        else:
            raise ValueError

class PredictiveModelCombinedModel(nn.Module):
    def __init__(self, c):
        super().__init__()

        assert c is not None
        self.c = c

        self.base_rating = self.c.base_rating

        if self.c.predictive_model_nonlinear:
            self.mu = nn.Sequential(nn.BatchNorm1d(int(self.c.n_latents)),
                                    nn.Linear(self.c.n_latents, self.c.num_combined_model_fc_units),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.c.num_combined_model_fc_units, 1))
            self.logvar = nn.Sequential(nn.BatchNorm1d(int(self.c.n_latents)),
                                        nn.Linear(self.c.n_latents, self.c.num_combined_model_fc_units),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.c.num_combined_model_fc_units, 1))
        else:
            self.mu = nn.Sequential(nn.BatchNorm1d(int(self.c.n_latents)),
                                    nn.Linear(int(self.c.n_latents), 1),
                                    )
            self.logvar = nn.Sequential(nn.BatchNorm1d(int(self.c.n_latents)),
                                        nn.Linear(int(self.c.n_latents), 1))

    def reparam(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, h):
        y_mu = self.mu(h)
        y_logvar = F.softplus(self.logvar(h))
        y_hat = self.reparam(y_mu, y_logvar) + self.base_rating # Note this is offset to help learning
        return y_hat, y_mu, y_logvar


class Encoder(nn.Module):
    def __init__(self,
                 num_latents,
                 n_label=111, # this is residual from car dataset, but overwrite for your own data
                 c=None):
        super().__init__()

        assert c is not None
        self.c = c
        
        # This may help you if modifying for your own code, here is an example mapping.
        # ConvBlock:
        # 1st layers - in/out
        #     Conv: 256x256, [-1, 32, 256, 256] ->  [-1, 64, 256, 256]
        #     SpectralNorm: [-1, 64, 256, 256] ->  [-1, 64, 256, 256]
        #     LReLu: [-1, 64, 256, 256] ->  [-1, 64, 256, 256]
        # 2nd layers - out/out
        #     Conv: 256x256, [-1, 64, 256, 256] ->  [-1, 64, 256, 256]
        #     SpectralNorm: [-1, 64, 256, 256] ->  [-1, 64, 256, 256]
        #     LReLu: [-1, 64, 256, 256] ->  [-1, 64, 256, 256]

        num_latents = num_latents + n_label
        # TODO - change this to just num_latents from Contstructor call

        self.progression = nn.ModuleList([ConvBlock(int(num_latents / 32), int(num_latents / 16), 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite),
                                          # 512
                                          ConvBlock(int(num_latents / 16), int(num_latents / 8), 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite),
                                          # 256
                                          ConvBlock(int(num_latents / 8), int(num_latents / 4), 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite),
                                          # 128
                                          ConvBlock(int(num_latents / 4), int(num_latents / 2), 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite),
                                          # 64
                                          ConvBlock(int(num_latents / 2), num_latents, 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite),
                                          # 32
                                          ConvBlock(num_latents, num_latents, 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite,
                                                    self_attention=self.c.use_self_attention),
                                          # 16
                                          ConvBlock(num_latents, num_latents, 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite),
                                          # 8
                                          ConvBlock(num_latents, num_latents, 3, 1,
                                                    use_residual=self.c.enc_use_residual,
                                                    squeeze_and_excite=self.c.enc_use_squeeze_and_excite),
                                          # 4
                                          ConvBlock(num_latents, num_latents, 3, 1, 4, 0,
                                                    use_residual=False)])

        # num_channels = self.c.nc + 1 if self.c.use_masks else self.c.nc
        num_channels = self.c.nc + 1 if self.c.inject_masks_into_encoder else self.c.nc

        self.from_rgb = nn.ModuleList([nn.Conv2d(num_channels, int(num_latents / 32), 1),
                                       nn.Conv2d(num_channels, int(num_latents / 16), 1),  # 512x512
                                       nn.Conv2d(num_channels, int(num_latents / 8), 1),  # 256x256 -> [-1, 32, 256, 256]
                                       nn.Conv2d(num_channels, int(num_latents / 4), 1),  # 128x128
                                       nn.Conv2d(num_channels, int(num_latents / 2), 1),
                                       nn.Conv2d(num_channels, num_latents, 1),
                                       nn.Conv2d(num_channels, num_latents, 1),
                                       nn.Conv2d(num_channels, num_latents, 1),
                                       nn.Conv2d(num_channels, num_latents, 1)])

        # Note: This is not strictly Gaussian Mixture Model since nonlinear transform
        if self.c.dataset == "vehicles":
            self.post_label_embedding = nn.Sequential(nn.Linear(self.c.n_latents, self.c.n_latents),
                                                      nn.LeakyReLU(0.2),
                                                      nn.Linear(self.c.n_latents, self.c.n_latents)
                                                      )

            self.post_label_embedding_2 = nn.Sequential(nn.Linear(self.c.n_latents, self.c.n_latents),
                                                        nn.LeakyReLU(0.2),
                                                        nn.Linear(self.c.n_latents, self.c.n_latents)
                                                        )
        elif self.c.dataset == "chairs":
            self.post_label_embedding = nn.Sequential(nn.Linear(self.c.n_latents, self.c.n_latents, bias=True),
                                                      nn.LeakyReLU(0.2),
                                                      )
            self.post_label_embedding_2 = nn.Sequential(nn.Linear(self.c.n_latents, self.c.n_latents, bias=True),
                                                      nn.LeakyReLU(0.2),
                                                      )
        self.attribute_logits = AttributeLogitsGumbelSample(self.c.gumbel_softmax_temperature)

        self.n_layer = len(self.progression)

    def reparam(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, inputs, phase, alpha=-1, attributes=None, rating=None, return_attributes_hat=False, return_concat_x_a=False):  # default was step=0, alpha=-1
        # Count backwards when creating layers
        # e.g., phase = 3 is 32x32 so this counts 3,2,1,0
        for cur_phase in range(phase, -1, -1):
            prog_layer_ind = self.n_layer - cur_phase - 1  # e.g., for 256x256, this = 2

            if cur_phase == phase:
                out = self.from_rgb[prog_layer_ind](inputs)

            out = self.progression[prog_layer_ind](out)

            if cur_phase > 0:
                # This is what halves the size
                out = F.avg_pool2d(out, 2)

                if cur_phase == phase and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(inputs, 2)
                    skip_rgb = self.from_rgb[prog_layer_ind + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out_latent, out_label_logits = training_utils.split_attributes_out(out)

        if self.c.use_gumbel_softmax:
            out_label_hat = self.attribute_logits(out_label_logits)
        else:
            out_label_hat = out_label_logits

        if attributes is not None:
            concat_x_a = torch.cat((out_latent, attributes), 1)  # true labels for conditional operation, out_label_hat is for cross entropy from true labels

            if self.c.vae_loss == 'stdvae' or self.c.vae_loss == 'both':
                out_mu = self.post_label_embedding(concat_x_a)
                out_logvar = self.post_label_embedding_2(concat_x_a)

                out = self.reparam(out_mu, out_logvar)

                if return_attributes_hat and return_concat_x_a:
                    return out, out_label_hat, concat_x_a, out_mu, out_logvar
                elif return_attributes_hat and not return_concat_x_a:
                    return out, out_label_hat, out_mu, out_logvar
                else:
                    return out, out_mu, out_logvar

            elif self.c.vae_loss == 'densityratio':
                out = self.post_label_embedding(concat_x_a)

                if return_attributes_hat and return_concat_x_a:
                    return out, out_label_hat, concat_x_a
                elif return_attributes_hat and not return_concat_x_a:
                    return out, out_label_hat
                else:
                    return out

            else:
                raise ValueError

        else:
            # TODO: Dec 6, 2021 - this all needs updating since model has always been conditional.

            # out_latent, out_label_hat = base_model.split_labels_out_of_latent(out)
            # out_label_hat = self.attribute_logits(out_label_hat)
            # Need to make this Gumbel entropy for unsupervised learning term
            if self.c.vae_params_cond_a:
                # out = torch.cat((out_latent, out_label_hat), 1)
                if self.c.std_vae_loss:
                    out_mu = self.post_label_embedding(out)
                    out_logvar = self.post_label_embedding_2(out)
                    out = self.reparam(out_mu, out_logvar)
                else:
                    out = self.post_label_embedding(out)

            else:
                if self.c.std_vae_loss:
                    out_mu = self.post_label_embedding(out_latent)
                    out_logvar = self.post_label_embedding_2(out_latent)
                    out = self.reparam(out_mu, out_logvar)
                else:
                    out = self.post_label_embedding(out)

            if self.c.noise == 'sphere':
                out = training_utils.normalize(out, eps=self.c.epsilon)

            if not self.c.vae_params_cond_a:
                out = torch.cat((out, out_label_hat), 1)

            # For putting into generator
            if not self.c.std_vae_loss:
                if return_attributes_hat:
                    return out, out_label_hat
                else:
                    return out
            else:
                if return_attributes_hat:
                    return out, out_label_hat, out_mu, out_logvar
                else:
                    return out, out_mu, out_logvar
