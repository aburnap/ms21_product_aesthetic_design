# -*- coding: utf-8 -*-
'''
        Project: Design and Evaluation of Product Aesthetics: A Human‐Machine Hybrid Approach
        Date: October 27, 2020
        Authors: Alex Burnap, Yale University
        Email: alex.burnap@yale.edu

        Description:
        Various utilities for creating videos of generated and interpolated aesthetic product designs.

        License:
        All Rights Reserved
        Unauthorized copying of this file, via any medium is strictly prohibited
        Proprietary and confidential

        OSS Code Attribution: Portions of Code From Open Source Projects:
        https://github.com/tkarras/progressive_growing_of_gans
        https://github.com/AaltoVision/pioneer
        https://github.com/DmitryUlyanov/AGE
        https://github.com/akanimax/attn_gan_pytorch/
'''

import os
import re

import numpy as np
import comet_ml
import torch
import experiment
from config import c
from torch.autograd import Variable
import moviepy.editor
import numpy as np
import pandas as pd
from collections import OrderedDict

from utils.training_utils import untransform, renormalize, renormalize_single, freeze_model, slerp, split_attributes_out
from training.evaluate import SampleAndReconstruct

import moviepy.editor
import numpy as np
import pandas as pd
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont

from config import c


def search_vehicle(year, make, model, vehicle_names_df):
    try:
        idx = vehicle_names_df[(vehicle_names_df['year'] == year) & \
                               (vehicle_names_df['make'] == make) & \
                               (vehicle_names_df['model'] == model)].iloc[0].name
    except IndexError:
        return None
    return idx



def create_video(idx_dict,
                 generator,
                 encoder,
                 session,
                 dataset,
                 force_color=True,
                 color = [0.75, 0.75, 0.75],
                 duration_per_interp_sec=10,
                 duration_per_interp_sec_fast=4,
                 mp4=None,
                 mp4_fps=10,
                 mp4_codec='libx265',
                 mp4_bitrate='16M'):
    '''
    Generate MP4 video of random interpolations using a previously trained network.
    To run, uncomment the appropriate line in config.py and launch train.py.
    :param idx_dict: Dict of product design models to interpolate between
    :param generator:
    :param encoder:
    :param session:
    :param dataset:
    :param force_color:
    :param color:
    :param duration_per_interp_sec:
    :param duration_per_interp_sec_fast:
    :param mp4:
    :param mp4_fps:
    :param mp4_codec:
    :param mp4_bitrate:
    :return:
    '''

    num_vehicles = len(idx_dict.items())
    num_fast = np.nonzero(list(idx_dict.values()))[0].shape[0]
    num_slow = len(list(idx_dict.values())) - num_fast
    num_frames_slow = int(np.rint(duration_per_interp_sec * mp4_fps * (num_slow)))
    num_frames_fast = int(np.rint(duration_per_interp_sec_fast * mp4_fps * (num_fast)))
    num_frames = num_frames_slow + num_frames_fast

    num_latent_steps_horizontal = int(duration_per_interp_sec * mp4_fps ) # 5 seconds per transition @ 30fps
    num_latent_steps_horizontal_fast = int(duration_per_interp_sec_fast * mp4_fps ) # 5 seconds per transition @ 30fps

    generator.eval()
    encoder.eval()
    freeze_model(generator, True)
    freeze_model(encoder, True)

    reso = 4 * 2 ** session.phase
    t = np.zeros([num_frames, reso, reso, 3], dtype=np.uint8)
    cur_step = 0

    for vehicle_ind, (idx, idx_speed) in enumerate(idx_dict.items()):
        print(idx)
        if vehicle_ind == 0:
            prev_image, prev_labels = dataset.__getitem__(idx, reso, with_attributes=True, astorch=True)
            prev_image, prev_labels = prev_image.unsqueeze(0), prev_labels.unsqueeze(0)
            if force_color:
                prev_labels[0, -3:] = torch.Tensor(color)

        try:
            real_image, batch_labels = dataset.__getitem__(idx, reso, with_attributes=True, astorch=True)
            real_image, batch_labels = real_image.unsqueeze(0), batch_labels.unsqueeze(0)
            if force_color:
                batch_labels[0, -3:] = torch.Tensor([0.75, 0.75, 0.75])

        except IndexError:
            break

        interpolation_x = Variable(real_image, volatile=True).cuda()
        prev_interpolated_labels = Variable(prev_labels, volatile=True).cuda()
        interpolated_labels = Variable(batch_labels, volatile=True).cuda()

        if session.c.conditional_model:
            prev_z0 = encoder(Variable(prev_image),
                              session.phase,
                              session.alpha,
                              attributes = prev_labels).detach()

            z0 = encoder(Variable(interpolation_x),
                         session.phase,
                         session.alpha,
                         attributes = interpolated_labels).detach()
        else:
            raise ValueError
            # z0 = encoder(Variable(x),
            #              session.phase,
            #              session.alpha,
            #              attributes=None, return_label_hat=False).detach()

        prev_z0 = prev_z0.cpu()
        z0 = z0.cpu()

        if idx_speed == 0:
            latent_steps_horizontal = num_latent_steps_horizontal

        else:
            latent_steps_horizontal = num_latent_steps_horizontal_fast

        print(cur_step)
        for x_i in range(latent_steps_horizontal):
            t_x = float(x_i) / (latent_steps_horizontal - 1)
            z0_x = slerp(prev_z0.squeeze().data, z0.squeeze().data, t_x)
            z0_x = z0_x.unsqueeze(0)

            if session.c.conditional_model:
                delta_z = ((interpolated_labels - prev_interpolated_labels) / (latent_steps_horizontal - 1))
                interpolated_labels_x = prev_interpolated_labels + float(x_i) * delta_z

            if session.c.conditional_model:
                z0_x, _ = split_attributes_out(z0_x)
                z0_x = z0_x.cuda()
                label = interpolated_labels_x
                label = label.cuda()
            else:
                z0_x, label = split_attributes_out(z0_x)
                z0_x = z0_x.cuda()
                label = label.cuda

            gex = generator(z0_x,
                            session.phase,
                            session.alpha,
                            attributes=label,
                            transform_onehot_attributes=True,
                            return_mask=False,
                            return_transformed_attributes=False).detach()

            t_step = renormalize(gex.data[:].cpu().numpy())
            t[cur_step+x_i] = t_step

        cur_step += latent_steps_horizontal
        prev_image, prev_labels = real_image, batch_labels

    return t

def create_video_with_ratings(idx_dict,
                              generator,
                              encoder,
                              predictor,
                              session,
                              dataset,
                              force_color=True,
                              color = [0.75, 0.75, 0.75],
                              duration_per_interp_sec=10,
                              duration_per_interp_sec_fast=4,
                              mp4=None,
                              mp4_fps=10,
                              mp4_codec='libx265',
                              mp4_bitrate='16M'):
    '''
    Generate MP4 video of random interpolations using a previously trained network.
    To run, uncomment the appropriate line in config.py and launch train.py.
    :param idx_dict: Dict of product design models to interpolate between
    :param generator:
    :param encoder:
    :param session:
    :param dataset:
    :param force_color:
    :param color:
    :param duration_per_interp_sec:
    :param duration_per_interp_sec_fast:
    :param mp4:
    :param mp4_fps:
    :param mp4_codec:
    :param mp4_bitrate:
    :return:
    '''

    num_vehicles = len(idx_dict.items())
    num_fast = np.nonzero(list(idx_dict.values()))[0].shape[0]
    num_slow = len(list(idx_dict.values())) - num_fast
    num_frames_slow = int(np.rint(duration_per_interp_sec * mp4_fps * (num_slow)))
    num_frames_fast = int(np.rint(duration_per_interp_sec_fast * mp4_fps * (num_fast)))
    num_frames = num_frames_slow + num_frames_fast

    num_latent_steps_horizontal = int(duration_per_interp_sec * mp4_fps ) # 5 seconds per transition @ 30fps
    num_latent_steps_horizontal_fast = int(duration_per_interp_sec_fast * mp4_fps ) # 5 seconds per transition @ 30fps

    generator.eval()
    encoder.eval()
    predictor.eval()
    freeze_model(generator, True)
    freeze_model(encoder, True)
    freeze_model(predictor, True)

    reso = 4 * 2 ** session.phase
    t = np.zeros([num_frames, reso, reso, 3], dtype=np.uint8)
    cur_step = 0

    for vehicle_ind, (idx, idx_speed) in enumerate(idx_dict.items()):
        print(idx)
        if vehicle_ind == 0:
            prev_image, prev_labels = dataset.__getitem__(idx, reso, with_attributes=True, astorch=True)
            prev_image, prev_labels = prev_image.unsqueeze(0), prev_labels.unsqueeze(0)
            if force_color:
                prev_labels[0, -3:] = torch.Tensor(color)

        try:
            real_image, batch_labels = dataset.__getitem__(idx, reso, with_attributes=True, astorch=True)
            real_image, batch_labels = real_image.unsqueeze(0), batch_labels.unsqueeze(0)
            if force_color:
                batch_labels[0, -3:] = torch.Tensor([0.75, 0.75, 0.75])

        except IndexError:
            break

        interpolation_x = Variable(real_image, volatile=True).cuda()
        prev_interpolated_labels = Variable(prev_labels, volatile=True).cuda()
        interpolated_labels = Variable(batch_labels, volatile=True).cuda()

        if session.c.conditional_model:
            prev_z0 = encoder(Variable(prev_image),
                              session.phase,
                              session.alpha,
                              attributes = prev_labels).detach()

            z0 = encoder(Variable(interpolation_x),
                         session.phase,
                         session.alpha,
                         attributes = interpolated_labels).detach()
        else:
            raise ValueError
            # z0 = encoder(Variable(x),
            #              session.phase,
            #              session.alpha,
            #              attributes=None, return_label_hat=False).detach()

        prev_z0 = prev_z0.cpu()
        z0 = z0.cpu()

        if idx_speed == 0:
            latent_steps_horizontal = num_latent_steps_horizontal

        else:
            latent_steps_horizontal = num_latent_steps_horizontal_fast

        print(cur_step)
        for x_i in range(latent_steps_horizontal):
            t_x = float(x_i) / (latent_steps_horizontal - 1)
            z0_x = slerp(prev_z0.squeeze().data, z0.squeeze().data, t_x)
            z0_x = z0_x.unsqueeze(0)

            if session.c.conditional_model:
                delta_z = ((interpolated_labels - prev_interpolated_labels) / (latent_steps_horizontal - 1))
                interpolated_labels_x = prev_interpolated_labels + float(x_i) * delta_z

            if session.c.conditional_model:
                z0_x, _ = split_attributes_out(z0_x)
                z0_x = z0_x.cuda()
                label = interpolated_labels_x
                label = label.cuda()
            else:
                z0_x, label = split_attributes_out(z0_x)
                z0_x = z0_x.cuda()
                label = label.cuda

            gex = generator(z0_x,
                            session.phase,
                            session.alpha,
                            attributes=label,
                            transform_onehot_attributes=True,
                            return_mask=False,
                            return_transformed_attributes=False).detach()

            gex_h, gex_attributes_hat, gex_concat_x_a = encoder(gex,
                                                                session.phase,
                                                                session.alpha,
                                                                attributes=label,
                                                                return_attributes_hat=True,
                                                                return_concat_x_a=True)

            y_hat, y_mu, y_logvar = predictor(gex_concat_x_a)
            pred_rating = y_mu + 3.0
            pred_rating = pred_rating[0][0].detach().cpu().numpy()
            pred_rating_text = np.round(pred_rating, 2)
            pred_rating_text = str(pred_rating_text)

            t_step = renormalize(gex.data[:].cpu().numpy())

            image_pil = Image.fromarray(t_step[0])
            image_pil_draw = ImageDraw.Draw(image_pil)
            font = ImageFont.truetype("./utils/fonts/OpenSans-Regular.ttf", 30)
            image_pil_draw.rectangle((150, 20, 360, 115), outline='red', fill='white')
            image_pil_draw.text((256-64, 24), "Prediction",  font=font, fill='rgb(0, 0, 0)')
            image_pil_draw.text((256-30, 64), pred_rating_text,  font=font, fill='rgb(0, 0, 0)')
            image = image_pil.convert("RGB")
            #             image = np.asarray(image, dtype=np.float32) / 255
            image = np.expand_dims(image, 0)
            t[cur_step+x_i] = image

        cur_step += latent_steps_horizontal
        prev_image, prev_labels = real_image, batch_labels


    return t