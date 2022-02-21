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
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from torch.autograd import Variable

import utils.training_utils
from losses import losses
from losses.losses import RatingLoss


# from data.vehicles import VehiclesforPredictiveModelOnly as Vehicles, data
from data.vehicles import DataGenerator
from utils import logging_utils, training_utils
from utils.training_utils import AverageMeter
from config import c

if c.env.computer == 'ab3349':
    # For headless servers
    mpl.use('Agg')
else:
    mpl.use('TkAgg')

class SampleAndReconstruct(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_intermediate_samples(generator, global_i, session, writer=None, collateImages=True, loader=None):
        generator.eval()
        # encoder.train()
        # generator.train()

        training_utils.freeze_model(generator, False)

        reso = 4 * 2 ** session.phase

        # x = Variable(real_image).cuda()
        if session.c.conditional_model:
            if session.phase < 1:
                dataset = DataGenerator.data_generator_phase0(loader, session.c.sample_N, reso)
            else:
                dataset = DataGenerator.data_generator_session(loader, session.c.sample_N, reso, session)
            # real_image, real_labels = next(dataset)
            if session.c.use_masks:
                _, batch_attributes, batch_masks = next(dataset)
            else:
                _, batch_attributes = next(dataset)

            # _, batch_attributes = next(dataset)
            batch_attributes = Variable(batch_attributes).cuda(non_blocking=(session.c.gpu_count > 1))

        # Total number is samplesRepeatN * colN * rowN
        # e.g. for 51200 samples, outcome is 5*80*128. Please only give multiples of 128 here.
        samplesRepeatN = int(session.c.sample_N / 128) if not collateImages else 1
        reso = 4 * 2 ** session.phase

        if not collateImages:
            special_dir = '../metrics/{}/{}/{}'.format(session.c.data, reso, str(global_i).zfill(6))
            while os.path.exists(special_dir):
                special_dir += '_'

            os.makedirs(special_dir)

        for outer_count in range(samplesRepeatN):

            colN = 1 if not collateImages else min(10, int(np.ceil(session.c.sample_N / 4.0)))
            rowN = 128 if not collateImages else min(5, int(np.ceil(session.c.sample_N / 4.0)))
            images = []
            for row_ind in range(rowN):
                # myz = Variable(torch.randn(args.n_label * colN, args.nz)).cuda()
                myz = Variable(torch.randn(colN, session.c.nz)).cuda()
                myz = training_utils.normalize(myz)
                if session.c.conditional_model:
                    myz, _ = training_utils.split_attributes_out(myz)
                    input_class = batch_attributes[(row_ind * colN):((row_ind + 1) * colN)]
                    new_imgs = generator(
                                        myz,
                                        session.phase,
                                        session.alpha,
                                        attributes=input_class,
                                        transform_onehot_attributes=True, return_mask=False).detach().data.cpu()
                else:
                    myz, input_class = training_utils.split_attributes_out(myz)
                    new_imgs = generator(
                    myz,
                    session.phase,
                    session.alpha,
                    attributes=input_class,
                    transform_onehot_attributes=False, return_mask=False).detach().data.cpu()

                

                images.append(new_imgs)

            if collateImages:
                sample_dir = '{}/sample'.format(session.c.save_dir)
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                save_name = 'random_samples_{}'.format(str(global_i + 1).zfill(6))
                save_path = '{}/{}.png'.format(sample_dir, save_name)
                torchvision.utils.save_image(
                    torch.cat(images, 0),
                    save_path,
                    # nrow=args.n_label * colN,
                    nrow=colN,
                    normalize=True,
                    range=(-1, 1),
                    padding=0)
                # Hacky but this is an easy way to rescale the images to nice big lego format:
                im = Image.open(save_path)
                im2 = im.resize((1024, 1024 if reso < 256 else 1024))
                im2.save(save_path)

                if session.c.use_cometML:
                    session.comet_experiment.log_image(image_data=save_path, name=save_name)

                if writer:
                    writer.add_image('samples_latest_{}'.format(session.phase), torch.cat(images, 0), session.phase)
            else:
                for ii, img in enumerate(images):
                    torchvision.utils.save_image(
                        img,
                        '{}/{}_{}.png'.format(special_dir, str(global_i + 1).zfill(6), ii + outer_count * 128),
                        nrow=session.c.n_label * colN,
                        normalize=True,
                        range=(-1, 1),
                        padding=0)

        generator.train()

    reconstruction_set_x = None

    @staticmethod
    def reconstruct(input_image, encoder, generator, session, cond_labels=None):
        encoder.train()
        generator.train()
        with torch.no_grad():
            if session.c.conditional_model:

                if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                    h_recon, _, _  = encoder(Variable(input_image), session.phase, session.alpha,
                                 attributes=cond_labels, return_attributes_hat=False)
                    h_recon = h_recon.detach()
                    h_recon, _ = training_utils.split_attributes_out(h_recon)
                    attributes = cond_labels
                    x_recon = generator(h_recon, session.phase, session.alpha, attributes=attributes, transform_onehot_attributes=True, return_mask=False).detach()
                else:
                    h_recon = encoder(Variable(input_image), session.phase, session.alpha,
                                 attributes=cond_labels, return_attributes_hat=False).detach()
                    h_recon, _ = training_utils.split_attributes_out(h_recon)
                    attributes = cond_labels
                    x_recon = generator(h_recon, session.phase, session.alpha, attributes=attributes, transform_onehot_attributes=True, return_mask=False).detach()
            else:
                h_recon = encoder(Variable(input_image), session.phase, session.alpha, return_attributes_hat=False).detach()
                h_recon, attributes = training_utils.split_attributes_out(h_recon)
                x_recon = generator(h_recon, session.phase, session.alpha, attributes=attributes, transform_onehot_attributes=False, return_mask=False).detach()
            return x_recon.data[:]

    @staticmethod
    def reconstruct_images(generator, encoder, loader, global_i, nr_of_imgs, prefix, reals, reconstructions, session,
                           writer=None):  # of the form"/[dir]"
        generator.eval()
        encoder.eval()
        # generator.train()
        # encoder.train()

        training_utils.freeze_model(generator, False)
        training_utils.freeze_model(encoder, False)

        if reconstructions and nr_of_imgs > 0:
            reso = 4 * 2 ** session.phase

            # First, create the single grid

            if SampleAndReconstruct.reconstruction_set_x is None or SampleAndReconstruct.reconstruction_set_x.size(2) != reso or (
                    session.phase >= 1 and session.alpha < 1.0):
                if session.phase < 1:
                    dataset = DataGenerator.data_generator_phase0(loader, min(nr_of_imgs, 16), reso)
                else:
                    dataset = DataGenerator.data_generator_session(loader, min(nr_of_imgs, 16), reso, session)

                if session.c.use_masks:
                    # SampleAndReconstruct.reconstruction_set_x, SampleAndReconstruct.reconstruction_set_labels, _ = next(dataset)
                    SampleAndReconstruct.reconstruction_set_x, SampleAndReconstruct.reconstruction_set_labels, SampleAndReconstruct.reconstruction_set_masks = next(dataset)
                else:
                    SampleAndReconstruct.reconstruction_set_x, SampleAndReconstruct.reconstruction_set_labels = next(dataset)

            if session.c.conditional_model:
                if session.c.inject_masks_into_encoder:
                    SampleAndReconstruct.reconstruction_set_x_and_masks = torch.cat((SampleAndReconstruct.reconstruction_set_x,SampleAndReconstruct.reconstruction_set_masks), dim=1)
                    reco_image = SampleAndReconstruct.reconstruct(SampleAndReconstruct.reconstruction_set_x_and_masks, encoder, generator, session,
                                                              cond_labels=SampleAndReconstruct.reconstruction_set_labels)
                else:
                    reco_image = SampleAndReconstruct.reconstruct(SampleAndReconstruct.reconstruction_set_x, encoder, generator, session,
                                                                  cond_labels=SampleAndReconstruct.reconstruction_set_labels)
            else:
                reco_image = SampleAndReconstruct.reconstruct(SampleAndReconstruct.reconstruction_set_x, encoder, generator, session)

            t = torch.FloatTensor(SampleAndReconstruct.reconstruction_set_x.size(0) * 2, SampleAndReconstruct.reconstruction_set_x.size(1),
                                  SampleAndReconstruct.reconstruction_set_x.size(2), SampleAndReconstruct.reconstruction_set_x.size(3))

            t[0::2] = SampleAndReconstruct.reconstruction_set_x[:]
            t[1::2] = reco_image

            sample_dir = '{}/sample'.format(session.c.save_dir)
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)

            save_name = 'reconstructions_phase{}_iter{}_alpha{}'.format(session.phase, global_i, session.alpha)
            save_path = '{}{}/{}.png'.format(session.c.save_dir, prefix, save_name)
            torchvision.utils.save_image(t[:nr_of_imgs] / 2 + 0.5, save_path, padding=0)

            # Hacky but this is an easy way to rescale the images to nice big lego format:
            if session.phase < 4:
                h = np.ceil(nr_of_imgs / 8)
                h_scale = min(1.0, h / 8.0)
                im = Image.open(save_path)
                im2 = im.resize((1024, int(1024 * h_scale)))
                im2.save(save_path)

            if writer:
                writer.add_image('reconstruction_latest_{}'.format(session.phase), t[:nr_of_imgs] / 2 + 0.5,
                                 session.phase)

            if session.c.use_cometML:
                session.comet_experiment.log_image(image_data=save_path, name=save_name)

        encoder.train()
        generator.train()

    @staticmethod
    def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0, p1) / np.sqrt(np.dot(p0, p0)) / np.sqrt(np.dot(p1, p1)))
        k1 = np.sin((1 - t) * omega) / np.sin(omega)
        k2 = np.sin(t * omega) / np.sin(omega)
        return k1 * p0 + k2 * p1

    interpolation_set_x = None

    @staticmethod
    def interpolate_images(generator, encoder, loader, epoch, prefix, session, writer=None):
        generator.eval()
        encoder.eval()

        training_utils.freeze_model(generator, False)
        training_utils.freeze_model(encoder, False)

        nr_of_imgs = 4  # "Corners"
        reso = 4 * 2 ** session.phase
        if True:
            # if Utils.interpolation_set_x is None or Utils.interpolation_set_x.size(2) != reso or (phase >= 1 and alpha < 1.0):
            if session.phase < 1:
                dataset = DataGenerator.data_generator_phase0(loader, nr_of_imgs, reso)

            else:
                dataset = DataGenerator.data_generator_session(loader, nr_of_imgs, reso, session)
            if session.c.use_masks:
                real_image, batch_attributes, _ = next(dataset)
            else:
                real_image, batch_attributes = next(dataset)
            with torch.no_grad():
                SampleAndReconstruct.interpolation_set_x = Variable(real_image).cuda()
                SampleAndReconstruct.reconstruction_set_attributes = Variable(batch_attributes).cuda()

        latent_steps_horizontal = 8
        latent_steps_vertical = 8

        x = SampleAndReconstruct.interpolation_set_x
        interpolated_attributes = SampleAndReconstruct.reconstruction_set_attributes

        if session.c.conditional_model:
            z0, _, _ = encoder(Variable(x), session.phase, session.alpha, attributes=interpolated_attributes)
            z0 = z0.detach()
        else:
            z0 = encoder(Variable(x), session.phase, session.alpha).detach()
        z0 = z0.cpu()

        t = torch.FloatTensor(latent_steps_horizontal * (latent_steps_vertical + 1) + nr_of_imgs, x.size(1),
                              x.size(2), x.size(3))
        t[0:nr_of_imgs] = x.data[:]

        special_dir = session.c.save_dir #if not session.c.aux_outpath else session.c.aux_outpath

        if not os.path.exists(special_dir):
            os.makedirs(special_dir)

        # for o_i in range(nr_of_imgs):
        #     single_save_path = '{}{}/interpolations_{}_{}_{}_orig_{}.png'.format(special_dir, prefix, session.phase,
        #                                                                          epoch, session.alpha, o_i)
        #     grid = torchvision.utils.save_image(x.data[o_i] / 2 + 0.5, single_save_path, nrow=1,
        #                                         padding=0)  # , normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?

        # Origs on the first row here
        # Corners are: z0[0] ... z0[1]
        #                .
        #                .
        #              z0[2] ... z0[3]

        delta_z_ver0 = ((z0[2] - z0[0]) / (latent_steps_vertical - 1))
        delta_z_verN = ((z0[3] - z0[1]) / (latent_steps_vertical - 1))
        for y_i in range(latent_steps_vertical):
            if False:  # Linear interpolation
                z0_x0 = z0[0] + y_i * delta_z_ver0
                z0_xN = z0[1] + y_i * delta_z_verN
                delta_z_hor = (z0_xN - z0_x0) / (latent_steps_horizontal - 1)
                z0_x = Variable(torch.FloatTensor(latent_steps_horizontal, z0_x0.size(0)))

                for x_i in range(latent_steps_horizontal):
                    z0_x[x_i] = z0_x0 + x_i * delta_z_hor

            if True:  # Spherical
                t_y = float(y_i) / (latent_steps_vertical - 1)
                # z0_y = Variable(torch.FloatTensor(latent_reso_ver, z0.size(0)))
                z0_y1 = SampleAndReconstruct.slerp(z0[0].data, z0[2].data, t_y)
                z0_y2 = SampleAndReconstruct.slerp(z0[1].data, z0[3].data, t_y)
                z0_x = Variable(torch.FloatTensor(latent_steps_horizontal, z0[0].size(0)))

                for x_i in range(latent_steps_horizontal):
                    t_x = float(x_i) / (latent_steps_horizontal - 1)
                    z0_x[x_i] = SampleAndReconstruct.slerp(z0_y1, z0_y2, t_x)

            if session.c.conditional_model:
                delta_z_ver0 = ((interpolated_attributes[2] - interpolated_attributes[0]) / (latent_steps_vertical - 1))
                delta_z_verN = ((interpolated_attributes[3] - interpolated_attributes[1]) / (latent_steps_vertical - 1))

                interpolated_attributes_a0 = interpolated_attributes[0] + float(y_i) * delta_z_ver0
                interpolated_attributes_aN = interpolated_attributes[1] + float(y_i) * delta_z_verN
                delta_interpolated_attributes_hor = (interpolated_attributes_aN - interpolated_attributes_a0) / (
                            latent_steps_horizontal - 1)
                interpolated_attributes_a = Variable(
                    torch.FloatTensor(latent_steps_horizontal, interpolated_attributes_a0.size(0)))

                for x_i in range(latent_steps_horizontal):
                    interpolated_attributes_a[x_i] = interpolated_attributes_a0 + x_i * delta_interpolated_attributes_hor

            if session.c.conditional_model:
                # ex = encoder(Variable(input_image, volatile=True), session.phase, session.alpha,
                #              labels=cond_labels).detach()
                z0_x, _ = training_utils.split_attributes_out(z0_x)
                z0_x = z0_x.cuda()
                attributes = interpolated_attributes_a
                attributes = attributes.cuda()
            else:
                # ex = encoder(Variable(input_image, volatile=True), session.phase, session.alpha).detach()
                z0_x, attributes = training_utils.split_attributes_out(z0_x)
                z0_x = z0_x.cuda()
                attributes = attributes.cuda
            gex = generator(z0_x, attributes, session.phase, session.alpha).detach()

            # z0_x, label = base_model.split_labels_out_of_latent(z0_x)
            # gex = generator(z0_x, label, session.phase, session.alpha).detach()

            # Recall that yi=0 is the original's row:
            t[(y_i + 1) * latent_steps_vertical:(y_i + 2) * latent_steps_vertical] = gex.data[:]

            # for x_i in range(latent_steps_horizontal):
            #     single_save_path = '{}{}/interpolations_{}_{}_{}_{}x{}.png'.format(special_dir, prefix, session.phase,
            #                                                                        epoch, session.alpha, y_i, x_i)
            #     grid = torchvision.utils.save_image(gex.data[x_i] / 2 + 0.5, single_save_path, nrow=1,
            #                                         padding=0)  # , normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?

        save_path = '{}{}/interpolations_{}_{}_{}.png'.format(special_dir, prefix, session.phase, epoch, session.alpha)
        grid = torchvision.utils.save_image(t / 2 + 0.5, save_path, nrow=latent_steps_vertical,
                                            padding=0)  # , normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?
        # Hacky but this is an easy way to rescale the images to nice big lego format:
        if session.phase < 4:
            im = Image.open(save_path)
            im2 = im.resize((1024, 1024))
            im2.save(save_path)

        if writer:
            writer.add_image('interpolation_latest_{}'.format(session.phase), t / 2 + 0.5, session.phase)

        generator.train()
        encoder.train()


def tests_run(generator_for_testing, encoder, test_data_loader, session, writer, reconstruction=True,
              interpolation=True, collated_sampling=True, individual_sampling=True):
    if reconstruction:
        SampleAndReconstruct.reconstruct_images(generator_for_testing, encoder, test_data_loader, session.sample_i,
                                                nr_of_imgs=session.c.reconstructions_N, prefix='/sample', reals=False,
                                                reconstructions=True,
                                                session=session, writer=writer)
    if interpolation:
        for ii in range(session.c.interpolate_N):
            SampleAndReconstruct.interpolate_images(generator_for_testing, encoder, test_data_loader, session.sample_i + ii, prefix='',
                                                    session=session, writer=writer)
    if collated_sampling and session.c.sample_N > 0:
        SampleAndReconstruct.generate_intermediate_samples(
            generator_for_testing,
            session.sample_i, session=session, writer=writer, collateImages=True, loader=test_data_loader)  # True)
    if individual_sampling and session.c.sample_N > 0:  # Full sample set generation.
        print("Full Test samples - generating...")
        SampleAndReconstruct.generate_intermediate_samples(
            generator_for_testing,
            session.sample_i, session=session, collateImages=False, loader=test_data_loader)
        print("Full Test samples generated.")


def evaluate_combined_model(dataset, session, data_split='valid'):
    assert data_split in ['train', 'valid', 'test']
    evaluated_losses = AverageMeter()
    session.encoder.eval()
    session.predictor.eval()
    rating_loss = RatingLoss()

    if session.c.evaluate_full_validation_set and data_split=='valid':
        if session.c.dataset == "vehicles":
            num_batches = len(dataset.valid_y) // session.batch_size
        elif session.c.dataset == "chairs":
            num_batches = len(dataset.valid_y) // session.batch_size
            # num_batches = (len(dataset.valid_y) * session.c.number_viewpoints_per_product) // session.batch_size
    elif session.c.evaluate_full_validation_set and data_split=='test':
        if session.c.dataset == "vehicles":
            num_batches = len(dataset.test_y) // session.batch_size
        elif session.c.dataset == "chairs":
            num_batches = len(dataset.test_y) // session.batch_size
            # num_batches = (len(dataset.test_y) * session.c.number_viewpoints_per_product) // session.batch_size
    elif session.c.evaluate_full_validation_set and data_split=='train':
        if session.c.dataset == "vehicles":
            num_batches = len(dataset.train_y) // session.batch_size
        elif session.c.dataset == "chairs":
            num_batches = len(dataset.train_y) // session.batch_size
            # num_batches = (len(dataset.train_y) * session.c.number_viewpoints_per_product) // session.batch_size
    else:
        num_batches = 1

    for batch_ind in range(num_batches):
        i = batch_ind if session.c.evaluate_full_validation_set else None

        batch_images, batch_attributes, batch_ratings = dataset(batch_size=session.batch_size,
                                                                phase=session.phase,
                                                                alpha=session.alpha,
                                                                with_masks=False,
                                                                with_ratings=True,
                                                                data_split=data_split,
                                                                batch_ind=i,
                                                                side_view=session.c.evaluate_side_view_only)

        batch_ratings = Variable(batch_ratings).cuda(non_blocking=(session.c.gpu_count > 1))
        batch_x_var = Variable(batch_images).cuda(non_blocking=(session.c.gpu_count > 1))
        if session.c.conditional_model:
            batch_attributes = Variable(batch_attributes).cuda(non_blocking=(session.c.gpu_count > 1)).add(
                session.c.epsilon)

        with torch.no_grad():


            if session.c.conditional_model:

                if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                    if session.c.cond_h_on_y:
                        real_h, attributes_hat,  concat_x_a, real_mu, real_logvar = session.encoder(batch_x_var,
                                                                      session.phase,
                                                                      session.alpha,
                                                                      attributes=batch_attributes,
                                                                      return_attributes_hat=True,
                                                                      return_concat_x_a=True)
                    else:
                        real_h, label_hat, real_mu, real_logvar = session.encoder(batch_x_var,
                                                                                 session.phase,
                                                                                 session.alpha,
                                                                                 attributes=batch_attributes,
                                                                                 return_attributes_hat=True)
                else:
                    # real_h, label_hat = encoder(batch_x_var,
                    #                             session.phase,
                    #                             session.alpha,
                    #                             attributes=batch_attributes,
                    #                             return_attributes_hat=True)
                    if session.c.cond_h_on_y:
                        real_h, attributes_hat, concat_x_a = session.encoder(batch_x_var,
                                                                            session.phase,
                                                                            session.alpha,
                                                                            attributes=batch_attributes,
                                                                            return_attributes_hat=True,
                                                                            return_concat_x_a=True)

                    else:
                        real_h, attributes_hat = session.encoder(batch_x_var,
                                                                session.phase,
                                                                session.alpha,
                                                                attributes=batch_attributes,
                                                                return_attributes_hat=True)

            else:
                # if session.c.std_vae_loss:
                if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                    real_h, real_mu, real_logvar = session.encoder(batch_x_var,
                                                                  session.phase,
                                                                  session.alpha,
                                                                  attributes=None,
                                                                  return_attributes_hat=False)
                else:
                    real_h = session.encoder(batch_x_var,
                                            session.phase,
                                            session.alpha,
                                            return_attributes_hat=False)


            if session.c.cond_h_on_y:
                y_hat, y_mu, y_logvar = session.predictor(concat_x_a)
            else:
                y_hat, y_mu, y_logvar = session.predictor(real_h)
            # y_hat, y_mu, _ = predictor(real_h)

        if not session.c.evaluate_using_y_mu:
            predictive_loss = rating_loss(batch_ratings, y_hat)
        else:
            y_mu = y_mu + 3.0
            predictive_loss = rating_loss(batch_ratings, y_mu)

        # if session.c.evaluate_using_y_mu:
        #     predictive_loss = losses.calc_loss(batch_ratings, y_mu, session.c.match_y_metric)
        # else:
        #     predictive_loss = losses.calc_loss(batch_ratings, y_hat, session.c.match_y_metric)

        pred_loss_np = predictive_loss.item()
        evaluated_losses.update(pred_loss_np, batch_ratings.size(0))

    session.encoder.train()
    session.predictor.train()
    return evaluated_losses.avg


def output_test_image_predictions(exp, save_plot=True, bw_images=True):
    errors = []
    NUM_ROWS = np.ceil(exp.test_y_sideview.shape[0] / 4).astype(int)
    NUM_COLS = 4
    fig, axarr = plt.subplots(
        NUM_ROWS, NUM_COLS, figsize=(6 * NUM_COLS, 6 * NUM_ROWS), dpi=90)
    test_design_ids = exp.suvs_list[exp.valid_x_design_iloc][:, -1].astype(int)
    test_design_strings = [' '.join(elm) for elm in exp.suvs_list[exp.valid_x_design_iloc][:, :3]]
    exp.model.eval()

    if c.experiment_type == 'features_and_pretrain' or c.experiment_type == 'encoder_only':
        if c.encoder_train_mode:
            exp.model.encoder_model.train() # We freeze weights but this is due to problem with SpectralNorm and DataParallel
        else:
            exp.model.encoder_model.eval()
        if bw_images:
            exp.images_test = VehiclesforPretrainedModel(c.images_dir,
                                                         c.labels_dir,
                                                         use_RAM=False,
                                                         bw_images=True,
                                                         resize=False)

        else:
            exp.images_test = VehiclesforPretrainedModel(c.images_dir,
                                                         c.labels_dir,
                                                         use_RAM=False,
                                                         bw_images=True,
                                                         resize=False)

    elif c.experiment_type == 'pretrain_only':
        if bw_images:
            exp.images_test = VehiclesforPretrainedModel(c.images_dir,
                                                         c.labels_dir,
                                                         use_RAM=False,
                                                         bw_images=True,
                                                         resize=c.image_size)
        else:
            exp.images_test = VehiclesforPretrainedModel(c.images_dir,
                                                         c.labels_dir,
                                                         use_RAM=False,
                                                         bw_images=False,
                                                         resize=c.image_size)
    elif c.experiment_type == 'binary_and_real_pretrain':
        if not hasattr(exp, 'side_view_test_loader'):
            exp.images_test_pretrain = VehiclesforPretrainedModel(
                c.images_dir,
                c.labels_dir,
                False,
                bw_images=True,
                resize=c.image_size)

    for i, design_id in enumerate(test_design_ids):
        row, col = int(i / NUM_COLS), i % NUM_COLS
        ax = axarr[row, col]
        image_id = get_side_image_id_given_design_id(design_id)
        image = exp.images_test([image_id], size=c.image_size)[0]
        viewable_image = utils.training_utils.untransform(image[0])
        ax.imshow(viewable_image)

        cur_name = test_design_strings[i]
        ax.set_title(str(cur_name), fontsize=20)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        if c.experiment_type == 'encoder_only':
            cur_pred = exp.model(None, images_encoder=Variable(torch.from_numpy(image).cuda()))
        elif c.experiment_type == 'pretrain_only':
            cur_pred = exp.model(torch.autograd.Variable(torch.from_numpy(image).cuda()))
        elif c.experiment_type == 'binary_and_real_pretrain':
            cur_pred = exp.model(torch.autograd.Variable(torch.from_numpy(image).cuda()))
        else:
            raise ValueError('Experiment type not found.')
        cur_pred = cur_pred.detach().cpu().numpy()[0][0]
        cur_real = exp.valid_y[i]
        errors.append(np.abs(cur_real - cur_pred))

        cur_pred = round(cur_pred, 2)
        cur_pred = 'Pred: ' + str(cur_pred)
        textstr_pred = str(cur_pred)

        #         cur_real = df_homo_real_test.iloc[[i + STARTING_IDX]]['real_value']

        cur_real = round(cur_real, 2)
        cur_real = 'True: ' + str(cur_real)
        textstr_true = str(cur_real)

        props_true = dict(boxstyle='round', facecolor='green', alpha=0.2)
        props_pred = dict(boxstyle='round', facecolor='red', alpha=0.2)
        axarr[row, col].text(
            0.65,
            0.95,
            textstr_true,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='top',
            bbox=props_true)
        axarr[row, col].text(
            0.65,
            0.85,
            textstr_pred,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='top',
            bbox=props_pred)
    plt.tight_layout()
    mae = np.mean(np.array(errors).flatten())
    print('Grayscale Sideview Error: {}'.format(mae))
    if save_plot:
        plt.savefig(exp.result_subdir + '/' + c.experiment_description + "_test_set_designs.png")
    return mae


def get_random_image_id_given_design_id(design_id, batch_size=1):
    design_labels = np.load(c.labels_dir)  # Get Design ID to Image ID labels
    design_ids_of_images = design_labels['arr_5']  # 2 variables for viewpoint
    design_ids_of_images = design_ids_of_images.flatten()
    #     assert type(design_id) == int
    random_0_to_35 = np.random.randint(0, 36, size=batch_size)
    image_ids = np.where(design_ids_of_images == design_id)[0][random_0_to_35]
    return image_ids


def get_side_image_id_given_design_id(design_id):
    design_labels = np.load(c.labels_dir)  # Get Design ID to Image ID labels
    design_ids_of_images = design_labels['arr_5']  # 2 variables for viewpoint
    design_ids_of_images = design_ids_of_images.flatten()
    #     assert type(design_id) == int
    #     random_0_to_35 = np.random.randint(0,36, size=batch_size)
    image_ids = np.where(design_ids_of_images == design_id)[0][0]
    #     [random_0_to_35]
    return image_ids
