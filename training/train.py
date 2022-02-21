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
from contextlib import ExitStack
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils import training_utils
from utils.training_utils import freeze_model
from training import evaluate
from losses.losses import DensityRatioApproxKL, VehicleAttributeLoss, ChairAttributeLoss, MaskLoss, calc_loss, RatingLoss
from config import c, batch_size_dict

def train_combined_model(generator,
                         encoder,
                         predictor,
                         dataset,
                         session,
                         total_steps):
    '''
    Main training code for the model proposed in the paper.
    '''
    pbar = tqdm(initial=session.sample_i, total=total_steps)
    batch_count = 0
    # if session.c.start_phase > 0:
    #     first_phase = True
    kl_div = DensityRatioApproxKL()
    rating_loss = RatingLoss()
    session.batch_size = batch_size_dict(session.cur_resolution)
    if session.c.dataset == "vehicles":
        label_loss_criterion = VehicleAttributeLoss()
    elif session.c.dataset == "chairs":
        label_loss_criterion = ChairAttributeLoss()
    else:
        raise ValueError

    h_sampled_variable = Variable(torch.FloatTensor(session.batch_size, session.c.n_latents)).cuda(non_blocking=(session.c.gpu_count > 1))
    if session.c.use_masks:
        mask_loss_criterion = MaskLoss()

    with session.comet_experiment.train() if session.c.use_cometML else ExitStack():
        while session.sample_i < total_steps:

            #######################  Training Maintenance #######################

            sample_i_current_stage = session.sample_i - session.phase * session.c.images_per_stage

            # Advance to next resolution
            if sample_i_current_stage >= session.c.images_per_stage:
                try:
                    del batch_x_var, batch_attributes, batch_images
                    if session.c.use_masks:
                        del batch_masks
                except:
                    raise Exception("TODO: check exceptions for edge cases for open source code commit")
                session.phase += 1
                session.cur_resolution = 4 * 2 ** session.phase
                session.update_hyperparameters()
                session.batch_size = batch_size_dict(session.cur_resolution)
                h_sampled_variable = Variable(torch.FloatTensor(session.batch_size, session.c.n_latents)).cuda(non_blocking=(session.c.gpu_count > 1))
                # first_phase = False

            # Switch from fade-training to stable-training
            if sample_i_current_stage >= session.c.images_per_stage / 2:
                match_x = session.c.match_x_scale
            else:
                match_x = session.c.match_x_fade_in_scale

            session.alpha = min(sample_i_current_stage * 2.0 / session.c.images_per_stage, 1)


            training_stats = {}
            training_stats['KL_y_rating'] = 0.0
            # kl_divergence_statistics = ""
            try:
                training_stats['label_loss'] += 0.0
            except KeyError:
                training_stats['label_loss'] = 0.0


            try:
                training_stats['mask_error'] += 0.0
                # stats['mask_sampled_error'] += 0.0
            except KeyError:
                training_stats['mask_error'] = 0.0
                # stats['mask_sampled_error']= 0.0

            ############################################################################################################
            #           ENCODER MODEL / PREDICTIVE MODEL
            ############################################################################################################
            freeze_model(encoder, False)
            freeze_model(predictor, False)
            freeze_model(generator, True)
            encoder.zero_grad()

            labeled_data = session.c.use_semisupervised_predictive_loss and \
                           (session.phase >= c.pred_loss_start_phase) and \
                           (batch_count % session.c.labeled_data_cycle == 0)

            if not labeled_data:
                # Unlabelled data
                if session.c.use_masks:
                    batch_images, batch_attributes, batch_masks = dataset(batch_size=session.batch_size,
                                                                          phase=session.phase,
                                                                          alpha=session.alpha,
                                                                          with_masks=True,
                                                                          with_ratings=False,
                                                                          side_view=session.c.train_side_view_only)
                else:
                    batch_images, batch_attributes = dataset(batch_size=session.batch_size,
                                                             phase=session.phase,
                                                             alpha=session.alpha,
                                                             with_masks=False,
                                                             with_ratings=False,
                                                             side_view=session.c.train_side_view_only)

            else:
                if session.c.use_masks:
                    batch_images, batch_attributes, batch_masks, batch_ratings = dataset(batch_size=session.batch_size,
                                                                                         phase=session.phase,
                                                                                         alpha=session.alpha,
                                                                                         with_masks=True,
                                                                                         with_ratings=True,
                                                                                         data_split='train',
                                                                                         side_view=session.c.train_side_view_only)

                else:
                    batch_images, batch_attributes, batch_ratings = dataset(batch_size=session.batch_size,
                                                                            phase=session.phase,
                                                                            alpha=session.alpha,
                                                                            with_masks=False,
                                                                            with_ratings=True,
                                                                            data_split='train',
                                                                            side_view=session.c.train_side_view_only)

                batch_ratings = Variable(batch_ratings).cuda(non_blocking=(session.c.gpu_count > 1))


            batch_images = Variable(batch_images).cuda(non_blocking=(session.c.gpu_count > 1))
            if session.c.conditional_model:
                batch_attributes = Variable(batch_attributes).cuda(non_blocking=(session.c.gpu_count > 1)).add(session.c.epsilon)
            if session.c.use_masks:
                batch_masks = Variable(batch_masks).cuda(non_blocking=(session.c.gpu_count > 1))
                if session.c.inject_masks_into_encoder:
                    batch_x_var = torch.cat((batch_images, batch_masks), dim=1).cuda(non_blocking=(session.c.gpu_count > 1))
                else:
                    batch_x_var = batch_images
            else:
                batch_x_var = batch_images

            e_losses = []
            p_losses = []

            if session.c.conditional_model:
                # if session.c.std_vae_loss:
                if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                    if session.c.cond_h_on_y:
                        real_h, attributes_hat,  concat_x_a, real_mu, real_logvar = encoder(batch_x_var,
                                                                      session.phase,
                                                                      session.alpha,
                                                                      attributes=batch_attributes,
                                                                      return_attributes_hat=True,
                                                                      return_concat_x_a=True)

                    else:
                        real_h, attributes_hat, real_mu, real_logvar = encoder(batch_x_var,
                                                                      session.phase,
                                                                      session.alpha,
                                                                      attributes=batch_attributes,
                                                                      return_attributes_hat=True)
                
                else:
                    if session.c.cond_h_on_y:
                        real_h, attributes_hat, concat_x_a = encoder(batch_x_var,
                                                                     session.phase,
                                                                     session.alpha,
                                                                     attributes=batch_attributes,
                                                                     return_attributes_hat=True,
                                                                     return_concat_x_a=True)

                    else:
                        real_h, attributes_hat = encoder(batch_x_var,
                                                         session.phase,
                                                         session.alpha,
                                                         attributes=batch_attributes,
                                                         return_attributes_hat=True)

            else:
                if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                    real_h, real_mu, real_logvar = encoder(batch_x_var,
                                                           session.phase,
                                                           session.alpha,
                                                           attributes=None,
                                                           return_attributes_hat=False)
                else:
                    real_h = encoder(batch_x_var,
                                     session.phase,
                                     session.alpha,
                                     attributes=None,
                                     return_attributes_hat=False)

            real_h_var = Variable(real_h.clone().data).cuda(non_blocking=(session.c.gpu_count > 1))
            real_h_var.requires_grad = False

            if session.c.use_label_loss:
                if session.phase >= session.c.label_loss_start_phase:  # only start with 32x32
                    if session.c.conditional_model:
                        label_loss = label_loss_criterion(attributes_hat, batch_attributes) * session.c.label_loss_scale
                        e_losses.append(label_loss)
                        try:
                            training_stats['label_loss'] += label_loss.item()
                        except KeyError:
                            training_stats['label_loss'] = label_loss.item()

            if session.phase >= session.c.pred_loss_start_phase:
                if session.c.cond_h_on_y:
                    y_hat, y_mu, y_logvar = predictor(concat_x_a)
                else:
                    y_hat, y_mu, y_logvar = predictor(real_h)

                if labeled_data:

                    if not session.c.train_using_y_mu:
                        predictive_loss = rating_loss(batch_ratings, y_hat)
                    else:
                        predictive_loss = rating_loss(batch_ratings, y_mu + session.c.base_rating)
                    predict_loss_scaled = predictive_loss * session.c.pred_loss_scale
                    p_losses.append(predict_loss_scaled)
                    pred_loss_train = predictive_loss.item()

                    # TODO - refactor into losses
                    if not session.c.train_using_y_mu:
                        kl_y_rating, _, _ = kl_div(y_mu)
                        kl_y_rating =  kl_y_rating * session.c.y_KL_scale

                    else:
                        kl_y_rating = torch.mean( -0.5 * torch.sum(1 + y_logvar - y_mu.pow(2) - y_logvar.exp(), dim=1), dim=0) * session.c.y_KL_scale

                    p_losses.append(kl_y_rating)
                    # e_losses.append(kl_y_rating)
                    training_stats['KL_y_rating'] = kl_y_rating.item()

            if session.c.use_real_x_KL:
                if session.c.vae_loss == 'both':
                    # TODO: Alex Dec 8, 2021 - For now, defaulting to only recording stdvae KL for "both" condition
                    KL_real_dr, real_mu_dr, real_logvar_dr = kl_div(real_mu)
                    KL_real_dr = KL_real_dr * session.c.real_x_KL_dr_scale

                    KL_real_stdvae = torch.mean(-0.5 * torch.sum(1 + real_logvar - real_mu.pow(2) - real_logvar.exp(), dim=1), dim=0) * session.c.real_x_KL_stdvae_scale

                    KL_real = KL_real_stdvae * (1.0 - session.c.dr_to_stdvae_ratio) + KL_real_dr * session.c.dr_to_stdvae_ratio

                elif session.c.vae_loss == 'densityratio':
                    KL_real, real_mu, real_logvar = kl_div(real_h)
                    KL_real = KL_real * session.c.real_x_KL_dr_scale
                else:
                    KL_real = torch.mean( -0.5 * torch.sum(1 + real_logvar - real_mu.pow(2) - real_logvar.exp(), dim=1), dim=0) * session.c.real_x_KL_stdvae_scale

            # The final entries are the label. Normal case, just 1. Extract it/them, and make it [b x 1]:
            if session.c.conditional_model:
                real_h_split, _ = training_utils.split_attributes_out(real_h)
                if session.c.use_masks:
                    recon_x, recon_mask = generator(real_h_split,
                                                    session.phase,
                                                    session.alpha,
                                                    attributes=batch_attributes,
                                                    transform_onehot_attributes=True,
                                                    return_mask=True)
                else:
                    recon_x = generator(real_h_split,
                                        session.phase,
                                        session.alpha,
                                        attributes=batch_attributes,
                                        transform_onehot_attributes=True)
            else:
                real_h_split, attributes = training_utils.split_attributes_out(real_h)
                recon_x = generator(real_h_split,
                                    attributes,
                                    session.phase,
                                    session.alpha,
                                    transform_onehot_attributes=False,
                                    return_mask=False)

            if session.c.use_loss_x_reco:
                recon_loss = calc_loss(recon_x, batch_images, session.c.match_x_metric) * match_x
                e_losses.append(recon_loss)
                try:
                    training_stats['x_reconstruction_error'] += recon_loss.item()
                except KeyError:
                    training_stats['x_reconstruction_error'] = recon_loss.item()

            if session.c.use_masks:
                mask_loss = mask_loss_criterion(recon_mask.squeeze(1), batch_masks) * session.c.mask_loss_scale
                e_losses.append(mask_loss)
                try:
                    training_stats['mask_error'] += mask_loss.item()
                except KeyError:
                    training_stats['mask_error'] = mask_loss.item()

            if session.c.use_loss_fake_D_KL:
                h_sampled = torch.randn_like(h_sampled_variable)
                if session.c.noise == 'sphere':
                    h_sampled = training_utils.normalize(h_sampled, p=2, dim=1, eps=session.c.epsilon)

                if session.c.conditional_model:
                    h_sampled_split, split_attributes = training_utils.split_attributes_out(h_sampled)
                    fake, fake_mask, transformed_labels = generator(h_sampled_split,
                                                                     session.phase,
                                                                     session.alpha,
                                                                     attributes=batch_attributes,
                                                                     transform_onehot_attributes=True,
                                                                     return_mask=True,
                                                                     return_transformed_attributes=True)
                    if session.c.inject_masks_into_encoder:
                        fake = torch.cat((fake, fake_mask),dim=1)
                    fake = fake.detach()

                    # if session.c.std_vae_loss:
                    if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                        egh, fake_mu, fake_logvar = encoder(fake,
                                                            session.phase,
                                                            session.alpha,
                                                            attributes=batch_attributes)
                    else:
                        egh = encoder(fake,
                                      session.phase,
                                      session.alpha,
                                      attributes=batch_attributes)

                else:
                    h_sampled, attributes = training_utils.split_attributes_out(h_sampled)
                    fake = generator(h_sampled,
                                     session.phase,
                                     session.alpha,
                                     attributes=attributes,
                                     transform_onehot_attributes=False,
                                     return_mask=False).detach()

                    if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                        egh, fake_mu, fake_logvar = encoder(fake,
                                                            session.phase,
                                                            session.alpha,
                                                            attributes=None)  # e(g(h))
                    else:
                        egh = encoder(fake,
                                      session.phase,
                                      session.alpha,
                                      attributes=None)  # e(g(h))

                if session.c.vae_loss == 'both':
                    # TODO: Alex Dec 8, 2021 - For now, defaulting to only recording stdvae KL for "both" condition
                    KL_fake_dr, fake_mu_dr, fake_logvar_dr = kl_div(fake_mu)
                    KL_fake_dr = KL_fake_dr * session.c.fake_D_KL_dr_scale

                    KL_fake_stdvae = torch.mean(-0.5 * torch.sum(1 + fake_logvar - fake_mu.pow(2) - fake_logvar.exp(), dim=1), dim=0) * session.c.fake_D_KL_stdvae_scale

                    KL_fake = KL_fake_stdvae * (1.0 - session.c.dr_to_stdvae_ratio) + KL_fake_dr * session.c.dr_to_stdvae_ratio
                    KL_fake = -1.0 * KL_fake

                elif session.c.vae_loss == 'densityratio':
                    KL_fake, fake_mu, fake_logvar = kl_div(egh)
                    KL_fake = -1.0 * KL_fake * session.c.fake_D_KL_dr_scale

                else:
                    KL_fake = -1.0 * torch.mean(-0.5 * torch.sum(1 + fake_logvar - fake_mu.pow(2) - fake_logvar.exp(), dim=1), dim=0) * session.c.fake_D_KL_stdvae_scale

                e_losses.append(KL_real)
                e_losses.append(KL_fake)

            training_stats['real_mean'] = real_mu.mean().item()
            training_stats['real_var'] = real_logvar.exp().mean().item()
            training_stats['KL_real'] = KL_real.item()
            training_stats['fake_mean'] = fake_mu.mean().item()
            training_stats['fake_var'] = fake_logvar.exp().mean().item()
            training_stats['KL_fake'] = -1.0 * KL_fake.item()

            # Update Encoder / Predictive Model Weights
            if len(e_losses) > 0:
                e_loss = sum(e_losses)
                e_loss_np = e_loss.item()

                e_loss.backward(retain_graph=True)

                try:
                    training_stats['E_loss'] += e_loss_np
                except KeyError:
                    training_stats['E_loss'] = e_loss_np

            if labeled_data:
                p_loss = sum(p_losses)
                p_loss.backward()
                session.optimizerE.step()
                session.optimizerP.step()

            # if session.c.empty_cache:
            #     torch.cuda.empty_cache()

            ############################################################################################################
            #           GENERATOR MODEL
            ############################################################################################################

            freeze_model(encoder, True)
            freeze_model(predictor, True)
            freeze_model(generator, False)
            

            for gen_ind in range(session.c.n_generator):
                generator.zero_grad()  # Before the gradient accumulation
                for ag_ind in range(session.c.n_accumulate_gradients):

                    g_losses = []

                    # TODO: Consider removing
                    # # Reconstruction error in pixel space / Only if running more autoencoder mode
                    # if session.c.use_loss_x_recon_in_gen:
                    #     g_losses.append(recon_loss * c.match_x_gen)

                    # Create fake h
                    h_sampled = torch.randn_like(h_sampled_variable)
                    if session.c.noise == 'sphere':
                        h_sampled = training_utils.normalize(h_sampled, p=2, dim=1, eps=session.c.epsilon)

                    # TODO: Consider removing
                    if session.c.use_gen_recon_min_chance > 0.0:
                        # if np.random.rand() > session.alpha or np.random.rand() > (1.0 - session.c.use_gen_recon_min_chance):
                        if np.random.rand() > (1.0 - session.c.use_gen_recon_min_chance):
                            h_sampled = real_h_var



                    if session.c.conditional_model:
                        h_sampled_split, split_attributes = training_utils.split_attributes_out(h_sampled)
                        fake, fake_mask, transformed_labels = generator(h_sampled_split,
                                                             session.phase,
                                                             session.alpha,
                                                             attributes=batch_attributes,
                                                             transform_onehot_attributes=True,
                                                             return_mask=True,
                                                             return_transformed_attributes=True)

                        if session.c.inject_masks_into_encoder:
                            fake = torch.cat((fake, fake_mask), dim=1)

                        if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                            egh_g, fake_mu, fake_logvar = encoder(fake,
                                                                session.phase,
                                                                session.alpha,
                                                                attributes=batch_attributes)  # e(g(h))
                        else:
                            egh_g = encoder(fake,
                                          session.phase,
                                          session.alpha,
                                          attributes=batch_attributes)  # e(g(h))

                    else:
                        h_sampled_split, split_attributes = training_utils.split_attributes_out(h_sampled)
                        fake = generator(h_sampled_split,
                                         session.phase,
                                         session.alpha,
                                         attributes=split_attributes,
                                         transform_onehot_attributes=False,
                                         return_mask=False)

                        # if not session.c.std_vae_loss:
                        if session.c.vae_loss == 'stdvae' or session.c.vae_loss == 'both':
                            egh_g, fake_mu, fake_logvar = encoder(fake,
                                                                  session.phase,
                                                                  session.alpha,
                                                                  attributes=None)
                        else:
                            egh_g = encoder(fake,
                                          session.phase,
                                          session.alpha,
                                          attributes=None)

                    if session.c.vae_loss == 'both':
                        # TODO: Alex Dec 8, 2021 - For now, defaulting to only recording stdvae KL for "both" condition
                        KL_fake_dr, fake_mu_dr, fake_logvar_dr = kl_div(fake_mu)
                        KL_fake_dr = KL_fake_dr * session.c.fake_G_KL_dr_scale

                        KL_fake_stdvae = torch.mean(-0.5 * torch.sum(1 + fake_logvar - fake_mu.pow(2) - fake_logvar.exp(), dim=1), dim=0) * session.c.fake_G_KL_stdvae_scale

                        KL_fake_g = KL_fake_stdvae * (1.0 - session.c.dr_to_stdvae_ratio) + KL_fake_dr * session.c.dr_to_stdvae_ratio

                    elif session.c.vae_loss == 'densityratio':
                        KL_fake_g, fake_mu, fake_logvar = kl_div(egh_g)
                        KL_fake_g = KL_fake_g * session.c.fake_G_KL_dr_scale

                    else:
                        KL_fake = torch.mean(-0.5 * torch.sum(1 + fake_logvar - fake_mu.pow(2) - fake_logvar.exp(), dim=1), dim=0) * session.c.fake_G_KL_stdvae_scale

                    if session.c.use_loss_KL_egh:
                        g_losses.append(KL_fake_g)  # G minimizes this KL

                        try:
                            training_stats['KL(EGh)'] += KL_fake_g.item()
                        except KeyError:
                            training_stats['KL(EGh)'] = KL_fake_g.item()

                    if len(g_losses) > 0:
                        g_loss = sum(g_losses)
                        g_loss_np = g_loss.item()
                    try:
                        training_stats['G_loss'] += g_loss_np
                    except KeyError:
                        training_stats['G_loss'] = g_loss_np

                    g_loss.backward()

                session.optimizerG.step()

            if session.c.empty_cache:
                torch.cuda.empty_cache()

            ########################  Statistics ########################
            num_g = float(session.c.n_accumulate_gradients * session.c.n_generator)
            num_e = float(session.c.n_accumulate_gradients * session.c.n_critic)
            bs = session.batch_size * session.c.n_accumulate_gradients

            training_stats['E_loss'] /= num_e
            training_stats['x_reconstruction_error'] /= num_e
            training_stats['KL_real'] /= num_e
            training_stats['fake_mean'] /= num_e
            training_stats['fake_var'] /= num_e
            training_stats['KL_fake'] /= num_e
            training_stats['label_loss'] /= num_e
            training_stats['mask_error'] /= num_e
            training_stats['G_loss'] /= num_g
            training_stats['KL(EGh)'] /= num_g
            training_stats['match_x'] = match_x
            kl_divergence_statistics = "{0:.6f}/{1:.6f}/{2:.6f}".format(training_stats['KL_real'],
                                                                        training_stats['KL_fake'],
                                                                        training_stats['KL(EGh)'])

            if session.c.use_cometML:
                session.comet_experiment.log_metrics(training_stats, step=session.sample_i)

            cur_batch_stats = 'batch_num: {0}; sample_it: {1}; phase: {2}; batch_size: {3:.1f}; Alpha: {4:.3f}; Res: {5}; E_loss: {6:.4f}; ' \
                              'G_Loss: {7:.4f}; Label_Loss: {12:.4f}, Mask_Loss: {13:.4f}, KL(real/fake/EGh): {8}; KL_y {14:.4f}; x-reco {9:.4f}; real_var {10:.4f}, match_x {11:.2f}'.format(
                              batch_count + 1,
                              session.sample_i + 1,
                              session.phase,
                              bs,
                              session.alpha,
                              session.cur_resolution,
                              training_stats['E_loss'],
                              training_stats['G_loss'],
                              kl_divergence_statistics,
                              # training_stats['h_reconstruction_error'],
                              training_stats['x_reconstruction_error'],
                              training_stats['real_var'],
                              match_x,
                              training_stats['label_loss'],
                              training_stats['mask_error'],
                              training_stats['KL_y_rating'])

            pbar.set_description(cur_batch_stats)
            pbar.update(bs)

            if batch_count % session.c.print_frequency == 0:
                print(cur_batch_stats)

            session.sample_i += bs

            # Predictive Evaluation
            if labeled_data:
                del batch_x_var, batch_attributes, batch_ratings, batch_images
                if session.c.use_masks:
                    del batch_masks

                pred_loss_valid = evaluate.evaluate_combined_model(dataset, session, data_split='valid')
                if session.c.evaluate_full_test_set:
                    pred_loss_test = evaluate.evaluate_combined_model(dataset, session, data_split='test')
                    print("Prediction Losses- Train: {:.4f} Valid: {:.4f} Test: {:.4f}".format(pred_loss_train, pred_loss_valid, pred_loss_test))
                else:
                    print("Prediction Losses- Train: {:.4f} Valid: {:.4f}".format(pred_loss_train, pred_loss_valid))
                if session.c.use_cometML:
                    session.comet_experiment.log_metric('Pred_Loss_Train', pred_loss_train, step=session.sample_i)
                    session.comet_experiment.log_metric('Pred_Loss_Valid', pred_loss_valid, step=session.sample_i)
                    if session.c.evaluate_full_test_set:
                        session.comet_experiment.log_metric('Pred_Loss_Test', pred_loss_test, step=session.sample_i)

            ########################  Saving ########################

            if batch_count % session.c.checkpoint_cycle == 0:
                if session.c.save_checkpoints and (batch_count % session.c.save_checkpoint_cycle == 0):
                    for postfix in {'latest', str(session.sample_i).zfill(6)}:
                        session.save_all('{}/{}_state'.format(session.c.checkpoint_dir, postfix))

                    print("Checkpointed to {}".format(session.sample_i))

                ########################  Generative Sampling  ########################
                try:
                    evaluate.tests_run(generator, encoder, dataset, session, writer=None,
                                       reconstruction=(batch_count % session.c.checkpoint_cycle == 0),
                                       interpolation=(batch_count % session.c.checkpoint_cycle == 0),
                                       collated_sampling=(batch_count % session.c.checkpoint_cycle == 0),
                                       individual_sampling=False
                                       )
                except (OSError, StopIteration):
                    print("Skipped periodic tests due to an exception.")

            batch_count += 1

    pbar.close()


