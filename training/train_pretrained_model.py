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
import time
import torch
from torch.autograd import Variable
from utils.training_utils import AverageMeter

def train_pretrained_model(dataset, session, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()

    session.model.train()
    session.model.zero_grad()
    session.optimizer.zero_grad()

    if session.c.finetune_model and epoch >= session.c.epoch_begin_finetune:
        if not hasattr(session, 'finetune_optimizer'):
        # if session.c.finetune_model:
            print("\n=========Beginning Finetuning=========\n")
            if session.c.experiment_type == 'features_and_pretrain' or session.c.experiment_type == 'pretrain_only':
                for param in session.model.pretrained_model.parameters():
                    param.requires_grad = True
                # if session.c.experiment_type == 'features_and_pretrain':
                #     for param in session.model.encoder_model.parameters():
                #         param.requires_grad = True
            elif session.c.experiment_type == 'encoder_only':
                for param in session.model.pretrained_model.parameters():
                    param.requires_grad = True
            else:
                raise Exception("No model to finetune!")

            session.finetune_optimizer = torch.optim.Adam(
                                        filter(lambda p: p.requires_grad, session.model.parameters()),
                                        session.c.lr_finetune,
                                        betas=session.c.adam_optim_betas_pretrain)

    print("\nTraining: Epoch {}".format(epoch))
    if session.c.finetune_model and epoch >= session.c.epoch_begin_finetune:
        print("Current Learning Rate {}:".format(session.finetune_optimizer.state_dict()['param_groups'][0]['lr']))
    else:
        print("Current Learning Rate {}:".format(session.optimizer.state_dict()['param_groups'][0]['lr']))

    end = time.time()
    # for i, (images, targets, labels) in enumerate(train_loader()):
    for batch_ind in range(session.c.num_batches_per_epoch_pretrained):
        batch_images, batch_attributes, batch_ratings = dataset(batch_size=session.batch_size,
                                                 phase=session.phase,
                                                 alpha=session.alpha,
                                                 with_masks=False,
                                                 with_ratings=True,
                                                 data_split='train',
                                                 side_view=session.c.train_side_view_only)

        if session.c.nc == 1:
            batch_images = batch_images.repeat_interleave(3, dim=1)

        batch_ratings = Variable(batch_ratings).cuda(non_blocking=(session.c.gpu_count > 1))
        batch_images = Variable(batch_images).cuda(non_blocking=(session.c.gpu_count > 1))

        data_time.update(time.time() - end)  # measure data loading time

        if session.c.experiment_type == "pretrain_only" and not session.c.pretrained_model_use_attributes:
            output = session.model(batch_images)
        elif session.c.experiment_type == "pretrain_only" and session.c.pretrained_model_use_attributes:
            batch_attributes = Variable(batch_attributes).cuda(non_blocking=(session.c.gpu_count > 1))
            output = session.model(batch_images, batch_attributes)
        elif session.c.experiment_type == "encoder_only":
            batch_attributes = Variable(batch_attributes).cuda(non_blocking=(session.c.gpu_count > 1))
            output = session.model(batch_images, batch_attributes)

        # measure accuracy and record loss
        loss = session.criterion_train(output, batch_ratings)


        batch_loss = loss.item()
        train_losses.update(batch_loss, batch_ratings.size(0))

         # compute gradient and do optimization
        if session.c.finetune_model and epoch >= session.c.epoch_begin_finetune:
            session.finetune_optimizer.zero_grad()
            loss.backward()
            session.finetune_optimizer.step()
            session.model.zero_grad()
            # self.session.optimizer.zero_grad()

        else:
            session.optimizer.zero_grad()
            loss.backward()
            session.optimizer.step()
            session.model.zero_grad()
            # self.optimizer.zero_grad()

        batch_time.update(time.time() - end)  # measure elapsed time
        end = time.time()

        if session.c.use_cometML:
            step = int(batch_ind*session.batch_size + epoch*session.c.num_batches_per_epoch_pretrained*session.batch_size)
            session.comet_experiment.log_metric("Train Loss", batch_loss, step=step)
        # (training_stats, step=session.sample_i)

        if batch_ind % session.c.num_batches_print_freq_train == 0:
            print('Train Batch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Rating Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch,
                batch_ind,
                session.c.num_batches_per_epoch_pretrained,
                batch_time=batch_time,
                data_time=data_time,
                loss=train_losses))
    print('Train Error:' + str(train_losses.avg))

    return session

def evaluate_pretrained_model(dataset, session, epoch, data_split='valid'):
    assert data_split in ['train', 'valid', 'test']
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    eval_type = "Validation" if data_split == 'valid' else "Test"
    evaluated_losses = AverageMeter()

    session.model.eval()
    session.model.regressor.eval()

    if session.c.evaluate_full_validation_set and data_split=='valid':
        num_batches = len(dataset.valid_y) // session.batch_size
    elif session.c.evaluate_full_validation_set and data_split=='test':
        num_batches = len(dataset.test_y) // session.batch_size
    elif session.c.evaluate_full_validation_set and data_split=='train':
        num_batches = len(dataset.train_y) // session.batch_size
    else:
        num_batches = session.c.num_random_batches_per_epoch_pretrained_evaluate

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

        if session.c.nc == 1:
            batch_images = batch_images.repeat_interleave(3, dim=1)

        batch_ratings = Variable(batch_ratings).cuda(non_blocking=(session.c.gpu_count > 1))
        batch_images = Variable(batch_images).cuda(non_blocking=(session.c.gpu_count > 1))

        data_time.update(time.time() - end)  # measure data loading time

        with torch.no_grad():
            if session.c.experiment_type == "pretrain_only" and not session.c.pretrained_model_use_attributes:
                output = session.model(batch_images)
            elif session.c.experiment_type == "pretrain_only" and session.c.pretrained_model_use_attributes:
                batch_attributes = Variable(batch_attributes).cuda(non_blocking=(session.c.gpu_count > 1))
                output = session.model(batch_images, batch_attributes)
            elif session.c.experiment_type == "encoder_only":
                batch_attributes = Variable(batch_attributes).cuda(non_blocking=(session.c.gpu_count > 1))
                output = session.model(batch_images, batch_attributes)


        # measure accuracy and record loss
        loss = session.criterion_evaluation(output, batch_ratings)
        batch_loss = loss.item()
        evaluated_losses.update(batch_loss, batch_ratings.size(0))

        batch_time.update(time.time() - end)  # measure elapsed time
        end = time.time()

        if session.c.use_cometML:
            step = int(batch_ind*session.batch_size + epoch*num_batches*session.batch_size)
            loss_name = "Valid Loss" if data_split=='valid' else "Test Loss"
            session.comet_experiment.log_metric(loss_name, batch_loss, step=step)

        if batch_ind % session.c.num_batches_print_freq_valid == 0:
            print('{0} Batch: [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
                eval_type,
                batch_ind,
                num_batches,
                batch_time=batch_time,
                data_time=data_time,
                losses=evaluated_losses))

    session.model.zero_grad()
    session.optimizer.zero_grad()
    print('{} Error: {}'.format(eval_type, evaluated_losses.avg))
    return evaluated_losses.avg

