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

import h5py
import numpy as np
# from comet_ml import Experiment
import torch
# from torch.utils.data import DataLoader
# from torch.utils import datadata.Dataset
from torch.autograd import Variable

import config
c = config.c


class Vehicles(torch.utils.data.Dataset):
    def __init__(self,
                 use_RAM,
                 train_x=None,
                 train_y=None,
                 valid_x=None,
                 valid_y=None,
                 test_x=None,
                 test_y=None,
                 c=None):

        self.use_RAM = use_RAM
        if train_x is not None:
            self.train_x = train_x
            self.train_y = train_y
            self.valid_x = valid_x
            self.valid_y = valid_y
            self.test_x = test_x
            self.test_y = test_y

        assert c is not None
        self.c = c

        # Load attributes
        attribute_labels = np.load(c.labels_dir)
        years = attribute_labels['arr_0']  # 15 years
        makes = attribute_labels['arr_1']  # 48 makes
        models = attribute_labels['arr_2']  # 23 models # TODO remove
        bodytypes = attribute_labels['arr_3']  # 20 bodytypes
        views = attribute_labels['arr_4']  # 2 variables for viewpoint
        design_ids = attribute_labels['arr_5']  # unique design_ids - merge with Artem resized images
        colors = attribute_labels['arr_6']  # 3 variables for viewpoint
        colors = colors / 127.5 - 1.0
        attributes = np.concatenate((years, makes, models, bodytypes, views, colors), axis=1)
        self.attributes = attributes.astype(np.float32)

        # self.image_mean_value = 225.36372369165196

        resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64',
                      'data128x128', 'data256x256', 'data512x512']

        self._base_key = 'data'
        self._base_masks_key = 'masks'
        if self.use_RAM:
            print('Loading Images into RAM...')
            self.dataset = h5py.File(self.c.images_dir, 'r', driver='core')
            if self.c.use_masks:
                self.masks = h5py.File(self.c.masks_dir, 'r', driver='core')
            print('Done loading Images into RAM...')
        else:
            self.dataset = h5py.File(self.c.images_dir, 'r')
            if self.c.use_masks:
                self.masks = h5py.File(self.c.masks_dir, 'r')
        self._len = {k: len(self.dataset[k]) for k in resolution}
        self.num_data = self._len['data8x8']
        assert all([resol in self.dataset.keys() for resol in resolution])

        # # Training image inds - OLD
        # self.design_ids_of_images = design_ids
        # valid_design_ids, test_design_ids = np.unique(self.valid_x), np.unique(self.test_x)
        # held_out_design_ids = np.concatenate((valid_design_ids, test_design_ids))
        # train_bool_array = ~np.isin(self.design_ids_of_images, held_out_design_ids)
        # self.training_image_inds = np.nonzero(train_bool_array)[0]

        # Training image inds
        # image_ids outside train, valid, test - take % of that
        # then concat back with train_image_inds as training_image_inds
        self.design_ids_of_images = design_ids
        train_design_ids, valid_design_ids, test_design_ids = np.unique(self.train_x), np.unique(self.valid_x), np.unique(self.test_x)
        train_bool_array = np.isin(self.design_ids_of_images, train_design_ids)
        self.training_image_inds = np.nonzero(train_bool_array)[0]
        num_training_images = len(self.training_image_inds)
        self.training_mean = np.mean(np.abs(self.valid_y - self.train_y.mean()))
        self.training_median = np.mean(np.abs(self.valid_y - np.median(self.train_y)))
        self.training_mid = np.mean(np.abs(self.valid_y - 3.0))

        labeled_design_ids = np.concatenate((train_design_ids, valid_design_ids, test_design_ids))
        unlabeled_bool_array = ~np.isin(self.design_ids_of_images, labeled_design_ids)
        self.unlabeled_image_inds = np.nonzero(unlabeled_bool_array)[0]
        if self.c.percentage_of_unlabeled_data != 1.0: # 0.5 # 1.0, 0.5, 0.25; 1.0 is full training set
            num_unlabeled = int(self.c.percentage_of_unlabeled_data * len(self.unlabeled_image_inds))
            unlabeled_image_inds = np.random.choice(self.unlabeled_image_inds, size=num_unlabeled, replace=False)
            self.training_image_inds = np.concatenate((self.training_image_inds, unlabeled_image_inds))

        # SUV/CUV training only
        if self.c.suv_cuv_only:
            bodytypes_raveled = np.nonzero(bodytypes)[1]
            suv_cuv_inds = np.where((bodytypes_raveled == 6) | (bodytypes_raveled == 7))[0]
            self.training_image_inds = np.intersect1d(suv_cuv_inds, self.training_image_inds)

        print("\nTotal Unlabeled Data at {}%: {}".format(int(self.c.percentage_of_unlabeled_data*100),
                                                         len(self.training_image_inds)-num_training_images))

        # Sideview image ids
        self.random_image_idxs = np.array([0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 33, 34, 35])

    def get_random_image_ids_given_design_ids(self, design_ids):
        random_sideish_view = np.random.choice(self.random_image_idxs)
        image_ids = np.array([
            np.where(self.design_ids_of_images == elm)[0][random_sideish_view] for elm in design_ids])
        return image_ids.flatten()

    def get_side_image_id_given_design_ids(self, design_ids):
        image_ids = np.array([
            np.where(self.design_ids_of_images == elm)[0][0] for elm in design_ids])
        return image_ids.flatten()

    def __call__(self,
                 batch_size,
                 phase,
                 alpha,
                 with_masks=True,
                 with_ratings=False,
                 data_split='train',
                 batch_ind=None,
                 side_view=False):

        size = 4 * (2 ** phase)
        key = self._base_key + '{}x{}'.format(size, size)

        if not with_ratings:
            # idx = np.random.randint(self.num_data, size=batch_size)
            image_ids = np.random.choice(self.training_image_inds, size=batch_size, replace=True)
        else:
            if data_split == 'train':
                if batch_ind is None:
                    inds = np.random.randint(self.train_x.shape[0], size=batch_size)
                else:
                    inds = np.arange(self.train_x.shape[0])[batch_ind * batch_size: (batch_ind + 1) * batch_size]
                design_ids = self.train_x[inds]
                batch_ratings = self.train_y[inds].astype(np.float32)

            elif data_split == 'valid':
                if batch_ind is None:
                    inds = np.random.randint(self.valid_x.shape[0], size=batch_size)
                else:
                    inds = np.arange(self.valid_x.shape[0])[batch_ind * batch_size: (batch_ind + 1) * batch_size]
                design_ids = self.valid_x[inds]
                batch_ratings = self.valid_y[inds].astype(np.float32)

            elif data_split == 'test':
                if batch_ind is None:
                    inds = np.random.randint(self.test_x.shape[0], size=batch_size)
                else:
                    inds = np.arange(self.test_x.shape[0])[batch_ind * batch_size: (batch_ind + 1) * batch_size]
                design_ids = self.test_x[inds]
                batch_ratings = self.test_y[inds].astype(np.float32)

            batch_ratings = torch.from_numpy(batch_ratings)
            batch_ratings = batch_ratings.reshape(-1, 1)

            if not side_view:
                image_ids = self.get_random_image_ids_given_design_ids(design_ids)
            else:
                image_ids = self.get_side_image_id_given_design_ids(design_ids)

        hi_res_batch_images = np.array([self.dataset[key][i] / 127.5 - 1.0 for i in image_ids], dtype=np.float32)
        batch_attributes = np.array([self.attributes[i] for i in image_ids], dtype=np.float32)
        if with_masks:
            key = self._base_masks_key + '{}x{}'.format(size, size)
            # hi_res_batch_masks = np.array([self.masks[key][i] / 127.5 - 1.0 for i in idx], dtype=np.float32)
            batch_masks = np.array([self.masks[key][i] / 255.0 for i in image_ids], dtype=np.float32)
            batch_masks = torch.from_numpy(batch_masks)

        if alpha < 1.0 and phase > 0:
            lr_key = self._base_key + '{}x{}'.format(size // 2, size // 2)
            low_res_batch_images = np.array([self.dataset[lr_key][i] / 127.5 - 1.0 for i in image_ids],
                                             dtype=np.float32).repeat(2, axis=2).repeat(2, axis=3)
            batch_images = hi_res_batch_images * alpha + low_res_batch_images * (1.0 - alpha)
        else:
            batch_images = hi_res_batch_images


        batch_images = torch.from_numpy(batch_images)
        batch_attributes = torch.from_numpy(batch_attributes)

        if not with_ratings:
            if self.c.conditional_model:
                if with_masks:
                    return batch_images, batch_attributes, batch_masks
                else:
                    return batch_images, batch_attributes
            else:
                if self.c.use_masks:
                    return batch_images, batch_masks
                else:
                    return batch_images
        else:
            if self.c.conditional_model:
                if with_masks:
                    return batch_images, batch_attributes, batch_masks, batch_ratings
                else:
                    return batch_images, batch_attributes, batch_ratings
            else:
                if with_masks:
                    return batch_images, batch_masks, batch_ratings
                else:
                    return batch_images, batch_ratings

    def __len__(self):
        return self.num_data

    def __getitem__(self, index, size=512, with_attributes=False, astorch=False):
        # size = 4 * (2 ** phase)
        key = self._base_key + '{}x{}'.format(size, size)
        image = self.dataset[key][index] / 127.5 - 1.0
        attributes = self.attributes[index]
        image = image.astype(np.float32)
        attributes = attributes.astype(np.float32)

        if astorch:
            image, attributes = torch.from_numpy(image), torch.from_numpy(attributes)
        if with_attributes:
            return image, attributes
        else:
            return image

# TODO Refactor this
class DataGenerator(object):
    def __init__(self):
        pass
    @staticmethod
    def data_generator_phase0(dataloader, batch_size, image_size=4):
        while True:  # This is an infinite iterator
            if c.conditional_model:
                if c.use_masks:
                    batch_images, batch_attributes, batch_masks = dataloader(batch_size, int(np.log2(image_size / 4)), 0.0)
                    # yield torch.from_numpy(batch_images), torch.from_numpy(batch_attributes), torch.from_numpy(batch_masks)
                    yield batch_images, batch_attributes, batch_masks
                else:
                    batch_images, batch_attributes = dataloader(batch_size, int(np.log2(image_size / 4)), 0.0)
                    yield batch_images, batch_attributes
            else:
                batch_images = dataloader(batch_size, int(np.log2(image_size / 4)), 0.0)
                yield batch_images, None  # no label

    @staticmethod
    def data_generator_session(dataloader, batch_size, image_size, session):
        """ This is another version of sample data that instead uses session information
        """
        while True:  # This is an infinite iterator
            if c.conditional_model:
                if c.use_masks:
                    batch_images, batch_attributes, batch_masks = dataloader(batch_size, session.phase, session.alpha)
                    yield batch_images, batch_attributes, batch_masks
                else:
                    batch_images, batch_attributes = dataloader(batch_size, session.phase, session.alpha)
                    yield batch_images, batch_attributes
            else:
                batch = dataloader(batch_size, session.phase, session.alpha)
                yield batch, None  # no label


# # -------------------- Data Loader for Train/Valid/Test Sets ------------------
# class DataLoader():
#
#     def __init__(self,
#                  x,
#                  y,
#                  batch_size,
#                  design_ids_of_images,
#                  images_pretrain,
#                  side_view=False,
#                  training=False):
#         self.x = x
#         self.y = y
#         self.batch_size = batch_size
#         self.num_batches = int(x.shape[0] / batch_size)
#         self.design_ids_of_images = design_ids_of_images
#         self.images_pretrain = images_pretrain
#         self.side_view = side_view
#         self.training = training
#         self.random_image_idxs = np.array([0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 33, 34, 35])
#
#     def get_random_image_ids_given_design_ids(self, design_ids):
#         random_sideish_view = np.random.choice(self.random_image_idxs)
#         image_ids = np.array([
#             np.where(
#                 self.design_ids_of_images == design_ids[elm])[0][random_sideish_view]
#             for elm in range(self.batch_size)
#         ])
#         return image_ids.flatten()
#
#     def get_side_image_id_given_design_id(self, design_ids):
#         image_ids = np.array([
#             np.where(self.design_ids_of_images == design_ids[elm])[0][0]
#             for elm in range(self.batch_size)
#         ])
#         return image_ids
#
#     def _numpy2var(self, x):
#         x = x.astype(np.float32)
#         var = Variable(torch.from_numpy(x))
#         var = var.cuda()
#         return var
#
#     def __call__(self):
#         '''
#         Generator for batch images and labels
#         '''
#         for idx in range(self.num_batches):
#             batch_x_ids = self.x[idx * self.batch_size:(idx + 1) *
#                                                        self.batch_size]
#             batch_y = self.y[idx * self.batch_size:(idx + 1) *
#                                                    self.batch_size].reshape(-1, 1)
#             if not self.side_view:
#                 image_ids = self.get_random_image_ids_given_design_ids(
#                     batch_x_ids)
#
#             else:
#                 image_ids = self.get_side_image_id_given_design_id(batch_x_ids)
#             # batch_images, batch_attributes = self.images_pretrain(image_ids, training = self.training)
#             batch_images, batch_attributes = self.images_pretrain(image_ids, size=c.image_size, training=self.training)
#             yield self._numpy2var(batch_images), self._numpy2var(
#                 batch_y), self._numpy2var(batch_attributes)
#
#
# class VehiclesforPretrainedModel():
#
#     def __init__(self,# # -------------------- Data Loader for Train/Valid/Test Sets ------------------
# class DataLoader():
#
#     def __init__(self,
#                  x,
#                  y,
#                  batch_size,
#                  design_ids_of_images,
#                  images_pretrain,
#                  side_view=False,
#                  training=False):
#         self.x = x
#         self.y = y
#         self.batch_size = batch_size
#         self.num_batches = int(x.shape[0] / batch_size)
#         self.design_ids_of_images = design_ids_of_images
#         self.images_pretrain = images_pretrain
#         self.side_view = side_view
#         self.training = training
#         self.random_image_idxs = np.array([0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 33, 34, 35])
#
#     def get_random_image_ids_given_design_ids(self, design_ids):
#         random_sideish_view = np.random.choice(self.random_image_idxs)
#         image_ids = np.array([
#             np.where(
#                 self.design_ids_of_images == design_ids[elm])[0][random_sideish_view]
#             for elm in range(self.batch_size)
#         ])
#         return image_ids.flatten()
#
#     def get_side_image_id_given_design_id(self, design_ids):
#         image_ids = np.array([
#             np.where(self.design_ids_of_images == design_ids[elm])[0][0]
#             for elm in range(self.batch_size)
#         ])
#         return image_ids
#
#     def _numpy2var(self, x):
#         x = x.astype(np.float32)
#         var = Variable(torch.from_numpy(x))
#         var = var.cuda()
#         return var
#
#     def __call__(self):
#         '''
#         Generator for batch images and labels
#         '''
#         for idx in range(self.num_batches):
#             batch_x_ids = self.x[idx * self.batch_size:(idx + 1) *
#                                                        self.batch_size]
#             batch_y = self.y[idx * self.batch_size:(idx + 1) *
#                                                    self.batch_size].reshape(-1, 1)
#             if not self.side_view:
#                 image_ids = self.get_random_image_ids_given_design_ids(
#                     batch_x_ids)
#
#             else:
#                 image_ids = self.get_side_image_id_given_design_id(batch_x_ids)
#             # batch_images, batch_attributes = self.images_pretrain(image_ids, training = self.training)
#             batch_images, batch_attributes = self.images_pretrain(image_ids, size=c.image_size, training=self.training)
#             yield self._numpy2var(batch_images), self._numpy2var(
#                 batch_y), self._numpy2var(batch_attributes)
#
#
# class VehiclesforPretrainedModel():
#
#     def __init__(self,
#                  images_dir=None,
#                  labels_dir=None,
#                  use_RAM=False,
#                  bw_images=False,
#                  resize=None,
#                  augment_images=False):
#         self.use_RAM = use_RAM
#         self.resize = resize
#         self.bw_images = bw_images
#         self.augment_images = augment_images
#         assert images_dir is not None
#         assert labels_dir is not None
#         self.images_dir = images_dir
#         data_labels = np.load(labels_dir)
#         years = data_labels['arr_0']  # 15 years
#         makes = data_labels['arr_1']  # 48 makes
#         models = data_labels['arr_2']  # 23 models
#         bodytypes = data_labels['arr_3']  # 20 bodytypes
#         views = data_labels['arr_4']  # 2 variables for viewpoint
#         design_ids = data_labels['arr_5']  # 2 variables for viewpoint
#         labels = np.concatenate(
#             (years, makes, models, bodytypes, views), axis=1)
#         self.labels = labels.astype(np.float32)
#         resolution = [
#             'data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32',
#             'data64x64', 'data128x128', 'data256x256', 'data512x512'
#         ]
#         self._base_key = 'data'
#         if self.use_RAM:
#             print('Loading Images into RAM...')
#             self.dataset = h5py.File(images_dir, 'r', driver='core')
#             print('Done loading Images into RAM...')
#         else:
#             self.dataset = h5py.File(images_dir, 'r')
#         self._len = {k: len(self.dataset[k]) for k in resolution}
#
#         # if self.augment_images:
#         #     sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#         #     self.seq_augment = iaa.Sequential([
#         #         # iaa.Fliplr(0.5), # horizontally flip 50% of the images
#         #         iaa.Sometimes(0.5,
#         #                       iaa.GaussianBlur(sigma=(0, 0.75))
#         #                       ),
#         #         sometimes(
#         #             iaa.Affine(
#         #                 scale=(0.8, 1.2),  # scale images to 80-120% of their size, same for both per axis
#         #                 rotate=(-5.0, 5.0),  # rotate by -5 to +5 degrees
#         #                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
#         #                 order=1,
#         #                 cval=255,  # , # if mode is constant, use a cval between 0 and 255
#         #                 backend='cv2'
#         #                 # mode="constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#         #             ))
#         #     ])
#
#         assert all([resol in self.dataset.keys() for resol in resolution])
#
#     def gray2rgbshape(self, im):
#         # im.resize((im.shape[0], im.shape[1], 1))
#         # return np.repeat(im, 3, 2)
#         return np.dstack([im.astype(float)] * 3)
#
#     def __call__(self, idx, size=256, training=False):
#         key = self._base_key + '{}x{}'.format(size, size)
#         # idx = np.random.randint(self._len[key], size=batch_size)
#
#         if self.resize == size and self.bw_images == False:
#             batch_images = np.array([self.dataset[key][i] / 127.5 - 1.0 for i in idx], dtype=np.float32)
#         else:
#             batch_images = np.array([self.dataset[key][i] for i in idx], dtype=np.float32)
#             batch_images = batch_images.transpose(0, 2, 3, 1)
#
#             if self.resize and self.bw_images:
#                 batch_images = np.array([self.gray2rgbshape(rgb2gray(scipy.misc.imresize(image, size=self.resize) / 127.5 - 1.0))
#                                          for image in batch_images], dtype=np.float32)
#             elif self.resize and not self.bw_images:
#                 batch_images = np.array([scipy.misc.imresize(image, size=self.resize) for image in batch_images], dtype=np.float32)
#             elif not self.resize and self.bw_images:
#                 batch_images = np.array([self.gray2rgbshape(rgb2gray(image / 127.5 - 1.0)) for image in batch_images], dtype=np.float32)
#
#             if training and self.augment_images:
#                 batch_images = batch_images.astype('uint8')
#                 batch_images = self.seq_augment.augment_images(batch_images)
#                 batch_images = batch_images.astype('float32')
#
#             if not self.bw_images:
#                 batch_images /= 127.5
#                 batch_images -= 1.0
#
#             batch_images = batch_images.transpose(0, 3, 1, 2)
#
#         batch_labels = np.array([self.labels[i] for i in idx], dtype=np.float32)
#         return batch_images, batch_labels

#                  images_dir=None,
#                  labels_dir=None,
#                  use_RAM=False,
#                  bw_images=False,
#                  resize=None,
#                  augment_images=False):
#         self.use_RAM = use_RAM
#         self.resize = resize
#         self.bw_images = bw_images
#         self.augment_images = augment_images
#         assert images_dir is not None
#         assert labels_dir is not None
#         self.images_dir = images_dir
#         data_labels = np.load(labels_dir)
#         years = data_labels['arr_0']  # 15 years
#         makes = data_labels['arr_1']  # 48 makes
#         models = data_labels['arr_2']  # 23 models
#         bodytypes = data_labels['arr_3']  # 20 bodytypes
#         views = data_labels['arr_4']  # 2 variables for viewpoint
#         design_ids = data_labels['arr_5']  # 2 variables for viewpoint
#         labels = np.concatenate(
#             (years, makes, models, bodytypes, views), axis=1)
#         self.labels = labels.astype(np.float32)
#         resolution = [
#             'data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32',
#             'data64x64', 'data128x128', 'data256x256', 'data512x512'
#         ]
#         self._base_key = 'data'
#         if self.use_RAM:
#             print('Loading Images into RAM...')
#             self.dataset = h5py.File(images_dir, 'r', driver='core')
#             print('Done loading Images into RAM...')
#         else:
#             self.dataset = h5py.File(images_dir, 'r')
#         self._len = {k: len(self.dataset[k]) for k in resolution}
#
#         # if self.augment_images:
#         #     sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#         #     self.seq_augment = iaa.Sequential([
#         #         # iaa.Fliplr(0.5), # horizontally flip 50% of the images
#         #         iaa.Sometimes(0.5,
#         #                       iaa.GaussianBlur(sigma=(0, 0.75))
#         #                       ),
#         #         sometimes(
#         #             iaa.Affine(
#         #                 scale=(0.8, 1.2),  # scale images to 80-120% of their size, same for both per axis
#         #                 rotate=(-5.0, 5.0),  # rotate by -5 to +5 degrees
#         #                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
#         #                 order=1,
#         #                 cval=255,  # , # if mode is constant, use a cval between 0 and 255
#         #                 backend='cv2'
#         #                 # mode="constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#         #             ))
#         #     ])
#
#         assert all([resol in self.dataset.keys() for resol in resolution])
#
#     def gray2rgbshape(self, im):
#         # im.resize((im.shape[0], im.shape[1], 1))
#         # return np.repeat(im, 3, 2)
#         return np.dstack([im.astype(float)] * 3)
#
#     def __call__(self, idx, size=256, training=False):
#         key = self._base_key + '{}x{}'.format(size, size)
#         # idx = np.random.randint(self._len[key], size=batch_size)
#
#         if self.resize == size and self.bw_images == False:
#             batch_images = np.array([self.dataset[key][i] / 127.5 - 1.0 for i in idx], dtype=np.float32)
#         else:
#             batch_images = np.array([self.dataset[key][i] for i in idx], dtype=np.float32)
#             batch_images = batch_images.transpose(0, 2, 3, 1)
#
#             if self.resize and self.bw_images:
#                 batch_images = np.array([self.gray2rgbshape(rgb2gray(scipy.misc.imresize(image, size=self.resize) / 127.5 - 1.0))
#                                          for image in batch_images], dtype=np.float32)
#             elif self.resize and not self.bw_images:
#                 batch_images = np.array([scipy.misc.imresize(image, size=self.resize) for image in batch_images], dtype=np.float32)
#             elif not self.resize and self.bw_images:
#                 batch_images = np.array([self.gray2rgbshape(rgb2gray(image / 127.5 - 1.0)) for image in batch_images], dtype=np.float32)
#
#             if training and self.augment_images:
#                 batch_images = batch_images.astype('uint8')
#                 batch_images = self.seq_augment.augment_images(batch_images)
#                 batch_images = batch_images.astype('float32')
#
#             if not self.bw_images:
#                 batch_images /= 127.5
#                 batch_images -= 1.0
#
#             batch_images = batch_images.transpose(0, 3, 1, 2)
#
#         batch_labels = np.array([self.labels[i] for i in idx], dtype=np.float32)
#         return batch_images, batch_labels
