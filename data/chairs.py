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
import torch
import config
c = config.c

class Chairs(torch.utils.data.Dataset):
    """
    This is the chair dataset for the open source / open data code release.
    It is different than the car dataset (primary dataset) in the paper due to
    data mapping, such that this code may not be as efficient as possible for the
    chair dataset.

    The dataset is built on wrapping around the Torch Dataset object as well as HDF5
    for the underlying dataformat. This is a very fast data format and supports both
    loading into RAM or directly off disk.

    Make sure your HDF5 installation is updated to support SWMR mode for parallel
    access, as most default OS packages are older than this support.
    """
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

        resolution = ['IMG_8', 'IMG_16', 'IMG_32', 'IMG_64', 'IMG_128', 'IMG_256', 'IMG_512']

        self._base_key = 'IMG_'
        self._base_masks_key = 'IMG_'
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

        self.chair_full_inds = np.loadtxt(self.c.dining_room_chair_full_inds_dir, dtype=int)
        print('{} chairs in the dataset'.format(self.chair_full_inds.shape[0]))
        self.chair_labeled_inds = np.loadtxt(self.c.dining_room_chair_labeled_inds_dir, dtype=int)


        self._len = {k: len(self.dataset[k]) for k in resolution}

        # self.num_data = self._len['data8x8']
        self.num_data = self.chair_full_inds.shape[0]

        assert all([resol in self.dataset.keys() for resol in resolution])

        # Training image inds
        # image_ids outside train, valid, test - take % of that
        # then concat back with train_image_inds as training_image_inds

        # Note: For chairs there is a different mapping scheme than vehicles dataset, so this code is unnecessarily complex

        # self.design_ids_of_images = design_ids
        self.design_ids_of_images = self.chair_full_inds
        train_design_ids, valid_design_ids, test_design_ids = np.unique(self.train_x), np.unique(self.valid_x), np.unique(self.test_x)
        # train_bool_array = np.isin(self.design_ids_of_images, train_design_ids)
        # self.training_image_inds = np.nonzero(train_bool_array)[0]
        # num_training_inds = len(self.training_image_inds)
        # labeled_design_ids = np.concatenate((train_design_ids, valid_design_ids, test_design_ids))

        unlabeled_bool_array = ~np.isin(self.chair_full_inds, self.chair_labeled_inds)
        self.unlabeled_image_inds = np.nonzero(unlabeled_bool_array)[0]

        if self.c.percentage_of_unlabeled_data != 1.0: # 0.5 # 1.0, 0.5, 0.25; 1.0 is full training set
            num_unlabeled = int(self.c.percentage_of_unlabeled_data * len(self.unlabeled_image_inds))
            self.unlabeled_image_inds = np.random.choice(self.unlabeled_image_inds, size=num_unlabeled, replace=False)
        self.training_image_inds = np.concatenate((train_design_ids, self.unlabeled_image_inds))
        self.training_mean = np.mean(np.abs(self.valid_y - self.train_y.mean()))
        self.training_median = np.mean(np.abs(self.valid_y - np.median(self.train_y)))
        self.training_mid = np.mean(np.abs(self.valid_y - 3.0))

        print("\nTotal Unlabeled Data at {}%: {}".format(int(self.c.percentage_of_unlabeled_data*100),
                                                         ( len(self.unlabeled_image_inds)) * self.c.number_viewpoints_per_product))

        # Sideview image ids
        self.random_image_idxs = np.arange(self.c.number_viewpoints_per_product)

        # self.valid_x_raveled = np.array([[ind, view] for ind in self.valid_x for view in np.arange(self.c.number_viewpoints_per_product)], dtype=np.uint8)
        self.valid_x_repeated = self.valid_x.repeat(self.c.number_viewpoints_per_product) # np.array([[ind, view] for ind in self.valid_x for view in np.arange(self.c.number_viewpoints_per_product)], dtype=np.uint8)
        self.valid_y_repeated = self.valid_y.repeat(self.c.number_viewpoints_per_product)
        self.test_x_repeated = self.test_x.repeat(self.c.number_viewpoints_per_product)
        self.test_y_repeated = self.test_y.repeat(self.c.number_viewpoints_per_product)

        # self.random_image_idxs = np.array([0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 33, 34, 35])
        self.random_image_idxs  = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    def get_random_image_ids_given_design_ids(self, design_ids):
        random_views = np.random.choice(self.random_image_idxs+self.c.rated_viewpoint, size=len(design_ids), replace=True)
        # image_ids = np.array([
        #     np.where(self.design_ids_of_images == elm)[0][random_views] for elm in design_ids])

        image_ids = np.array([design_ids, random_views]).swapaxes(0,1)

        return image_ids
        # return image_ids.flatten()

    # def get_side_image_id_given_design_ids(self, design_ids):
    #     image_ids = np.array([
    #         np.where(self.design_ids_of_images == elm)[0][0] for elm in design_ids])
    #     return image_ids.flatten()

    def get_side_image_id_given_design_ids(self, design_ids):
        image_ids = np.array([design_ids, np.ones(design_ids.shape[0], dtype=int)*self.c.rated_viewpoint])
        image_ids = image_ids.swapaxes(0, 1)

        return image_ids

    def __call__(self,
                 batch_size,
                 phase,
                 alpha,
                 with_masks=True,
                 # with_masks=False,
                 with_ratings=False,
                 data_split='train',
                 batch_ind=None,
                 side_view=False):

        size = 4 * (2 ** phase)
        key = self._base_key + '{}'.format(size)

        if not with_ratings:
            # Can't do fast lookup anymore, need to do slow functions using code from ratings below
            # image_ids = np.random.choice(self.training_image_inds, size=batch_size, replace=True)

            if batch_ind is None:
                # inds = np.random.randint(self.train_x.shape[0], size=batch_size)
                # This is for training rated data + unlabelled data
                design_ids = np.random.choice(self.training_image_inds, size=batch_size, replace=True)
            else:
                # Null case here
                # inds = np.arange(self.train_x.shape[0])[batch_ind * batch_size: (batch_ind + 1) * batch_size]
                # design_ids = self.train_x[inds]
                raise Exception("got to null case for chair design")

            if not side_view:
                image_ids = self.get_random_image_ids_given_design_ids(design_ids)
            else:
                image_ids = self.get_side_image_id_given_design_ids(design_ids)

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

            # print(image_ids)
        hi_res_batch_images = np.array([self.dataset[key][i[0]][i[1]] / 127.5 - 1.0 for i in image_ids], dtype=np.float32)
        hi_res_batch_images = hi_res_batch_images.transpose(0, 3, 1, 2) # because this is in RGB vs BGR
        batch_attributes = np.array([self.dataset["info"][i[0]][i[1]] for i in image_ids], dtype=np.float32)

        if with_masks:
            key = self._base_masks_key + '{}'.format(size)
            # batch_masks = np.array([self.masks[key][i][:,:,0] / 255.0 for i in image_ids], dtype=np.float32)
            # batch_masks = np.array([self.masks[key][i[0]][i[1]][:, :, 0] / 255.0 for i in image_ids], dtype=np.float32)
            batch_masks = np.array([self.masks[key][i[0]][i[1]][:, :, 0] for i in image_ids], dtype=np.float32)
            batch_masks = np.expand_dims(batch_masks, 3).transpose(0, 3, 1, 2)
            batch_masks = torch.from_numpy(batch_masks)

        if alpha < 1.0 and phase > 0:
            lr_key = self._base_key + '{}'.format(size // 2)
            low_res_batch_images = np.array([self.dataset[lr_key][i[0]][i[1]] / 127.5 - 1.0 for i in image_ids],
                                            dtype=np.float32).transpose(0, 3, 1, 2).repeat(2, axis=2).repeat(2, axis=3)
#             low_res_batch_images = np.array([self.dataset[lr_key][i] / 127.5 - 1.0 for i in image_ids],
# dtype=np.float32).repeat(2, axis=2).repeat(2, axis=3)
#             low_res_batch_images = low_res_batch_images  # because this is in RGB vs BGR
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
        key = self._base_key + '{}'.format(size)
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


