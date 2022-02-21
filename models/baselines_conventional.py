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
import numpy as np
import cv2
import skimage.feature

def extract_features_single_image(image,
                                  spatial_size=(32, 32),
                                  hist_bins=100,
                                  orient=9,
                                  pix_per_cell=8,
                                  cell_per_block=3,
                                  hog_channel=0,  # hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL" / grayscale use 0
                                  spatial_feat=True,
                                  hist_feat=True):
    '''
    Extract traditional computer vision features from a single image.

    Default params for HOG extraction for open access codebase.
    orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys'

    If larger resolution (e.g., 512 x 512), consider pixels_per_cell and cells_per_block
    or too many features and overwhelming computational time with conventional ML.
    '''
    img_features = []
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(
                get_hog_features(
                    image[:, :, channel],
                    orient,
                    pix_per_cell,
                    cell_per_block,
                    vis=False,
                    feature_vec=True))
    else:
        hog_features = get_hog_features(
            image[:, :, hog_channel],
            orient,
            pix_per_cell,
            cell_per_block,
            vis=False,
            feature_vec=True)

    assert any(np.isnan(hog_features)) == False
    img_features.append(hog_features)
    img_features = np.array(img_features).flatten()
    if spatial_feat == True:
        spatial_features = bin_spatial(image, size=spatial_size)
        img_features = np.concatenate([img_features, spatial_features])
    if hist_feat == True:
        hist_features = color_hist(image, n_bins=hist_bins)
        img_features = np.concatenate([img_features, hist_features])
    return img_features


def extract_features_array(images, verbose=True):
    '''
    Extract features for an array of images.
    '''
    num_images = images.shape[0]
    num_features = extract_features_single_image(images[0]).shape[0]

    features = np.zeros([num_images, num_features])
    for ind, image in enumerate(images):
        if ind % 500 == 0:
            if verbose:
                print("Extracting conventional CV features {}/{}".format(ind + 1, num_images))
        features[ind] = extract_features_single_image(image)
    return features


def get_hog_features(img,
                     orient = 9, # HOG orientations
                     pix_per_cell = 8, # HOG pixels per cell
                     cell_per_block = 3,# HOG cells per block
                     transform_sqrt=False,
                     vis=False,
                     feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = skimage.feature.hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=transform_sqrt,
            visualize=vis,
            feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = skimage.feature.hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=transform_sqrt,
            visualize=vis,
            feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features

def color_hist(img,
               n_bins=100,
               n_channels = 1,
               bins_range=(-1, 1) # (0, 256) if uint8 or (-1, 1) if scaled and float32
              ):
    if n_channels == 3: # Compute the histogram of the color channels separately
        channel0_hist = np.histogram(img[:, :, 0], bins=n_bins, range=bins_range)
        channel1_hist = np.histogram(img[:, :, 1], bins=n_bins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 2], bins=n_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel0_hist[0], channel1_hist[0],
                                        channel2_hist[0]))
    elif n_channels == 1: # Grayscale
        hist_features = np.histogram(img[:, :, 0], bins=n_bins, range=bins_range)[0]
    # Return the individual histograms, bin_centers and feature vector
    else:
        raise ValueError
    return hist_features



