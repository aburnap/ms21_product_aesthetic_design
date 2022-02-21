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
from contextlib import ExitStack
import os
import sys
from comet_ml import Experiment as CometExperiment
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from numpy.random import RandomState
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

from models import baselines_conventional
from training.train_pretrained_model import train_pretrained_model, evaluate_pretrained_model
from utils import logging_utils

from data.vehicles import Vehicles
from data.chairs import Chairs

from training.train import train_combined_model
from training.session import CombinedTrainSession, PretrainedSession


#----------------------------------------------------------------------------------------------------------------------
#           Globals
#----------------------------------------------------------------------------------------------------------------------


torch.backends.cudnn.benchmark = True

# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------------

class Experiment(object):
    '''
    Experiment object that holds experiment data and runs several classes of
    baseline, pretrained deep learning, and custom model deep learning.
    Allows introspection and saving state to disk for reproducibility.
    '''

    def __init__(
            self,
            c=None,  # Configuration params
    ):
        assert c is not None
        self.c = c  # Main configuration file.  Required for experiment.

        # Globals
        np.random.seed(self.c.random_seed)
        os.environ.update(self.c.env)

    # ------- Main run code --------------------------------------------------
    def run_experiment(self, skip_initialization=False):
        '''
        Main experiment run code.
        '''
        try:
            print("\nBeginning experiment on: " + torch.cuda.get_device_name(0))
            print("Using ", torch.cuda.device_count(), " GPUs\n")
            print('PyTorch {}'.format(torch.__version__))
        except RuntimeError:
            raise RuntimeError('Out of memory on GPU.')

        if not skip_initialization:
            print("Initializing Experiment...")
            print("Loading Data...")
            self.init_experiment_data()
            self.print_vehicle_data_details()
            print("Initializing Models...")
            self.init_experiment()
            if self.c.save_output_log:
                if self.c.experiment_type == 'baseline_conventional_ML_and_CV':
                    self.c.use_cometML = False
                self.init_logging()

        if self.c.experiment_type == 'combined_model_train':
            train_combined_model(self.session.generator,
                                 self.session.encoder,
                                 self.session.predictor,
                                 self.dataset,
                                 session=self.session,
                                 total_steps=self.c.total_kimg * 1000)

        elif self.c.experiment_type == 'pretrain_only':
            with self.session.comet_experiment.train() if self.session.c.use_cometML else ExitStack():
                for epoch in range(0, self.c.epochs_pretrained):

                    if self.c.adjust_learning_rate_during_training:
                        self.adjust_learning_rate(self.session.optimizer, epoch)

                    train_pretrained_model(self.dataset,
                                           self.session,
                                           epoch)

                    evaluate_pretrained_model(self.dataset,
                                              self.session,
                                              epoch,
                                              data_split='valid')

                    evaluate_pretrained_model(self.dataset,
                                              self.session,
                                              epoch,
                                              data_split='test')


        elif self.c.experiment_type == 'baseline_conventional_ML_and_CV':

            print("Beginning training of baseline model")
            self.model.fit(self.dataset.train_x_features, self.dataset.train_y)
            print("Finished training of baseline model")
            test_y_hat = self.model.predict(self.dataset.test_x_features)
            mae = np.mean(np.abs(test_y_hat-self.dataset.test_y))
            print("Test Accuracy: {}".format(mae))

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return True

    #-----------------------------------------------------------------------------------
    #                           Experiment Initialization
    #-----------------------------------------------------------------------------------
    def init_experiment(self):
        '''
        Experimental set up of all experiment attributes depending on experiment type.
        '''

        # Begin Model Estimation and Evaluation
        if self.c.experiment_type == 'combined_model_train':
            self.session = CombinedTrainSession(c=self.c)
            self.session.setup()

        elif self.c.experiment_type == 'pretrain_only':

            # Pretrained Deep Learning Model Session
            self.session = PretrainedSession(c=self.c)
            self.session.setup()

        elif self.c.experiment_type == 'baseline_conventional_ML_and_CV':
            # TODO 10/27/20: remove other baseline classifiers.  Just stick with random_forest since it did best.
            if self.c.conventional_baseline_model == 'svr':
                self.model = LinearSVR(
                    dual=True,
                    loss='l1',
                    fit_intercept=True,
                    C=1e-2,
                    verbose=True,
                    random_state=self.c.random_seed,
                    max_iter=1e4)
            elif self.c.conventional_baseline_model == 'random_forest':
                self.model = RandomForestRegressor(
                    criterion='mae',
                    n_estimators=self.c.num_random_forest_trees,
                    n_jobs=self.c.num_baseline_cpu_jobs,
                    random_state=self.c.random_seed,
                    verbose=3)
            else:
                raise ValueError("Baseline model not found")
        else:
            raise ValueError('Experiment type not found.')

    #-----------------------------------------------------------------------------------
    #                           Data Initialization
    #-----------------------------------------------------------------------------------
    def init_experiment_data(self):
        '''
        Sets up all data depending on experiment type.
        '''

        # Obtain Raw Rating Data
        ratings_df_full = pd.read_csv(self.c.ratings_dataset_path, header=0)
        ratings_df_full = ratings_df_full[["real_value", "design_id", "name"]]
        ratings_df_full['design_id'] = ratings_df_full['design_id'].astype(int)
        ratings_df_full.index = ratings_df_full.design_id

        # Vehicle Detail Lists
        if self.c.dataset == 'vehicles':
            self.products_list = np.loadtxt(self.c.suvs_list_path, delimiter=',', dtype=str)

            # Get Design ID to Image ID labels
            design_labels = np.load(self.c.labels_dir)
            design_ids_of_images = design_labels['arr_5']
            self.design_ids_of_images = design_ids_of_images.flatten()

        elif self.c.dataset == 'chairs':
            ratings_df_full['name'] = ratings_df_full['name'].astype(int)

        # Train/Valid/Test Split
        if self.c.train_test_split_unique_models and self.c.dataset=="vehicles":
            ratings_df_full["model"] = ratings_df_full["name"].apply(lambda x: " ".join(x.split(" ")[2:]))
            unique_models = ratings_df_full["model"].unique()

            # Stratified Splitting
            if self.c.train_test_split_unique_models_and_average_ratings:
                for ind, unique_model in enumerate(unique_models):
                    # Average rating for unique models
                    matched_designs = ratings_df_full[ratings_df_full["model"] == unique_model]
                    ratings_df_full.loc[matched_designs['design_id'], ['real_value']] = matched_designs.real_value.mean()

            train_inds, valid_and_test_inds = train_test_split(np.arange(len(unique_models)),
                                                               test_size=self.c.train_valid_test_ratio,
                                                               random_state=self.c.random_seed)
        else:
            ratings_df_full["model"] = ratings_df_full["name"]
            unique_models = ratings_df_full["model"].unique()
            train_inds, valid_and_test_inds = train_test_split(np.arange(ratings_df_full.shape[0]),
                                                               test_size=self.c.train_valid_test_ratio,
                                                               random_state=self.c.random_seed)
        half_length = int(valid_and_test_inds.shape[0] / 2)
        valid_inds, test_inds = valid_and_test_inds[:half_length], valid_and_test_inds[half_length:]
        self.ratings_df_train = ratings_df_full[ratings_df_full['model'].isin(unique_models[train_inds])]
        self.ratings_df_valid = ratings_df_full[ratings_df_full['model'].isin(unique_models[valid_inds])]
        self.ratings_df_test = ratings_df_full[ratings_df_full['model'].isin(unique_models[test_inds])]

        self.train_x, self.train_y = self.ratings_df_train['design_id'].values, self.ratings_df_train['real_value'].values
        self.valid_x, self.valid_y = self.ratings_df_valid['design_id'].values, self.ratings_df_valid['real_value'].values
        self.test_x, self.test_y = self.ratings_df_test['design_id'].values, self.ratings_df_test['real_value'].values

        if self.c.percentage_of_training_data != 1.0:
            print("Artificially reducing training data to {}% of full training set.".format(int(self.c.percentage_of_training_data*100)))
            new_num_products = int(self.c.percentage_of_training_data*self.train_x.shape[0])
            training_data_mask = np.random.choice(np.arange(self.train_x.shape[0]), new_num_products, replace=False)
            self.train_x = self.train_x[training_data_mask]
            self.train_y = self.train_y[training_data_mask]

        if self.c.create_duplicate_ratings_for_viewpoints:
            self.train_x, self.train_y = np.repeat(
                self.train_x, repeats=self.c.number_viewpoints_per_product), np.repeat(
                self.train_y, repeats=self.c.number_viewpoints_per_product)

            self.valid_x, self.valid_y = np.repeat(
                self.valid_x, repeats=self.c.number_viewpoints_per_product), np.repeat(
                self.valid_y, repeats=self.c.number_viewpoints_per_product)

            self.test_x, self.test_y = np.repeat(
                self.test_x, repeats=self.c.number_viewpoints_per_product), np.repeat(
                self.test_y, repeats=self.c.number_viewpoints_per_product)


        print("Using {} Train, {} Validation, {} Test Data".format(
            self.train_x.shape[0], self.valid_x.shape[0], self.test_x.shape[0]))

        print("Shuffling Data...")
        self.shuffle_experiment_data(seed=self.c.random_seed)

        # Setup data attributes particular to the experiment
        if self.c.experiment_type == 'combined_model_train' or self.c.experiment_type == 'pretrain_only' or self.c.experiment_type == 'baseline_conventional_ML_and_CV':
            if self.c.dataset == 'vehicles':
                self.dataset = Vehicles(use_RAM=self.c.use_ram_for_image_load,
                                        train_x=self.train_x,
                                        train_y=self.train_y,
                                        valid_x=self.valid_x,
                                        valid_y=self.valid_y,
                                        test_x=self.test_x,
                                        test_y=self.test_y,
                                        c=self.c)
            elif self.c.dataset == 'chairs':
                self.dataset = Chairs(use_RAM=self.c.use_ram_for_image_load,
                                        train_x=self.train_x,
                                        train_y=self.train_y,
                                        valid_x=self.valid_x,
                                        valid_y=self.valid_y,
                                        test_x=self.test_x,
                                        test_y=self.test_y,
                                        c=self.c)


        if self.c.experiment_type == 'baseline_conventional_ML_and_CV':
            print("Loading Images")
            key = self.dataset._base_key + '{}'.format(self.c.image_size)

            train_image_ids = self.dataset.get_random_image_ids_given_design_ids(self.dataset.train_x)
            train_images = np.array([self.dataset.dataset[key][i[0]][i[1]] / 127.5 - 1 for i in train_image_ids], dtype=np.float32)
            test_image_ids = self.dataset.get_side_image_id_given_design_ids(self.dataset.test_x)
            test_images = np.array([self.dataset.dataset[key][i[0]][i[1]] / 127.5 -1 for i in test_image_ids], dtype=np.float32)

            print("Beginning Feature Extraction")
            train_x_features = baselines_conventional.extract_features_array(train_images)
            test_x_features = baselines_conventional.extract_features_array(test_images)

            self.dataset.train_x_features = train_x_features
            self.dataset.test_x_features = test_x_features

            assert self.dataset.train_x_features.shape[1] == self.dataset.test_x_features.shape[1]
            print("Using {} Train and {} Validation Data with feature size of {}".
                  format(self.dataset.train_x_features.shape[0], self.dataset.test_x_features.shape[0],
                         self.dataset.train_x_features.shape[1]))
        else:
            pass
            # raise ValueError('Experiment type not found.')

    #-----------------------------------------------------------------------------------
    #                           Helper Functions (logging, shuffling, etc.)
    #-----------------------------------------------------------------------------------
    def init_logging(self):
        if self.c.experiment_type == 'combined_model_train':
            if not hasattr(self, 'session'):
                raise ValueError('Could not find Training Session')
            checkpoint_str = 'ContCheckpoint' if self.c.load_checkpoint else 'NewRun'
            self.c.experiment_description = '{}_{}_{}_iter{}_seed{}_{}fracdata'.format(
                self.c.experiment_type,
                self.c.attribute,
                checkpoint_str,
                self.session.sample_i,
                self.c.random_seed,
                self.c.percentage_of_training_data)

        elif self.c.experiment_type == 'pretrain_only':
            self.c.experiment_description = '{}_{}_e{}_seed{}_{}fracdata'.format(
                self.c.experiment_type,
                self.c.attribute,
                self.c.epochs_pretrained,
                self.c.random_seed,
                self.c.percentage_of_training_data)

        elif self.c.experiment_type == 'baseline_conventional_ML_and_CV':
            self.c.experiment_description = '{}_{}_{}_seed{}'.format(
                self.c.experiment_type,
                self.c.attribute,
                self.c.conventional_baseline_model,
                self.c.random_seed)

        print("Creating result directory and logging for reproducibility")

        self.c.result_subdir = logging_utils.create_result_subdir(self.c.result_dir, self.c.experiment_description)
        self.c.summary_dir = self.c.result_subdir + "/summary"
        self.c.save_dir = self.c.result_subdir

        logging_utils.make_dirs(self.c)
        logging_utils.save_config(self.c)
        logging_utils.set_output_log_file(os.path.join(self.c.result_subdir, 'experiment_log.txt'))
        logging_utils.init_output_logging()

        f = open('{}/config.txt'.format(self.c.save_dir), 'w')
        for key, val in self.c.items():
            f.write("{}={}\n".format(key, val))
        f.close()

        # Setup Online Logging
        # TODO: 10/27/20 setup comet ml logging for existing experiment - low priority
        if self.c.use_cometML:
            self.session.comet_experiment = CometExperiment(api_key=self.c.comet_api_key,
                                                            project_name=self.c.comet_project_name,
                                                            workspace=self.c.comet_workspace,
                                                            log_graph=True)

            exp_name = '_'.join(self.c.save_dir.split('/')[-2:])
            self.session.comet_experiment.set_name(exp_name)
            self.session.comet_experiment.set_filename(exp_name)
            self.session.comet_experiment.log_parameters(self.c)

    def shuffle_data_helper(self, seed=0, *arrays):
        """ Shuffles an arbirary number of data arrays by row in consistent manner"""
        for array in arrays:
            prng = RandomState(seed)
            prng.shuffle(array)

    def shuffle_experiment_data(self, seed=0):
        self.shuffle_data_helper(seed, self.train_x, self.train_y,
                                       self.valid_x, self.valid_y,
                                       self.test_x, self.test_y)

    def print_vehicle_data_details(self,
                                   print_train_set=False,
                                   print_valid_set=True,
                                   print_test_set=True):

        self.print_sanity_check_values()
        if self.c.train_test_split_unique_models == True:
            if print_train_set:
                train_set_details = self.ratings_df_train['name'].values
                print("\nTrain Set: {}\n".format(train_set_details))
            if print_valid_set:
                valid_set_details = self.ratings_df_valid['name'].values
                print("\nValidation Set: {}\n".format(valid_set_details))
            if print_test_set:
                test_set_details = self.ratings_df_test['name'].values
                print("\nTest Set: {}\n".format(test_set_details))
        else:
            if print_train_set:
                train_set_details = np.array([
                    ' '.join(elm)
                    for elm in self.products_list[self.train_x_design_iloc][:, :3]])
                print("\nTrain Set: {}\n".format(train_set_details))
            if print_valid_set:
                valid_set_details = np.array([
                    ' '.join(elm)
                    for elm in self.products_list[self.test_x_design_iloc][:, :3]])
                print("\nValidation Set: {}\n".format(valid_set_details))
            if print_test_set:
                test_set_details = np.array([
                    ' '.join(elm)
                    for elm in self.products_list[self.valid_x_design_iloc][:, :3]])
                print("\nTest Set: {}\n".format(test_set_details))

    def print_sanity_check_values(self, print_guess_three=False):
        print("\nNaive Baselines:")
        print('Guess training mean: {:.4f}'.format(self.dataset.training_mean))
        print('Guess training median: {:.4f}'.format(self.dataset.training_median))
        if print_guess_three:
            print('Guess 3.0: {:.4f}\n'.format(self.dataset.training_mid))

    def adjust_learning_rate(self, epoch):
        """Stepwise decrease in the learning rate by 10 every 10 epochs"""
        new_lr = self.c.lr * (0.1 ** (epoch // 10))
        for param_group_ind, _ in enumerate(self.optimizer.state_dict()['param_groups']):
            self.optimizer.param_groups[param_group_ind]['lr'] = new_lr

if __name__ == "__main__":
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    #torch.backends.cudnn.benchmark = True
    from config import c
    np.random.seed(c.random_seed)
    print('Beginning Experiment')
    os.environ.update(c.env)
    exp = Experiment(c)
    exp.run_experiment()
    print('Finished Experiment')