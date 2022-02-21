# -*- coding: utf-8 -*-
'''
        Project: Product Aesthetic Design: A Machine Learning Augmentation
        Authors: Alex Burnap, Yale University
        Email: alex.burnap@yale.edu

        Description:
        Main configuration file for running and replicating all experiments in paper, as well as usage
        in your own projects.  You should generally only need to modify this file.

        License: MIT License

        OSS Code Attribution (see Licensing Inheritance):
        Portions of Code From or Modified from Open Source Projects:
        https://github.com/tkarras/progressive_growing_of_gans
        https://github.com/AaltoVision/pioneer
        https://github.com/DmitryUlyanov/AGE
        https://github.com/akanimax/attn_gan_pytorch/
'''
import os

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super(EasyDict, self).__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

c = EasyDict()
c.env = EasyDict()

#-----------------------------------------------------------------------------
#           Main Experiment Settings
#-----------------------------------------------------------------------------

# Three main experimental conditions to replicate all results in paper
# c.experiment_type = 'combined_model_train' # Train the proposed model from paper.
# c.experiment_type = 'pretrain_only'        # Train/Fine-tune the benchmark VGG pretrained deep model
# c.experiment_type = 'baseline_conventional_ML_and_CV' # Train the conventional ML + computer vision features

c.experiment_type = 'combined_model_train' # Train the proposed model from paper.
c.load_checkpoint = True # Do we start training from scratch or from given checkpoint?
c.random_seed = 0 # Used for seeding random number generators. Needed for reproducibility. Paper uses seeds 0,1,2.
c.dataset = 'chairs'  # 'vehicles' # Change this to your IMAGE dataset name
c.attribute = 'modern'  # Aesthetic rating attribute name (if ratings dataset has multiple)  # 'appeal', luxurious, sporty, innovative

# Else this will default to path directories at bottom of this config file (e.g., for the chair dataset, only one attribute)

#----------- GPU Params --------------------------
c.env.computer = os.environ['USER'] # 'server' # 'dev' #

if not c.env.computer == 'YOUR_WORKSTATION_NAME':
    if c.experiment_type == 'combined_model_train':
        c.env.CUDA_VISIBLE_DEVICES = '1' # Number of GPUs to use. CUDA device identifiers.
        # c.env.CUDA_VISIBLE_DEVICES = '0,1' # Number of GPUs to use. CUDA device identifiers.
    else:
        c.env.CUDA_VISIBLE_DEVICES = '1' # Number of GPUs to use. CUDA device identifiers.
else:
    c.env.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']

c.env.CUDA_DEVICE_ORDER = "PCI_BUS_ID"
c.epsilon = 1e-9

#-----------------------------------------------------------------------------
#           Experimental Logging Settings
#-----------------------------------------------------------------------------
c.save_output_log = True
c.save_checkpoints = True
c.use_cometML = True # CometML is a SaaS provider of ML Training and model tracking.
c.comet_api_key = ""
c.comet_project_name = ""
c.comet_workspace = ""

c.use_checkpoint_sample_i = True
c.comet_project_id = None

if not c.use_checkpoint_sample_i:
    c.encoder_phase = 6
    c.encoder_alpha = 1
    c.force_sample_i = 0
    c.start_phase = 6
    c.comet_project_id = None

c.checkpoint_cycle = 100
c.save_checkpoint_cycle = 1000
c.print_frequency = 50
c.sample_N = 16
c.interpolate_N = 0
c.reconstructions_N = 16
c.load_state_dict_strict = False
c.use_TB = False
c.summarize_architecture = False

#-----------------------------------------------------------------------------
#           Training Settings - Predictive, Generative, Encoder
#-----------------------------------------------------------------------------
c.e_lr = 0.001 # Encoder Learning Rate
c.g_lr = 0.001 # Generator Learning Rate
c.p_lr = 0.0002 # Predictive Model Learning Rate

c.n_generator = 10 # Number of generator updates versus encoder
c.n_critic = 1
c.n_accumulate_gradients = 1 # If low VRAM, accumulate gradients over multiple updates.

c.p_optimizer = 'adam'  # 'sgd'
c.opt_epsilon = 1e-8 # ADAM optimizer epsilon for floating point drama
c.optimizer = 'adam'  # 'sgd'
c.opt_betas = (0.0, 0.99)
c.opt_betas_predictor = (0.9, 0.99)
c.reset_optimizers = True
c.reconstruct_transformed_labels = False

# Progressive Training Scheduling
c.start_phase = 0
c.max_phase = 7
c.images_per_stage = 1000000 # Takes ~ 1 week cumulative to train_combined_model to 256x256, 2+ weeks to 512x512
c.total_kimg = int(c.images_per_stage * (c.max_phase + 1) / 1000)  # All stages trained once

# Progressive Training Extra Settings (if manual override load of prev. checkpoint)
c.progressive_phase_offset = 0 # Use the reloaded model but start fresh from next phase (when manually moving to the next)
c.progressive_step_offset = 0 # Use the reloaded model but ignore the given number of steps (use -1 for all steps)'

c.train_side_view_only = False
c.evaluate_side_view_only = True
c.evaluate_full_validation_set = True
c.evaluate_full_test_set = True

#-----------------------------------------------------------------------------
#           Data Settings
#-----------------------------------------------------------------------------

c.image_size = 512 # Max image resolution (e.g., 512x512)
c.bw_images = True

c.percentage_of_training_data = 1.0# 1.0 # 1.0, 0.5, 0.25; 1.0 is full labeled dataset
c.percentage_of_unlabeled_data = 1.0 # 1.0, 0.5, 0.25; 1.0 is full unlabeled dataset

c.augment_images = False
c.train_valid_test_ratio = 0.50  # Train/Validation/Test Data Split
c.min_self_consistency = 0.75 # Survey Panelist Rating Self-Consistency Threshold (Krippendorf Alpha)
c.suv_cuv_only = False # Only for vehicle dataset (main proprietary dataset in paper)
c.train_test_split_unique_models = True # Stratified Split On Vehicle Models?
c.train_test_split_unique_models_and_average_ratings = False # Turn off for chair dataset since not stratified
c.create_duplicate_ratings_for_viewpoints = True

if c.dataset == "vehicles":
    c.number_viewpoints_per_product = 36
elif c.dataset == "chairs":
    c.number_viewpoints_per_product = 62
    c.rated_viewpoint = 25

#   Speed Up
c.num_data_workers = 6  # CPU Parallelization - Number of data loading runners per instance
c.use_ram_for_image_load = True # Load images directly into RAM?  Usually faster than SSD memmap.

#-----------------------------------------------------------------------------
#           Loss Function Settings
#-----------------------------------------------------------------------------

c.training_criteria = 'l1'  # 'l2' - do you want MAE or MSqE -- paper uses MAE
c.evaluation_criteria = 'l1'  # 'l2' - do you want MAE or MSqE -- these should be same
c.match_x_metric = 'L1' # TODO: check why defined twice
c.match_y_metric = 'L1'

c.use_loss_x_reco = True
c.match_x_scale = 4.0 #1.0 #2.0
c.match_x_fade_in_scale = 4.0 #1.5 #3.0

c.conditional_model = True # Do we have design attributes
c.use_label_loss = True
c.label_loss_scale = 0.01 # 0.25
c.label_loss_start_phase = 1

c.use_semisupervised_predictive_loss = True # Do we have aesthetic ratings for all (full supervision) or just some (semi-supervised)
c.labeled_data_cycle = 50 # How often to updated semisupervised training?
c.pred_loss_scale = 0.5 # 1.0, 0.25
c.pred_loss_start_phase = 4
c.y_KL_scale = 0.05

c.sequentially_freeze_weights_during_training = True
c.use_gen_recon_min_chance = 0.5

c.train_using_y_mu = True
c.base_rating = 3.0
c.evaluate_using_y_mu = True # we may train with full KL term, but evaluate using variational mean parameter
c.evaluate_full_validation_set_combined_model = True

c.use_masks = True
c.mask_loss_scale = 0.025
c.inject_masks_into_encoder = False

c.use_real_x_KL = True
c.use_loss_fake_D_KL = True
c.use_loss_KL_egh = True

c.vae_loss = 'both' # 'stdvae', 'densityratio' do we use VAE reparam trick or approximate KL via density ratio, or some ratio ("both")
c.dr_to_stdvae_ratio = 0.5

c.real_x_KL_dr_scale = 0.5
c.fake_D_KL_dr_scale = 0.5
c.fake_G_KL_dr_scale = 0.5

c.real_x_KL_stdvae_scale = 0.000002
c.fake_D_KL_stdvae_scale = 0.000002
c.fake_G_KL_stdvae_scale = 0.000002

c.use_gumbel_softmax = True # p(a|X) ~ multinomial. p(h|a, X) ~ Gaussian.  Need to calculate the KLD of this term when we don't have attribute information.
c.gumbel_softmax_temperature = 0.1

if c.experiment_type == 'pretrain_only' or c.experiment_type == 'encoder_only':
    c.use_masks = False
    c.use_label_loss = False

#-----------------------------------------------------------------------------
#           Model Architecture Settings (Predictive, Generative, Encoder)
#-----------------------------------------------------------------------------
c.n_latents = 512  # Dimensionality of the latent embedding h
if c.dataset == "vehicles":
    c.n_label = 111 # Number of design attributes
    c.nc = 3  # Number of color channels (e.g., RGB)
elif c.dataset == "chairs":
    c.n_label = 3 # Number of design attributes
    c.nc = 1  # Number of color channels (e.g., RGB)

c.nz = c.n_latents - c.n_label
c.gen_spectral_norm = True
c.gen_use_residual = True
c.use_generator_output_tanh = True # Do we have a tanh final output? (depends on your data's scale)
c.enc_spectral_norm = True
c.enc_use_residual = True
c.manual_spec_norm = False # Overwrite pytorch built-in spectral norm since had a bug (FIXED in 2020)

c.enc_use_squeeze_and_excite = True
c.gen_use_squeeze_and_excite = True

c.vae_params_cond_a = True # p(h) is latent mixture model marginal density
c.num_combined_model_fc_units = 128 # Number of neurons in fully connected predictive model layers
c.reparametrize = False
c.norm_vae_params = False
c.predictive_model_nonlinear = True
c.cond_h_on_y = True # Says where the predictor model takes information

# Optional Architecture Settings (not in Journal Paper, maybe useful for you though)
c.use_self_attention = False # Do we include self-attention? Note: Powerful but VRAM memory heavy.
c.self_attention_layer = 5 # If self-attention, which layer?

#-----------------------------------------------------------------------------
#           BASELINE: Pretrained Deep Learning / Fine-Tuning Settings
#-----------------------------------------------------------------------------
c.arch = 'vgg16' # Pretrained Model used for baseline
#   NOTE: Many other Pretrained Models, uncomment below code to see options.
#   import torchvision.models as pretrained_models
#   c._pretrained_model_names = sorted(name for name in pretrained_models.__dict__ if name.islower() and not name.startswith("__"))

c.epochs_pretrained = 60  # Number of training epochs
c.num_batches_per_epoch_pretrained = 100
c.num_random_batches_per_epoch_pretrained_evaluate = 50

c.num_pretrained_fc_units = 128 # Number of neurons in fully connected predictive model layers
c.lr_pretrained = 0.0002  # Pretrained Model Learning Rate
c.pretrained_model_use_attributes = True
c.pretrained_tower_batchnorm = True

c.adam_optim_betas_pretrain = (0.99, 0.999)
c.adjust_learning_rate_during_training = False
c.number_of_epochs_before_dropping_lr = 10 # Only meaningful if above is True

c.num_batches_print_freq_train = 10  # How often do we output prediction accuracy figures.
c.num_batches_print_freq_valid = 5  # How often do we output prediction accuracy figures.
c.freeze_pretrained_weights = True
c.freeze_encoder_weights = True

c.finetune_model = True
c.epoch_begin_finetune = 10
c.finetune_optimizer = 'sgd'  # 'sgd'
c.lr_finetune = 0.00001
c.momentum = 0.9  # momentum if using SGD
c.weight_decay = 0.0002  # weight decay if using SGD
c.num_epoch_save_model_pretrained = 5  # How often do we save the model.

#-----------------------------------------------------------------------------
#           BASELINE: Conventional ML + Computer Vision Model Settings
#-----------------------------------------------------------------------------
c.conventional_baseline_model = 'random_forest' #'svr' @ baseline convention machine learning model
c.num_baseline_cpu_jobs = 12  # number of CPUs for baseline models run
c.num_random_forest_trees = 100
c.use_saved_baseline_features = False  # Use precalculated features for baseline model
c.save_computed_baseline_features = False  # Save precalculated features for baseline model

#-----------------------------------------------------------------------------
#           Directory Paths and Logging Directory Creation
#-----------------------------------------------------------------------------

try:
    c.gpu_count = len(c.env.CUDA_VISIBLE_DEVICES.split(
        ","))  # Set to 1 manually if don't want multi-GPU support; torch.cuda.device_count()
except:
    c.gpu_count = 0  # Set manually multi-gpu


if c.dataset == 'vehicles':
    c.ratings_dataset_path = ""
    c.suvs_list_path = ""
    c.result_dir = "" + os.getcwd().split('/')[-1] # Results
    c.images_dir = ''
    c.labels_dir = ''
    c.masks_dir = ''
    c.vehicle_names_path = ''

elif c.dataset == 'chairs':
    c.ratings_dataset_path = "<RATINGS_DATASET_PATH>"
    c.result_dir = "<RESULTS_AND_LOGGING_PATH>" + os.getcwd().split('/')[-1]  # Results
    c.images_dir = '/path/to/your/chair_data_grayscale.h5'
    c.labels_dir = None
    c.masks_dir = '/path/to/your/chair_data_masks.h5'
    c.dining_room_chair_full_inds_dir = "/path/to/your/dining_room_chair_full_inds.csv"
    c.dining_room_chair_labeled_inds_dir = "/path/to/your/dining_room_chair_labeled_inds.csv"

else:
    raise ValueError("Need to define paths to the dataset, see bottom of config.py file")

if c.load_checkpoint:
    c.load_checkpoint_dir = "/path/to/your/model_checkpoint/for/resuming/from/checkpoint/7995240_state"

def batch_size_dict(reso):
    if c.gpu_count == 1:
        # Change these so it fits your workstations / clusters capabilities
        batch_table = {4: 256, 8: 256, 16: 128, 32: 128, 64: 64, 128: 24, 256: 12, 512: 8, 1024: 1} # this works for 48GB VRAM without any monitor / X-window
    elif c.gpu_count == 2:
        batch_table = {4: 256, 8: 256, 16: 256, 32: 256, 64: 128, 128: 64, 256: 48, 512: 24, 1024: 4}
    elif c.gpu_count == 3:
        batch_table = {4: 256, 8: 256, 16: 256, 32: 128, 64: 96, 128: 48, 256: 24, 512: 12, 1024: 4}
    elif c.gpu_count == 4:
        batch_table = {4:512, 8:256, 16:128, 32:256, 64:256, 128:96, 256:64, 512:32, 1024:4}
    elif c.gpu_count == 8: # Big money
        batch_table = {4:512, 8:512, 16:512, 32:256, 64:256, 128:128, 256:64, 512:32, 1024:8}
    else:
        assert(False)

    return batch_table[reso]


