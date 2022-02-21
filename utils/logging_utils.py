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
import glob
import shutil
import sys
import os
from collections import OrderedDict
import numpy as np
import torch
from torch import nn


def summarize_architecture(model, input_size, phase, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    # x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    x = [torch.rand(1, *in_size).type(dtype) for in_size in input_size]
    model(*x, phase=phase)


    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------------------")
    # line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    line_new = "{:>25}  {:>25} {:>25} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
    print(line_new)
    print("============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>25}  {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return summary

def save_config(c):
    # Export config.
    try:
        with open(os.path.join(c.result_subdir, 'config.txt'), 'wt') as fout:
            for k, v in sorted(c.items()):
                if not k.startswith('_'):
                    fout.write("%s = %s\n" % (k, str(v)))
    except:
        pass

def save_checkpoint(state,
                    result_subdir,
                    exp_desc,
                    epoch,
                    is_best,
                    filename_base='ckpt.pth.tar'):
    filename = '{}_{}_{}'.format(exp_desc, epoch, filename_base)
    save_path = result_subdir + "/"
    torch.save(state, save_path + filename)
    if is_best:
        shutil.copyfile(save_path + filename, save_path + 'best_model.pth.tar')

# Logging of stdout and stderr to a file.
class OutputLogger(object):

    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename, mode='wt'):
        assert self.file is None
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()


class TeeOutputStream(object):

    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush

    def write(self, data):
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()


output_logger = None


def init_output_logging():
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream(
            [sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream(
            [sys.stderr, output_logger], autoflush=True)


def set_output_log_file(filename, mode='wt'):
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)


# Result Logging - From PGGAN Repo
def create_result_subdir(result_dir, desc):
    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print("Saving results to", result_subdir)
    set_output_log_file(os.path.join(result_subdir, 'experiment_log.txt'))

    # Export config.
    try:
        with open(os.path.join(result_subdir, 'config.txt'), 'wt') as fout:
            for k, v in sorted(c.items()):
                if not k.startswith('_') or k.startswith('..') or k.startswith(
                        '/'):
                    fout.write("%s = %s\n" % (k, str(v)))
    except:
        pass

    return result_subdir


def make_dirs(c):
    if not os.path.exists(c.save_dir):
        os.makedirs(c.save_dir)
    c.checkpoint_dir = c.save_dir + '/checkpoint'
    if not os.path.exists(c.checkpoint_dir):
        os.makedirs(c.checkpoint_dir)
    if c.use_TB:
        if not os.path.exists(c.summary_dir):
            os.makedirs(c.summary_dir)