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
import torch
import torch.nn
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import config
c = config.c

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 padding,
                 kernel_size2=None,
                 padding2=None,
                 use_residual=False,
                 self_attention=False,
                 squeeze_and_excite=False):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        layers = [SpectralNormConv2d(in_channel, out_channel, kernel1, padding=pad1),
                  nn.LeakyReLU(0.2),
                  SpectralNormConv2d(out_channel, out_channel, kernel2, padding=pad2),
                  nn.LeakyReLU(0.2)]

        self.conv = nn.Sequential(*layers)

        if self_attention:
            layers.append(SelfAttention(out_channel))

        if squeeze_and_excite:
            layers.append(SELayer(out_channel))

        self.use_residual = use_residual
        if self.use_residual:
            self.short_cut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
            self.last_lrelu = nn.LeakyReLU(0.2, inplace = True)


    def forward(self, inputs):
        if self.use_residual:
            residual = self.short_cut(inputs)
            out = self.conv(inputs)
            out += residual
            out = self.last_lrelu(out)
            return out
        else:
            out = self.conv(inputs)
            return out


class SpectralNormConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        nn.init.kaiming_normal_(conv.weight)
        conv.bias.data.zero_()
        if c.manual_spec_norm:
            self.conv = spectral_norm(conv)
        else:
            self.conv = nn.utils.spectral_norm(conv)

    def forward(self, inputs):
        return self.conv(inputs)


class SELayer(nn.Module):
    # Squeeze and Excitation
    # https://github.com/moskomule/senet.pytorch
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttributeLogitsGumbelSample(torch.nn.Module):

    def __init__(self, temperature):
        super(AttributeLogitsGumbelSample, self).__init__()
        self.temperature = temperature
        self.sm = torch.nn.Softmax(dim=1)

        # years = data_labels['arr_0']  # 15 years
        # makes = data_labels['arr_1']  # 48 makes
        # models = data_labels['arr_2']  # 23 models
        # bodytypes = data_labels['arr_3']  # 20 bodytypes
        # views = data_labels['arr_4']  # 2 variables for viewpoint
        # design_ids = data_labels['arr_5']  # 2 variables for viewpoint
        # colors = data_labels['arr_6']  # 3 variables for color
        self.label_inds = [15, 48, 23, 20, 2, 3]
        # 15, 63, 86, 106, 108, 111


    def sample_gumbel(self, shape, is_cuda=True, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / self.temperature, dim=1)

    def forward(self, label_logits):

        year_logits = label_logits[:, :15]
        make_logits = label_logits[:, 15:63]
        model_logits = label_logits[:, 63:86]
        bodytype_logits = label_logits[:, 86:106]
        view_logits = label_logits[:, 106:108]
        color_logits = label_logits[:, 108:]

        year_probs = self.gumbel_softmax_sample(year_logits)
        make_probs = self.gumbel_softmax_sample(make_logits)
        model_probs = self.gumbel_softmax_sample(model_logits)
        bodytype_probs = self.gumbel_softmax_sample(bodytype_logits)

        # labels = torch.cat([year_labels, make_labels, model_labels, bodytype_labels, view_labels, color_labels], dim=1)
        labels = torch.cat([year_probs, make_probs, model_probs, bodytype_probs, view_logits, color_logits], dim=1)
        labels = labels.squeeze(1)
        return labels


class SelfAttention(torch.nn.Module):
    """
    from: https://github.com/akanimax/attn_gan_pytorch/blob/master/attn_gan_pytorch/CustomLayers.py
    Layer implements the self-attention module
    which is the main logic behind this architecture.
    args:
        channels: number of channels in the image tensor
        activation: activation function to be applied (default: lrelu(0.2))
        squeeze_factor: squeeze factor for query and keys (default: 8)
        bias: whether to apply bias or not (default: True)
    """

    def __init__(self, channels, activation=None, squeeze_factor=8, bias=True):
        """ constructor for the layer """


        # base constructor call
        super().__init__()

        # state of the layer
        self.activation = activation
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        # Modules required for computations
        self.query_conv = torch.nn.Conv2d(  # query convolution
            in_channels=channels,
            out_channels=channels // squeeze_factor,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=bias
        )

        self.key_conv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels // squeeze_factor,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=bias
        )

        self.value_conv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=bias
        )

        # softmax module for applying attention
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, return_attention_map = False):
        """
        forward computations of the layer
        :param x: input feature maps (B x C x H x W)
        :return:
            out: self attention value + input feature (B x O x H x W)
            attention: attention map (B x C x H x W)
        """

        # extract the shape of the input tensor
        m_batchsize, c, height, width = x.size()

        # create the query projection
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B x (N) x C

        # create the key projection
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B x C x (N)

        # calculate the attention maps -- softmax
        energy = torch.bmm(proj_query, proj_key)  # energy
        attention = self.softmax(energy)  # attention (B x (N) x (N))

        # create the value projection
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        # calculate the output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, c, height, width)

        attention = attention.view(m_batchsize, -1, height, width)

        if self.activation is not None:
            out = self.activation(out)

        out = self.gamma * out + x
        if return_attention_map:
            return out, attention
        else:
            return out

class SpectralNormManual(object):
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda(non_blocking=(c.gpu_count > 1))
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNormManual(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, inputs):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNormManual.apply(module, name)
    return module