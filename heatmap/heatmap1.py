# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53
@author: mick.yi
入口类
"""
import argparse
import os
import re

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from torchvision import models

from network.ClassicNetwork.AlexNet import AlexNet
from network.ClassicNetwork.DenseNet import DenseNet121
from network.ClassicNetwork.InceptionV1 import InceptionV1
from network.ClassicNetwork.InceptionV2 import InceptionV2
#   from network.ClassicNetwork.InceptionV4 import InceptionV4
from network.ClassicNetwork.VGGNet import VGG16
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """
        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=None):
        """
        :param inputs: [1,3,H,W]
        :param index: class_id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]

        target.backward()

        inputs.backward()
        print(inputs.grad)
        print(type(inputs.grad))
        input()

        return inputs.grad[0]  # [3,H,W]

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam


# def get_net(net_name, weight_path=None):
#     """
#     根据网络名称获取模型
#     :param net_name: 网络名称
#     :param weight_path: 与训练权重路径
#     :return:
#     """
#     pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
#     if net_name in ['vgg', 'vgg16']:
#         net = torch.load(weight_path)
#         print(net)
#         print(type(net))
#         input()
#         # net.load_state_dict()
#     elif net_name == 'vgg19':
#         net = models.vgg19(pretrained=pretrain)
#     elif net_name in ['resnet', 'resnet50']:
#         net = models.resnet50(pretrained=pretrain)
#     elif net_name == 'resnet101':
#         net = models.resnet101(pretrained=pretrain)
#     elif net_name in ['densenet', 'densenet121']:
#         net = models.densenet121(pretrained=pretrain)
#     elif net_name in ['inception']:
#         net = models.inception_v3(pretrained=pretrain)
#     elif net_name in ['mobilenet_v2']:
#         net = models.mobilenet_v2(pretrained=pretrain)
#     elif net_name in ['shufflenet_v2']:
#         net = models.shufflenet_v2_x1_0(pretrained=pretrain)
#     else:
#         raise ValueError('invalid network name:{}'.format(net_name))
#     # 加载指定路径的权重参数
#     if weight_path is not None and net_name.startswith('densenet'):
#         pattern = re.compile(
#             r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
#         state_dict = torch.load(weight_path)
#         for key in list(state_dict.keys()):
#             res = pattern.match(key)
#             if res:
#                 new_key = res.group(1) + res.group(2)
#                 state_dict[new_key] = state_dict[key]
#                 del state_dict[key]
#         net.load_state_dict(state_dict)
#     elif weight_path is not None:
#         net.load_state_dict(torch.load(weight_path))
#     return net

def get_net(weight_path):
    return torch.load(weight_path)


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True).cuda()


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def main(args):
    # 输入
    img = io.imread(args.image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)
    # 输出图像
    image_dict = {}
    # 网络
    # net = get_net(args.network, args.weight_path)
    net = get_net(args.weight_path)
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(inputs, args.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)
    # print(inputs)
    # print(type(inputs))
    # print(inputs.grad)
    # print(type(inputs.grad))
    # input()
    # inputs.grad.backward().zero_() # 梯度置零
    # inputs.backward()
    # print(type(inputs.grad))
    # input()
    # print(inputs.grad.backward())
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='vgg',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./data/para_test/Babesia/Babesia_test_1.jpg',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default='model/VGG16_1209/net.pkl',
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='results/0117test',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
