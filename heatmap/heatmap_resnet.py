import torch
import argparse
import cv2
import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Function
from torchvision import models, transforms
# from utils.dataloader import MyDataSet
from torch import nn

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.target_layers = target_layers
        self.gradients = []

    def get_gradients(self):
        return self.gradients

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        target_activations = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # print(name)
            # print(self.target_layers)
            if name in self.target_layers:
                x = x.view(x.size(0), -1)
                x = module(x)
                x.register_hook(self.save_gradient)
                target_activations += [x]
            elif "avgpool" in name.lower():
                x = module(x)
                x.register_hook(self.save_gradient)
                target_activations += [x]
                # x = x.view(x.size(0), -1)
            else:
                x = module(x)
                x.register_hook(self.save_gradient)
                target_activations += [x]

        return target_activations, x


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return preprocessing(img)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        # print(self.feature_module)
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        # print(self.feature_module.weight.grad)
        # input()

        # self.feature_module.zero_grad()
        # self.model.zero_grad()
        # ico = 0

        # print(self.model.conv1)
        # print(type(self.model.conv1._modules))
        # print(self.model.conv1._modules)
        dicttmp = self.model.conv1._modules
        for name in self.model.conv1._modules:
            try:
                dicttmp[name].weight.grad = None
            except AttributeError as e:
                pass

        dicttmp = self.model.layer1._modules
        for name in self.model.layer1._modules:
            bot = dicttmp[name].bottleneck._modules
            for nm2 in bot:
                try:
                    bot[nm2].weight.grad = None
                except AttributeError as e:
                    pass
            if dicttmp[name].downsampling:
                bot = dicttmp[name].downsample._modules
                for nm3 in bot:
                    try:
                        bot[nm3].weight.grad = None
                    except AttributeError as e:
                        pass

        dicttmp = self.model.layer2._modules
        for name in self.model.layer2._modules:
            bot = dicttmp[name].bottleneck._modules
            for nm2 in bot:
                try:
                    bot[nm2].weight.grad = None
                except AttributeError as e:
                    pass
            if dicttmp[name].downsampling:
                bot = dicttmp[name].downsample._modules
                for nm3 in bot:
                    try:
                        bot[nm3].weight.grad = None
                    except AttributeError as e:
                        pass
        dicttmp = self.model.layer3._modules
        for name in self.model.layer3._modules:
            bot = dicttmp[name].bottleneck._modules
            for nm2 in bot:
                try:
                    bot[nm2].weight.grad = None
                except AttributeError as e:
                    pass
            if dicttmp[name].downsampling:
                bot = dicttmp[name].downsample._modules
                for nm3 in bot:
                    try:
                        bot[nm3].weight.grad = None
                    except AttributeError as e:
                        pass

        dicttmp = self.model.layer4._modules
        for name in self.model.layer4._modules:
            bot = dicttmp[name].bottleneck._modules
            for nm2 in bot:
                try:
                    bot[nm2].weight.grad = None
                except AttributeError as e:
                    pass
            if dicttmp[name].downsampling:
                bot = dicttmp[name].downsample._modules
                for nm3 in bot:
                    try:
                        bot[nm3].weight.grad = None
                    except AttributeError as e:
                        pass


        self.feature_module.weight.grad = None
        # input()
        one_hot.backward(retain_graph=True)

        # print(len(self.extractor.get_gradients()))
        # for fea in features:
        #     print(fea.shape)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        # print(grads_val.shape)
        target = features[0]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        # print(type(model))
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output, feature = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--weight-path', type=str, default='./model/res50_1205/net.pkl',
                        help='pretrained weight path')
    parser.add_argument('--image-path', type=str, default='./data/para_test',
                        help='Input image path')
    parser.add_argument('--output-path', type=str, default='./results/0117test/resnet',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def get_last_conv_name(net):
    """
    ?????????????????????????????????????????????
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    # model = models.resnet50()
    # print(model)
    # for name, module in model.layer4._modules.items():
    #     print(name)

    model = torch.load(args.weight_path)
    # print(model)
    # print(model)
    # input()
    layer4 = None
    name4 = None
    for name, module in model._modules.items():
        layer4 = module
    # print(model.layer4._modules.named_)
    for name, module in model._modules.items():
        name4 = name
    # print(model.layer4)
    # input()
    model.layer4.zero_grad()
    grad_cam = GradCam(model=model, feature_module=layer4,
                       target_layer_names=[name4], use_cuda=args.use_cuda)

    patht = args.image_path
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    for dir in os.listdir(patht):
        folderpath = patht + '/' + dir
        outfolderpath = args.output_path + '/' + dir
        if not os.path.exists(outfolderpath):
            os.mkdir(outfolderpath)
        # print(outfolderpath)
        # input()
        count = 0
        oplen = min(len(os.listdir(folderpath)), 20)
        for img_name in os.listdir(folderpath):
            count += 1
            if count > oplen:
                break
            print("{}/{}".format(count, oplen))
            image_path = folderpath + '/' + img_name
            iop = outfolderpath + '/' + img_name.split('.')[0]
            # print(image_path)
            # print(iop)
            img = Image.open(image_path)
            img = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])(img)
            # print(type(img))
            # input()
            img = np.float32(img) / 255
            # Opencv loads as BGR:
            img = img[:, :, ::-1]
            cv2.imwrite(iop + "_resize.jpg", np.uint8(255 * img))
            # input()
            input_img = preprocess_image(Image.open(image_path))
            # print(input_img)
            # print(type(input_img))
            # print(input_img.shape)
            input_img = input_img.unsqueeze(0)
            # print(input_img.shape)
            # input()
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = None
            grayscale_cam = grad_cam(input_img, target_category)

            grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
            cam = show_cam_on_image(img, grayscale_cam)

            gb = gb_model(input_img, target_category=target_category)
            gb = gb.transpose((1, 2, 0))

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)

            cv2.imwrite(iop + "_cam.jpg", cam)
            cv2.imwrite(iop + '_gb.jpg', gb)
            cv2.imwrite(iop + '_cam_gb.jpg', cam_gb)
