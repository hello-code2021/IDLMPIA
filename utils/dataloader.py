import torch

torch.backends.cudnn.benchmark = True
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os


class6 = ['Babesia', 'Leishmania', 'Plasmodium', 'Toxoplasma', 'Trichomonad', 'Trypanosome']

mean_train = [0, 0, 0]
std_train = [255, 255, 255]

mean_test = [0, 0, 0]
std_test = [255, 255, 255]

mean_val = [0, 0, 0]
std_val = [255, 255, 255]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_test, std_test)
    ]),
    'val': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_val, std_val)
    ]),
    'test': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_test,std_test)
    ])
}


class MyDataSet(data.Dataset):
    def __init__(self, path, type_data='train'):
        self.data = []
        self.transform = data_transforms[type_data]
        for i, class1 in enumerate(class6):
            img_list = os.listdir(os.path.join(path, class1))
            for img in img_list:
                img_data = dict()
                img_data['path'] = os.path.join(path, class1, img)
                img_data['label'] = i
                self.data.append(img_data)

    def __getitem__(self, index):
        img_path = self.data[index]['path']
        img_label = self.data[index]['label']

        img = self.transform(Image.open(img_path))
        return img, img_label

    def __len__(self):
        return len(self.data)
