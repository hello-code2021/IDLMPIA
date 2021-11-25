# coding=gbk
import os
import torch
import matplotlib.pyplot as plt
from network.ClassicNetwork.InceptionV2 import InceptionV2
from torchvision import datasets
import time
import numpy as np
from utils.dataloader import data_transforms as data_transforms
from utils import opt

NUM_CLASS = 6

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def save_file(path, list):
    f=open(path,'w')
    for line in list:
        f.write(line)
    f.close()

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):

        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * TP / (2 * TP + FP + FN), 3) if TP + FP + FN != 0 else 0.
        print("Precision:{},  Recall:{},  Specificity:{},  F1:{}".format(Precision, Recall, Specificity, F1))

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)


        plt.xticks(range(self.num_classes), self.labels, rotation=45)

        plt.yticks(range(self.num_classes), self.labels)

        # plt.colorbar()
        # plt.xlabel('True Labels')
        # plt.ylabel('Predicted Labels')
        # plt.title('Confusion matrix')


        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):

                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

def main():

    test_data = datasets.ImageFolder(root=os.path.join(opt.test_data_dir, "test"),
                                     transform=data_transforms["test"])
    print("test data dir:{}".format(os.path.join(opt.test_data_dir, "test")))
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=16, shuffle=True,
                                              num_workers=16, pin_memory=True)

    labels = ['Babesia', 'Leishmania', 'Plasmodium', 'Toxoplasma', 'Trichomonad', 'Trypanosome']
    confusion = ConfusionMatrix(num_classes=NUM_CLASS, labels=labels)
    print(labels)
    # create model
    model = InceptionV2(num_classes=NUM_CLASS).to(device)

    model.load_state_dict(torch.load(os.path.join(opt.model_weight_path,'InceptionV2', 'net_best.pth'), map_location=device))

    model.eval()

    print('start to test')
    f_acc1 = open(os.path.join(opt.test_result_dir, 'InceptionV2', 'test_acc.txt'), 'a')


    with torch.no_grad():
        correct = 0
        total = 0
        labels_list = []
        predited_list = []
        preValue_list = []

        for data in test_loader:
            since = time.time()
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preValue, predicted = torch.max(outputs.data, 1)

            confusion.update(predicted.to("cpu").numpy(), labels.to("cpu").numpy())
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
            for i in predicted:
                predited_list.append(str(i.item()) + '\n')
            for i in labels.data:
                labels_list.append(str(i.item()) + '\n')
            for i in outputs.cpu().data.numpy():
                preValue_list.append(i)

        acc = 100. * correct / total
        f_acc1.write(str(float(acc)) + '\n')
        print('accuracy:{}%, time:{}'.format(round(acc.item(), 3), time.time() - since))


        save_file(os.path.join(opt.test_result_dir, 'InceptionV2', 'con_predicted.txt'), predited_list)
        save_file(os.path.join(opt.test_result_dir, 'InceptionV2', 'con_labels.txt'), labels_list)

    confusion.plot()
    confusion.summary()

if __name__ == '__main__':
    main()
