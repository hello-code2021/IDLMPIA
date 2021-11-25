# coding=gbk
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import data_transforms as data_transforms
import torch.utils.data as _data
from network.ClassicNetwork.ResNet import ResNet50
from torchvision import datasets
import os
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import opt

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print('device:', device)
CLASS_NUMBER = 6


EPOCH = opt.epoch
pre_epoch = 0
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 5e-4
STEP_SIZE=50
GAMMA=0.1




def save_file(path, list):
    f=open(path,'w')
    for line in list:
        f.write(line)
    f.close()


def main():

    best_acc = 0.00
    net = ResNet50(num_classes=CLASS_NUMBER).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)



    train_data = datasets.ImageFolder(root=os.path.join(opt.train_dataset_dir, "train"),
                                         transform=data_transforms["train"])
    val_data = datasets.ImageFolder(root=os.path.join(opt.train_dataset_dir, "val"),
                                      transform=data_transforms["val"])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=16, shuffle=True,
                                               num_workers=16, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                                  batch_size=16, shuffle=True,
                                                  num_workers=16, pin_memory=True)

    f_loss = open(os.path.join(opt.train_result, 'ResNet50', 'train_loss.txt'), 'a')
    f_acc = open(os.path.join(opt.train_result, 'ResNet50', 'train_acc.txt'), 'a')

    print("Start Training ResNet50")

    for epoch in range(pre_epoch, EPOCH):
        since = time.time()
        print('\n Epoch: {}'.format(epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        length = len(train_data)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, pre = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(pre == labels.data)
            if i % 10 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% |time:%.3f'
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, time.time() - since))

        scheduler.step(epoch)
        f_loss.write(str(float(sum_loss / (i + 1))) + '\n')
        f_acc.write(str(float(100. * correct / total)) + '\n')


        # validation
        if (epoch+1) % 1 == 0:
            print('start to validation')
            f_acc1 = open(os.path.join(opt.train_result, 'ResNet50', 'test_acc.txt'), 'a')
            f_loss1 = open(os.path.join(opt.train_result, 'ResNet50', 'test_loss.txt'), 'a')
            with torch.no_grad():
                correct = 0
                total = 0
                labels_list = []
                predited_list = []
                preValue_list = []
                loss = 0.0
                for data in val_loader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    preValue, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels.data)
                    for i in predicted:
                        predited_list.append(str(i.item()) + '\n')
                    for i in labels.data:
                        labels_list.append(str(i.item()) + '\n')
                    for i in outputs.cpu().data.numpy():
                        preValue_list.append(i)
                acc = 100. * correct / total
                f_loss1.write(str(loss.item()) + '\n')
                f_acc1.write(str(float(acc)) + '\n')
                print('the test accuracy is:{}%, time:{}'.format(round(acc.item(), 3), time.time() - since))

                if acc >= best_acc:
                    best_acc = acc
                    print("upgrade best_acc:{}".format(best_acc))
                    torch.save(net.state_dict(), os.path.join(opt.train_result, 'ResNet50', 'net_best.pth'))

                else:
                    print("the best_acc was:{},no upgrade best_acc, no save model".format(best_acc))
                if acc>= 99:
                    torch.save(net.state_dict(), os.path.join(opt.train_result, 'AlexNet', 'net_latest.pth'))
                    return


    torch.save(net.state_dict(), os.path.join(opt.train_result, 'ResNet50', 'net_latest.pth'))

if __name__ == '__main__':
    main()
