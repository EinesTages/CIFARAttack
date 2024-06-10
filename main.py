import sys
import torch
import torch.nn as nn
import torchattacks
import robustbench
from robustbench.utils import load_model, clean_accuracy
from Dataset import TrainSet
from torch.utils.data import DataLoader, TensorDataset
from module import CIFAR10Model
import argparse

from torchattacks import PGD

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Adversarial Attack with Different Epsilons')
parser.add_argument('--eps', type=float, default=8.0, help='Epsilon for the attacks')
args = parser.parse_args()

eps = args.eps / 255

all_classifiers = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "resnet18", "resnet34", "resnet50", "densenet121",
                   "densenet161", "densenet169", "mobilenet_v2", "googlenet", "inception_v3",
                   'Wang2023Better_WRN-28-10', 'Wang2023Better_WRN-70-16', 'Sridhar2021Robust_34_15',
                   'Sehwag2021Proxy_ResNest152', 'Sehwag2021Proxy_R18', 'Rebuffi2021Fixing_R18_ddpm',
                   'Rebuffi2021Fixing_106_16_cutmix_ddpm', 'Engstrom2019Robustness', 'Debenedetti2022Light_XCiT-M12',
                   'Bai2024MixedNUTS', 'Bai2023Improving_edm']
device = "cuda:0"
dataset = TrainSet("./data/cifar10_clean_500/images", "./data/cifar10_clean_500/label.txt")
images, labels = dataset.get_all_data()
images, labels = images.to(device), labels.to(device)
batch_size = 32
tensor_dataset = TensorDataset(images, labels)
data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)


acc_all = 0
l2_all = 0
for i in range(24):
    if i < 13:
        model = CIFAR10Model(all_classifiers[i]).to(device)
    else:
        model = load_model(all_classifiers[i], norm='Linf').to(device)
    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print(all_classifiers[i])
    print('[Model loaded]')
    print('Acc: %2.2f %%' % (acc * 100))
    atk1 = torchattacks.FGSM(model, eps=eps)
    atk2 = torchattacks.PGD(model, eps=eps, alpha=2 / 255, steps=80, random_start=True)
    atk = torchattacks.MultiAttack([atk1, atk2])
    res = atk.save(data_loader=data_loader,
                   save_path=f"./results/" + all_classifiers[i] + '.pt',
                   save_clean_images=False,
                   return_verbose=True)
    acc_all += res[0]
    l2_all += res[2]
    del model
    torch.cuda.empty_cache()
print(acc_all)
print(l2_all)
