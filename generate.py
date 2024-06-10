import sys
import torch
import os
from Dataset import TrainSet
from pytorch_ssim import SSIM
from module import CIFAR10Model
from robustbench.utils import load_model
results_dir = "./results"
all_classifiers = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "resnet18", "resnet34", "resnet50", "densenet121",
                   "densenet161", "densenet169", "mobilenet_v2", "googlenet", "inception_v3",
                   'Wang2023Better_WRN-28-10', 'Wang2023Better_WRN-70-16', 'Sridhar2021Robust_34_15',
                   'Sehwag2021Proxy_ResNest152', 'Sehwag2021Proxy_R18', 'Rebuffi2021Fixing_R18_ddpm',
                   'Rebuffi2021Fixing_106_16_cutmix_ddpm', 'Engstrom2019Robustness', 'Debenedetti2022Light_XCiT-M12',
                   'Bai2024MixedNUTS', 'Bai2023Improving_edm']

dataset = TrainSet("./data/cifar10_clean_500/images", "./data/cifar10_clean_500/label.txt")
device = "cuda:0"
ori_images, ori_labels = dataset.get_all_data()
adv_imgs = []
ssim_score = []
model_list = []
submit = []

ssim_loss = SSIM(window_size=11)
# 这里即便是用8/255无穷范数攻击，也发现基本稳定在0.94~0.95之间，因此基本可以专注攻击性，后续再考虑

for i in range(24):
    save_path = os.path.join(results_dir, all_classifiers[i] + '.pt')
    res = torch.load(save_path, map_location="cuda:0")
    adv_imgs.append(res["adv_inputs"])
    if i < 13:
        model = CIFAR10Model(all_classifiers[i]).to(device)
    else:
        model = load_model(all_classifiers[i], norm='Linf').to(device)
    model_list.append(model)


def evaluate_attack(image, label):
    attack_success_count = 0
    min_label_rank = 100

    for model in model_list:
        with torch.no_grad():
            predictions = model(image)
            predicted_labels = predictions.max(1)[1]
            is_wrong = predicted_labels.item() != label
            if is_wrong:
                attack_success_count += 1

            label_prob = predictions[:, label].item()
            sorted_probs, _ = predictions.sort(descending=True)
            label_rank = (sorted_probs >= label_prob).sum().item()

            if label_rank < min_label_rank:
                min_label_rank = label_rank
    return attack_success_count, min_label_rank


def select_most_adv_image():
    for i in range(500):
        max_wrong_models = 0
        most_attacking_img = adv_imgs[0][i]
        min_label = 0
        for adv_img in adv_imgs:
            # 评估当前对抗样本的攻击性
            s, r = evaluate_attack(adv_img[i].unsqueeze(0), ori_labels[i].to(device))
            if s > max_wrong_models or (s == max_wrong_models and r > min_label):
                min_label = r
                max_wrong_models = s
                most_attacking_img = adv_img[i]
        submit.append(most_attacking_img)
        print(i)
        print(max_wrong_models)
        print(min_label)
    return submit


stacked_adv_imgs = torch.stack(select_most_adv_image())
save_path = "./submit.pt"
torch.save(stacked_adv_imgs, save_path)
