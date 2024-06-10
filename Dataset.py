import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class TrainSet(Dataset):
    def __init__(self, images_dir, labels_path):
        self.images_dir = images_dir
        self.labels_path = labels_path
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.labels = self._load_labels()

    def _load_labels(self):
        labels = []
        with open(self.labels_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 2:
                    raise ValueError("每行应该有两个元素: 文件名和标签")
                img_name, label = parts
                labels.append((img_name, int(label)))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        image_path = os.path.join(self.images_dir, img_name)
        image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_all_data(self):
        images = []
        labels = []

        for idx in range(len(self)):
            image, label = self[idx]
            images.append(image)
            labels.append(label)

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)
        return images_tensor, labels_tensor
