import torch
from torchvision.utils import save_image

tensor = torch.load("./submit.pt")

for i in range(tensor.size(0)):
    img = tensor[i]
    save_image(img, f'./submit/{i}.png')
