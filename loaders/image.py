import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_img(root, augement=False, n_view=2, img_size=224):
    all_filenames = sorted(list(root.glob('h_*.png')), key=lambda x: (len(str(x)), x))
    all_view = len(all_filenames)
    filenames = all_filenames[::all_view//n_view][:n_view]
    assert len(filenames) == n_view
    if augement:
        transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    else:
        transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
    imgs = []
    for v in filenames:
        imgs.append(transform(Image.open(v).convert("RGB")))
    imgs = torch.stack(imgs)
    return imgs


if __name__ == '__main__':
    pass
