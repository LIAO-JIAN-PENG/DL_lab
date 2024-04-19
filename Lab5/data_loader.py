import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset


def load_image(filename):
    return Image.open(filename)

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale_size: tuple):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale_size = scale_size

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale_size: tuple, is_mask):
        newW, newH = scale_size
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            return img

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        rescale_img = self.preprocess(img, self.scale_size, is_mask=False)
        rescale_mask = self.preprocess(mask, self.scale_size, is_mask=True)

        img = np.asarray(img).transpose((2, 0, 1)) / 255.0
        mask = np.asarray(mask)

        return {
            'name': name,
            'rescale_img': torch.as_tensor(rescale_img.copy()).float().contiguous(),
            'rescale_mask': torch.as_tensor(rescale_mask.copy()).long().contiguous(),
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CCAgTDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale_size=(256, 256)):
        super().__init__(images_dir, mask_dir, scale_size)
