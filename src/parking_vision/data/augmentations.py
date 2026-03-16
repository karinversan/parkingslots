from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_augmentations(image_size: int):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.4),
            A.RandomShadow(p=0.3),
            A.RandomRain(p=0.2),
            A.RandomFog(p=0.15),
            A.GaussNoise(p=0.25),
            A.MotionBlur(p=0.15),
            A.CLAHE(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def build_eval_augmentations(image_size: int):
    return A.Compose([A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()])
