import albumentations as A
import cv2


def train_transform(image_size = 32):
    return A.Compose([
        A.Resize(image_size, image_size),

        A.PadIfNeeded(
            min_height=image_size + 8,
            min_width=image_size + 8,
            border_mode=0,
            p=0.8,
        ),
        A.RandomCrop(image_size, image_size),

        A.HorizontalFlip(p=0.5),
    ])


def test_transform(image_size = 32):
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
