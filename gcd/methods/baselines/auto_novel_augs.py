from torchvision import transforms
from methods.baselines.auto_novel_utils import TransformTwice, RandomTranslateWithReflect

def get_aug(aug, image_size=32, mean=None, std=None):

    mean = (0.4914, 0.4822, 0.4465) if mean is None else mean
    std = (0.2023, 0.1994, 0.2010) if std is None else std

    if aug == None:

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif aug == 'once':

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=int(image_size / 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif aug == 'twice':

        transform = TransformTwice(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RandomTranslateWithReflect(int(image_size / 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]))

    else:

        raise NotImplementedError

    return transform