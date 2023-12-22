"""
Author: Le Gia Tai
"""
import os
from pathlib import Path

import numpy as np
import torchvision.datasets as torch_dataset

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from torchvision.datasets.folder import DatasetFolder, default_loader, has_file_allowed_extension
from typing import List

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class ImageNet30(Dataset):
    """
    ImageNet30
    """
    root = os.path.join(Path.home(), "Documents/path/to/imagenet30")

    def __init__(self, train, chosen_classes: List[int], transform):
        super(ImageNet30, self).__init__()
        self.transform = transform

        if train:
            root = os.path.join(self.root, "train")

        else:
            root = os.path.join(self.root, "test")

        self.data = torch_dataset.ImageFolder(root=root, is_valid_file=self.is_valid_file)

        if chosen_classes is None:
            chosen_classes = [i for i in range(30)]

        if not isinstance(chosen_classes, list):
            chosen_classes = [chosen_classes]

        indices = []

        for idx, tgt in enumerate(self.data.targets):
            if tgt in chosen_classes:
                indices.append(idx)

        self.data = Subset(self.data, indices)

    def __getitem__(self, item):
        img, target = self.data.__getitem__(item)

        return {"image": self.transform(img), "target": target}

    def __len__(self):
        return len(self.data)

    @staticmethod
    def is_valid_file(file: str) -> bool:
        return has_file_allowed_extension(file, IMG_EXTENSIONS) and not file.endswith('airliner/._1.JPEG')


class ImageNet1k(torch_dataset.ImageFolder):
    """
    ImageNet1k as a surrogate for OE in case we don't want to use the cumbersome ImageNet21k dataset as OE.
    """
    remove_classes = [
        ('acorn', 'n12267677'),
        ('airliner', 'n02690373'),
        ('ambulance', 'n02701002'),
        ('american_alligator', 'n01698640'),
        ('banjo', 'n02787622'),
        ('barn', 'n02793495'),
        ('bikini', 'n02837789'),
        ('digital_clock', 'n03196217'),
        ('dragonfly', 'n02268443'),
        ('dumbbell', 'n03255030'),
        ('forklift', 'n03384352'),
        ('goblet', 'n03443371'),
        ('grand_piano', 'n03452741'),
        ('hotdog', 'n07697537'),
        ('hourglass', 'n03544143'),
        ('manhole_cover', 'n03717622'),
        ('mosque', 'n03788195'),
        ('nail', 'n03804744'),
        ('parking_meter', 'n03891332'),
        ('pillow', 'n03938244'),
        ('revolver', 'n04086273'),
        ('rotary_dial_telephone', 'n03187595'),
        ('schooner', 'n04147183'),
        ('snowmobile', 'n04252077'),
        ('soccer_ball', 'n04254680'),
        ('stingray', 'n01498041'),
        ('strawberry', 'n07745940'),
        ('tank', 'n04389033'),
        ('toaster', 'n04442312'),
        ('volcano', 'n09472597')
    ]
    root = os.path.join(Path.home(), "Documents/path/to/imagenet1k/train")

    def __init__(self, transform, **kwargs):
        super(DatasetFolder, self).__init__(root=self.root)
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.data = self.make_dataset(directory=self.root, class_to_idx=self.class_to_idx,
                                      extensions=kwargs.get('extensions',
                                                            IMG_EXTENSIONS if kwargs.get('is_valid_file',
                                                                                         None) is None else None),
                                      is_valid_file=kwargs.get("is_valid_file", None))
        self.targets = [s[1] for s in self.data]

        imagenet30_idxs = tuple([self.class_to_idx.get(label[1]) for label in self.remove_classes])
        self.data = [s for s in self.data if s[1] not in imagenet30_idxs]

        for label in self.remove_classes:
            try:
                self.classes.remove(label)
                del self.class_to_idx[label]

            except:
                pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, target = self.data.__getitem__(item)
        img = default_loader(path)

        return {"image": self.transform(img), "target": target}


class DIOR(torch_dataset.ImageFolder):
    CLASSES = ['ship', 'tenniscourt', 'baseballfield', 'groundtrackfield', 'dam', 'bridge', 'trainstation', 'harbor',
               'golffield',
               'Expressway-toll-station', 'overpass', 'basketballcourt', 'airport', 'airplane', 'storagetank',
               'Expressway-Service-area',
               'stadium', 'windmill', 'chimney']
    root = os.path.join(Path.home(), "Documents/path/to/dior_ad")

    def __init__(self, transform, train, chosen_classes: List[int], **kwargs):
        super(DatasetFolder, self).__init__(root=self.root)
        self.transform = transform

        if train:
            self.root = os.path.join(self.root, "train")

        else:
            self.root = os.path.join(self.root, "test")

        self.classes, self.class_to_idx = self.find_classes(self.root)
        print(self.class_to_idx)
        self.data = self.make_dataset(directory=self.root, class_to_idx=self.class_to_idx,
                                      extensions=kwargs.get('extensions',
                                                            IMG_EXTENSIONS if kwargs.get('is_valid_file',
                                                                                         None) is None else None),
                                      is_valid_file=kwargs.get("is_valid_file", None))

        self.targets = [s[1] for s in self.data]

        if chosen_classes is None:
            chosen_classes = [i for i in range(len(self.classes))]

        if not isinstance(chosen_classes, list):
            chosen_classes = [chosen_classes]

        indices = []

        for idx, tgt in enumerate(self.targets):
            if tgt in chosen_classes:
                indices.append(idx)

        self.classes = [self.classes[i] for i in chosen_classes]

        self.data = Subset(self.data, indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, target = self.data.__getitem__(item)
        img = default_loader(path)

        return {"image": self.transform(img), "target": target}


class WBC(torch_dataset.ImageFolder):
    root = os.path.join(Path.home(), "Documents/path/to/wbc")

    def __init__(self, train, transform, chosen_classes, **kwargs):
        super(DatasetFolder, self).__init__(root=self.root)

        self.transform = transform
        if train:
            self.root = os.path.join(self.root, "train")

        else:
            self.root = os.path.join(self.root, "test_A")

        self.classes, self.class_to_idx = self.find_classes(self.root)
        print(self.class_to_idx)
        self.data = self.make_dataset(directory=self.root, class_to_idx=self.class_to_idx,
                                      extensions=kwargs.get('extensions',
                                                            IMG_EXTENSIONS if kwargs.get('is_valid_file',
                                                                                         None) is None else None),
                                      is_valid_file=kwargs.get("is_valid_file", None))

        self.targets = [s[1] for s in self.data]

        if chosen_classes is None:
            chosen_classes = [i for i in range(len(self.classes))]

        if not isinstance(chosen_classes, list):
            chosen_classes = [chosen_classes]

        indices = []

        for idx, tgt in enumerate(self.targets):
            if tgt in chosen_classes:
                indices.append(idx)

        self.classes = [self.classes[i] for i in chosen_classes]
        print("Chosen classes {}".format(self.classes))

        self.data = Subset(self.data, indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, target = self.data.__getitem__(item)
        img = default_loader(path)

        return {"image": self.transform(img), "target": target}


class CIFAR10(torch_dataset.CIFAR10):
    """
    cifar10
    """
    root = os.path.join(Path.home(), "Documents/path/to/cifar10")

    def __init__(self, train, transform, chosen_classes: List[int], download=False):
        super(CIFAR10, self).__init__(root=self.root, train=train, download=download)
        self.classes = chosen_classes  # the chosen chosen_classes
        self.download = download
        self.transform = transform

        if not isinstance(chosen_classes, list):
            chosen_classes = [chosen_classes]

        if isinstance(chosen_classes, list):
            boolean_check = [target in np.array(self.classes) for target in self.targets]
            if True not in boolean_check:
                raise ValueError("There is no chosen_classes with name {}".format(chosen_classes))
            self.data = self.data[boolean_check]
            self.targets = np.array(self.targets)[boolean_check]

        else:
            raise TypeError("Invalid types of chosen_classes: {}".format(type(chosen_classes)))

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        img = Image.fromarray(img)
        img = self.transform(img)

        return {"image": img, "target": target}

    def __len__(self):
        return len(self.data)


class CIFAR100(torch_dataset.CIFAR100):
    """
    cifar100 for OE in case we don't want to use Tiny80M as OE
    """
    root = os.path.join(Path.home(), "Documents/path/to/cifar100")

    def __init__(self, transform, download=False):
        super(CIFAR100, self).__init__(root=self.root, download=download)
        self.transform = transform

    def __getitem__(self, index):
        """
        @param index: index
        @return: {'image': image, 'target': index of target class, "metadata":...}
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        return {"image": self.transform(img)}


class TinyImages(Dataset):
    """
    tiny 80 million images dataset
    """

    def __init__(self, transform, exclude_cifar=True):

        data_file = open(os.path.join(Path.home(), "Documents/path/to/tiny80m", "tiny_images.bin"), "rb")

        self.transform = transform

        def load_image(index):
            """
            @param index:
            @return:
            """
            data_file.seek(index * 3072)
            data = data_file.read(3072)
            return np.frombuffer(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0  # offset index

        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open(os.path.join(Path.home(), "Documents/path/to/tiny80m", '80mn_cifar_idxs.txt'), 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = Image.fromarray(self.load_image(index))

        return {"image": self.transform(img), "target": 0, "metadata": "Tiny80M"}

    def __len__(self):
        return 79302017
