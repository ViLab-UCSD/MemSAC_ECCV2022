#from __future__ import print_function, division

import numpy as np
from PIL import Image
import os.path as osp

def make_dataset(image_list, labels, use_path_for_labels = False):
    if(use_path_for_labels):
        all_make_ids = [img.split("/")[0] for img in image_list]
        set_make_ids = list(set(all_make_ids))
        make_to_class = {idx: _id for idx, _id in enumerate(set_make_ids)}
        images = [( img, make_to_class[img.split("/")[0]] ) for img in image_list]
        return images

    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
        return pil_loader(path)


class ImageList(object):
    """
    Args:
        root (string): Root directory path.
        image_list (file_handler) : File containing list of files in the format <impath> <label>
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader, dataset = None):
        use_path_for_labels = True if dataset == 'compcars' else False
        imgs = make_dataset(image_list, labels, use_path_for_labels=use_path_for_labels)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        path = osp.join(self.root, path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)