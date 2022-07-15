import numpy as np
from numpy.core.fromnumeric import resize
from torchvision import transforms
import os
# from PIL import Image, ImageOps
# import numbers
# import torch
# from torchvision.transforms.transforms import CenterCrop
# from randaugment import RandAugmentMC

# def image_target(resize_size=256, crop_size=224):
#     return transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
#             RandAugmentMC(n=2, m=10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#     ])

def image_train(resize_size=256, crop_size=224):
#     return transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#     ])

  return  transforms.Compose([
        transforms.Resize(resize_size),
        # ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])
#   return transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

def image_test(resize_size=256, crop_size=224):
  #ten crops for image when validation, input the data_transforms dictionary
  start_first = 0
  start_center = (resize_size - crop_size - 1) / 2
  start_last = resize_size - crop_size - 1
 
  return transforms.Compose([
    transforms.Resize(resize_size),
    # ResizeImage(resize_size),
    # PlaceCrop(crop_size, start_center, start_center),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   # std=[0.229, 0.224, 0.225])
  ])

# def image_test_10crop(resize_size=256, crop_size=224):
#   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#   #ten crops for image when validation, input the data_transforms dictionary
#   start_first = 0
#   start_center = (resize_size - crop_size - 1) / 2
#   start_last = resize_size - crop_size - 1
#   data_transforms = {}
#   data_transforms['val0'] = transforms.Compose([
#       ResizeImage(resize_size),ForceFlip(),
#       PlaceCrop(crop_size, start_first, start_first),
#       transforms.ToTensor(),
#       normalize
#   ])
#   data_transforms['val1'] = transforms.Compose([
#       ResizeImage(resize_size),ForceFlip(),
#       PlaceCrop(crop_size, start_last, start_last),
#       transforms.ToTensor(),
#       normalize
#   ])
#   data_transforms['val2'] = transforms.Compose([
#       ResizeImage(resize_size),ForceFlip(),
#       PlaceCrop(crop_size, start_last, start_first),
#       transforms.ToTensor(),
#       normalize
#   ])
#   data_transforms['val3'] = transforms.Compose([
#       ResizeImage(resize_size),ForceFlip(),
#       PlaceCrop(crop_size, start_first, start_last),
#       transforms.ToTensor(),
#       normalize
#   ])
#   data_transforms['val4'] = transforms.Compose([
#       ResizeImage(resize_size),ForceFlip(),
#       PlaceCrop(crop_size, start_center, start_center),
#       transforms.ToTensor(),
#       normalize
#   ])
#   data_transforms['val5'] = transforms.Compose([
#       ResizeImage(resize_size),
#       PlaceCrop(crop_size, start_first, start_first),
#       transforms.ToTensor(),
#       normalize
#   ])
#   data_transforms['val6'] = transforms.Compose([
#     ResizeImage(resize_size),
#     PlaceCrop(crop_size, start_last, start_last),
#     transforms.ToTensor(),
#     normalize
#   ])
#   data_transforms['val7'] = transforms.Compose([
#     ResizeImage(resize_size),
#     PlaceCrop(crop_size, start_last, start_first),
#     transforms.ToTensor(),
#     normalize
#   ])
#   data_transforms['val8'] = transforms.Compose([
#     ResizeImage(resize_size),
#     PlaceCrop(crop_size, start_first, start_last),
#     transforms.ToTensor(),
#     normalize
#   ])
#   data_transforms['val9'] = transforms.Compose([
#     ResizeImage(resize_size),
#     PlaceCrop(crop_size, start_center, start_center),
#     transforms.ToTensor(),
#     normalize
#   ])
#   return data_transforms

