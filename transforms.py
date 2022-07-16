from torchvision import transforms

def image_train(resize_size=256, crop_size=224):

  return  transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])
    
def image_test(resize_size=256, crop_size=224):
  #ten crops for image when validation, input the data_transforms dictionary
 
  return transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  ])
