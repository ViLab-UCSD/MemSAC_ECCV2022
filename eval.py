import torch
import torch.nn as nn
import model as model_no
import numpy as np
import argparse

from data_list import ImageList
import pre_process as prep
from train import Encoder

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# class predictor(nn.Module):
#     def __init__(self, feature_len, cate_num):
#         super(predictor, self).__init__()
#         self.classifier = nn.Linear(feature_len, cate_num)
#         self.classifier.weight.data.normal_(0, 0.01)
#         self.classifier.bias.data.fill_(0.0)

#     def forward(self, features):
#         activations = self.classifier(features)
#         return (activations)

# class fine_net(nn.Module):
#     def __init__(self, total_classes):
#         super(fine_net, self).__init__()
#         self.model_fc = model_no.Resnet50Fc()
#         feature_len = self.model_fc.output_num()
#         self.bottleneck_0 = nn.Linear(feature_len, 256)
#         self.bottleneck_0.weight.data.normal_(0, 0.005)
#         self.bottleneck_0.bias.data.fill_(0.1)
#         self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.ReLU())
#         self.classifier_layer = predictor(256, total_classes)

#     def forward(self, x):
#         features = self.model_fc(x)
#         out_bottleneck = self.bottleneck_layer(features)
#         logits = self.classifier_layer(out_bottleneck)
#         return logits

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MemSAC')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, nargs='?', default='c', help="target dataset")
    parser.add_argument('--target', type=str, nargs='?', default='c', help="target domain")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size should be samples * classes")
    parser.add_argument('--nClasses', type=int, help="#Classes")
    parser.add_argument('--checkpoint' , type=str, help="Checkpoint to load from.")
    parser.add_argument('--multi_gpu', type=int, default=0, help="use dataparallel if 1")
    parser.add_argument('--data_dir', required=True)

    args = parser.parse_args() 

    if args.dataset in ["birds123" , "cub2011" , "domainNet" , "office-home"]:
        if args.dataset == "domainNet":
            file_path = {
                "real": "./data/visDA_126/real.txt" ,
                "sketch": "./data/visDA_126/sketch.txt" ,
                "painting": "./data/visDA_126/painting.txt" ,
                "clipart": "./data/visDA_126/clipart.txt"
            }
        elif args.dataset == "cub2011":
            file_path = {
                "cub": "./data/cub200/cub200_2011.txt" ,
                "drawing": "./data/cub200/cub200_drawing.txt" ,
            }
        elif args.dataset == "birds123":
            file_path = {
                "c": "./data/birds123/bird123_cub2011.txt",
                "n": "./data/birds123/bird123_nabirds_list.txt",
                "i": "./data/birds123/bird123_ina_list_2017.txt"
            }
        elif args.dataset == "office-home":
            file_path = {
                "real_world": "./data/officeHome/Real_World.txt" ,
                "art": "./data/officeHome/Art.txt",
                "product": "./data/officeHome/Product.txt",
            }
        print("Target" , args.target)
        dataset_test = file_path[args.target]
    elif args.dataset == "domainNet_full":
        file_path = {
            "real": "./data/visDA_full/real_train.txt" ,
            "sketch": "./data/visDA_full/sketch_train.txt" ,
            "painting": "./data/visDA_full/painting_train.txt" ,
            "clipart": "./data/visDA_full/clipart_train.txt"}
        print("Target" , args.target)
        dataset_test = file_path[args.target].replace("train" , "test")
    # elif args.dataset== "imagenet_c":
    #     file_path = {
    #         "fog": "data/imagenet/imagenet_val_fog_2.txt",
    #         "brightness": "data/imagenet/imagenet_val_brightness_2.txt",
    #         "defocus_blur": "data/imagenet/imagenet_val_defocus_blur_2.txt",
    #         "zoom_blur": "data/imagenet/imagenet_val_zoom_blur_2.txt",
    #     }
    #     dataset_test = file_path[args.target].replace("_2.txt" , "_5.txt")
    else:
        raise NotImplementedError


    dataset_loaders = {}

    dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(), transform=prep.image_test(resize_size=256, crop_size=224))
    print("Size of target dataset:" , len(dataset_list))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=16, drop_last=False)

    # network construction
    print(args.nClasses)
    my_fine_net = Encoder(256, args.nClasses)
    my_fine_net = my_fine_net.cuda()
        
    accuracy = AverageMeter()

    saved_state_dict = torch.load(args.checkpoint)
    try:
        my_fine_net.load_state_dict(saved_state_dict, strict=True)
    except:
        saved_state_dict = {k.partition("module.")[-1]:v for k,v in saved_state_dict.items()}
        my_fine_net.load_state_dict(saved_state_dict, strict=True)
    my_fine_net.eval()
    start_test = True
    iter_test = iter(dataset_loaders["test"])
    with torch.no_grad():
        for i in range(len(dataset_loaders['test'])):
            print("{0}/{1}".format(i,len(dataset_loaders['test'])) , end="\r")
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, outputs = my_fine_net(inputs)
            predictions = outputs.argmax(1)
            correct = torch.sum((predictions == labels).float())
            accuracy.update(correct/len(outputs), len(outputs))
    print_str = "\nCorrect Predictions: {}/{}".format(int(accuracy.sum), accuracy.count)
    print_str1 = '\ntest_acc:{:.4f}'.format(accuracy.avg)
    print(print_str + print_str1)