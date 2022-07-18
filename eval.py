import torch
from model import Encoder
import numpy as np
import argparse

from load_images import ImageList
import transforms

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
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

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
    parser.add_argument('--resnet', default="resnet50", help="Resnet backbone")

    args = parser.parse_args() 


    if args.dataset == "cub2011":
        file_path = {
            "cub": "./data_files/cub200/cub200_2011.txt" ,
            "drawing": "./data_files/cub200/cub200_drawing.txt" ,
        }
    elif args.dataset == "domainNet":
        file_path = {
        "real": "./data/DomainNet/real_test.txt" ,
        "sketch": "./data/DomainNet/sketch_test.txt" ,
        "painting": "./data/DomainNet/painting_test.txt" ,
        "clipart": "./data/DomainNet/clipart_test.txt"}
    else:
        raise NotImplementedError

    dataset_test = file_path[args.target]
    print("Target" , args.target)

    dataset_loaders = {}

    dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(), transform=transforms.image_test(resize_size=256, crop_size=224))
    print("Size of target dataset:" , len(dataset_list))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=16, drop_last=False)

    # network construction
    print(args.nClasses)
    base_network = Encoder(args.resnet, 256, args.nClasses).cuda()
        
    accuracy = AverageMeter()

    saved_state_dict = torch.load(args.checkpoint)
    base_network.load_state_dict(saved_state_dict, strict=True)
    base_network.eval()
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
            _, outputs = base_network(inputs)
            predictions = outputs.argmax(1)
            correct = torch.sum((predictions == labels).float())
            accuracy.update(correct, len(outputs))
    print_str = "\nCorrect Predictions: {}/{}".format(int(accuracy.sum), accuracy.count)
    print_str1 = '\ntest_acc:{:.4f}'.format(accuracy.avg)
    print(print_str + print_str1)