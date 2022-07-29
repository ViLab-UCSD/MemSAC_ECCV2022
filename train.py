import torch
import torch.optim as optim
import torch.nn as nn
from model.model import Encoder, AdversarialLayer, discriminator
import numpy as np
import argparse
import os

from model.memory import MemoryModule
from data_loader.load_images import ImageList
import data_loader.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import torch.backends.cudnn as cudnn
cudnn.enabled = False
torch.backends.cudnn.deterministic=True

import numpy as np
seed=1234
torch.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def test_target(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for data in iter_test:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).detach() / float(all_label.size()[0])
    return accuracy

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer Learning')

    ## Training parameters
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', required=True, help="Name of the dataset")
    parser.add_argument('--source', type=str, nargs='?', default='c', help="source dataset")
    parser.add_argument('--target', type=str, nargs='?', default='p', help="target dataset")
    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--max_iteration', type=int, nargs='?', default=102500, help="target dataset")
    parser.add_argument('--out_dir', type=str, nargs='?', default='e', help="output dir")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size should be samples * classes")
    parser.add_argument('--data_dir', type=str, default="./data", help="Path for data directory")
    parser.add_argument('--multi_gpu', type=int, default=0)
    parser.add_argument('--total_classes', type=int, default=31, help="total # classes in the dataset")

    ## Testing parameters
    parser.add_argument('--test_10crop', action="store_true", help="10 crop testing")
    parser.add_argument('--test-iter', type=int, default=10000, help="Testing freq.")

    ## Architecture
    parser.add_argument('--resnet', default="resnet50", help="Resnet backbone")
    parser.add_argument('--bn-dim', type=int, default=256, help="bottleneck embedding dimension")

    ## Adaptation parameters
    parser.add_argument('--only_da_iter', type=int, default=100,
                        help="number of iterations when only DA loss works and MSC doesn't")
    parser.add_argument('--simi_func', type=str, default='cosine', choices=['cosine', 'euclidean', "gaussian"])
    parser.add_argument('--method', type=str, default="MemSAC")
    parser.add_argument('--knn_method', type=str, nargs='?', default='ranking', choices=['ranking', 'classic'])
    parser.add_argument('--ranking_k', type=int, default=4, help="use number of samples per class")
    parser.add_argument('--top_ranked_n', type=int, default=20,
                        help="these many target samples are used finally, 1/3 of batch")
    parser.add_argument('--k', type=int, default=5, help="k for knn")

    ## Memory network
    parser.add_argument('--queue_size', type=int, default=24000, help="size of queue")
    parser.add_argument('--momentum', type=float, default=0, help="momentum value")
    parser.add_argument('--tau', type=float, default=0.07, help="temperature value")

    ## Loss coeffecients
    parser.add_argument('--sim-coeff', type=float, default=0.1, help="coeff for similarity loss")
    parser.add_argument('--adv-coeff', type=float, default=1., help="Adversarial Loss")

    args = parser.parse_args() 
    out_dir = os.path.join("work_dirs" , args.dataset , args.out_dir )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "log.txt")
    log_acc = os.path.join(out_dir, "logAcc.txt")
    print("Writing log to" , out_file)
    out_file = open(out_file, "w")
    best_file = os.path.join(out_dir, "best.txt")
    args.multi_gpu = bool(args.multi_gpu)
    print(args)

    ##### TensorBoard & Misc Setup #####
    writer_loc = os.path.join(out_dir , 'tensorboard_logs')
    writer = SummaryWriter(writer_loc)

    if args.dataset == "cub2011":
        file_path = {
            "cub": "./data_files/cub200/cub200_2011.txt" ,
            "drawing": "./data_files/cub200/cub200_drawing.txt" ,
        }
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target]
    elif args.dataset == "domainNet":
        file_path = {
            "real": "./data_files/DomainNet/real_train.txt" ,
            "sketch": "./data_files/DomainNet/sketch_train.txt" ,
            "painting": "./data_files/DomainNet/painting_train.txt" ,
            "clipart": "./data_files/DomainNet/clipart_train.txt"}
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target].replace("train" , "test")
    else:
        raise NotImplementedError
    print("Source: " , args.source)
    print("Target" , args.target)

    batch_size = {"train": args.batch_size, "val": args.batch_size*4}
    
    out_file.write('args = {}\n'.format(args))
    out_file.flush()

    dataset_loaders = {}
    print(dataset_source)

    dataset_list = ImageList(args.data_dir, open(dataset_source).readlines(),
                             transform=transforms.image_train(resize_size=256, crop_size=224))
    
    print(f"{len(dataset_list)} source samples")

    dataset_loaders["source"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'],
                                                               shuffle=True, num_workers=8,
                                                               drop_last=True)


    dataset_list = ImageList(args.data_dir, open(dataset_target).readlines(),
                             transform=transforms.image_train(resize_size=256, crop_size=224))
    dataset_loaders["target"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'], shuffle=True,
                                                         num_workers=8, drop_last=True)

    print(f"{len(dataset_list)} target samples")

    dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(),
                                transform=transforms.image_test(resize_size=256, crop_size=224))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['val'], shuffle=False,
                                                          num_workers=8)
    print(f"{len(dataset_list)} target test samples")

    # network construction
    base_network = Encoder(args.resnet, args.bn_dim, args.total_classes)
    base_network = base_network.cuda()

    discriminator = discriminator(args.bn_dim, args.total_classes).cuda()
    discriminator.train(True)

    # gradient reversal layer
    grl = AdversarialLayer()

    # criterion and optimizer
    criterion = {
        "classifier" : nn.CrossEntropyLoss(),
        "adversarial": nn.BCEWithLogitsLoss()
    }

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, base_network.model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, base_network.bottleneck_0.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, base_network.classifier_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, discriminator.parameters()), "lr": 1}  # ,
    ]

    optimizer = optim.SGD(optimizer_dict, momentum=0.9, weight_decay=0.0005)

    if args.multi_gpu:
        base_network = nn.DataParallel(base_network).cuda()

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    len_source = len(dataset_loaders["source"]) - 1
    len_target = len(dataset_loaders["target"]) - 1
    iter_source = iter(dataset_loaders["source"])
    iter_target = iter(dataset_loaders["target"])

    
    memory_network = MemoryModule(args.bn_dim, K=args.queue_size, m=args.momentum, T=args.tau, knn=args.k, top_ranked_n=args.top_ranked_n, similarity_func=args.simi_func, batch_size=batch_size["train"], ranking_k=args.ranking_k)
    memory_network = memory_network.cuda()

    with open(os.path.join(out_dir , "best.txt"), "a") as fh:
        fh.write("Best Accuracy file\n")

    if os.path.exists(os.path.join(out_dir , "checkpoint.pth.tar")):
        print("Loading from pretrained model ...")
        checkpoint = torch.load(os.path.join(out_dir , "checkpoint.pth.tar"))
        base_network.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        memory_network.load_state_dict(checkpoint["memory_state_dict"])
        start_iter = checkpoint["iter"]

    start_iter=1
    best_acc = 0

    for iter_num in range(start_iter, args.max_iteration + 1):
        base_network.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer.zero_grad()
        print("Iter:" , iter_num , end="\r")

        if iter_num % len_source == 0:
            iter_source = iter(dataset_loaders["source"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loaders["target"])
        data_source = iter_source.next()
        data_target = iter_target.next()
        inputs_source, labels_source = data_source
        inputs_target, _ = data_target
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        inputs = inputs.cuda()

        labels_source = labels_source.cuda()
        assert len(inputs_source) == len(inputs_target)
        domain_labels = torch.tensor([[1], ] * len(inputs_source)+ [[0], ] * len(inputs_target), device=torch.device('cuda:0'), dtype=torch.float)

        features, logits = base_network(inputs)
        logits_source = logits[:len(inputs_source)]
        logits_target = logits[len(inputs_source):]

        pseudo_labels_target = logits_target.argmax(1)

        ## Classifier Loss
        classifier_loss = criterion["classifier"](logits_source, labels_source)

        ## CDAN Loss
        domain_predicted = discriminator(grl.apply(features), torch.softmax(logits, dim=1).detach())
        transfer_loss = criterion["adversarial"](domain_predicted, domain_labels)
        transfer_loss = args.adv_coeff*transfer_loss

        ## MemSAC Loss
        sim_loss = memory_network(features, labels_source)
        sim_loss = args.sim_coeff*sim_loss*(iter_num > args.only_da_iter)

        ## Total Loss
        total_loss = classifier_loss + transfer_loss + sim_loss

        total_loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/classifier_loss" , classifier_loss.detach(), iter_num)
        writer.add_scalar("Loss/transfer" , transfer_loss.detach(), iter_num)
        writer.add_scalar("Loss/sim_loss" , sim_loss.detach(), iter_num)

        # test
        test_interval = args.test_iter
        if iter_num % test_interval == 0:
            print()
            base_network.eval()
            test_acc = test_target(dataset_loaders, base_network)
            writer.add_scalar("Acc/test" , test_acc , iter_num)
            print_str1 = '\niter: {:05d}, test_acc:{:.4f}\n'.format(iter_num, test_acc)
            print(print_str1)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = base_network.state_dict()
                with open(os.path.join(out_dir , "best.txt"), "a") as fh:
                    fh.write("Best Accuracy : {:.4f} at iter: {:05d}\n".format(best_acc, iter_num))
                torch.save(best_model , os.path.join(out_dir , "best_model.pth.tar"))

            checkpoint_dict = {
                "state_dict" : base_network.state_dict(),
                "discriminator_state_dict" : discriminator.state_dict(),
                "memory_state_dict" : memory_network.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "iter" : iter_num+1,
                "args" : args
            }
            torch.save(checkpoint_dict , os.path.join(out_dir , "checkpoint.pth.tar"))
