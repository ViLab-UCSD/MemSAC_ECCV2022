import torch
import torch.optim as optim
import torch.nn as nn
import fg_model_no_class as model_no
import fg_adversarial1 as ad
import numpy as np
import argparse
import os

from memoryNetwork import queue_ila
from fg_data_list import ImageList
import fg_pre_process as prep
import math
import criterion_factory as cf
import sampler
import loss

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

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

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

class discriminatorDANN(nn.Module):
    def __init__(self, feature_len):
        super(discriminatorDANN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ad_layer1 = nn.Linear(feature_len, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)

        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3, nn.Sigmoid())

    def forward(self, x, y):
        f2 = self.fc1(x)
        f = self.fc2_3(f2)
        return f


class discriminatorCDAN(nn.Module):
    def __init__(self, feature_len, total_classes):
        super(discriminatorCDAN, self).__init__()

        self.ad_layer1 = nn.Linear(feature_len * total_classes, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3, nn.Sigmoid())

    def forward(self, x, y):
        op_out = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        ad_in = op_out.view(-1, y.size(1) * x.size(1))
        f2 = self.fc1(ad_in)
        f = self.fc2_3(f2)
        return f


class predictor(nn.Module):
    def __init__(self, feature_len, cate_num):
        super(predictor, self).__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return (activations)


class Encoder(nn.Module):
    def __init__(self, total_classes=None):
        super(Encoder, self).__init__()
        self.model_fc = model_no.Resnet50Fc()
        feature_len = self.model_fc.output_num()
        self.bottleneck_0 = nn.Linear(feature_len, 256)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.ReLU())
        self.total_classes = total_classes
        if total_classes:
            self.classifier_layer = predictor(256, total_classes)

    def forward(self, x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        if not self.total_classes:
            return (out_bottleneck, None)
        logits = self.classifier_layer(out_bottleneck)
        return (out_bottleneck, logits)

    # def get_parameters(self): 
    #     parameter_list = [{"params": self.model_fc.parameters(), "lr_mult": 0.1}, \
    #                       {"params": self.bottleneck_layer.parameters(), "lr_mult": 1}]

    #     return parameter_list


if __name__ == '__main__':
    #######NOTE:
    # bottleneck already being used

    parser = argparse.ArgumentParser(description='Transfer Learning')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=str, nargs='?', default='c', help="source dataset")
    parser.add_argument('--target', type=str, nargs='?', default='p', help="target dataset")
    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--max_iteration', type=int, nargs='?', default=102500, help="target dataset")
    parser.add_argument('--out_dir', type=str, nargs='?', default='e', help="output dir")
    parser.add_argument('--sim_net', type=int, default=0, help="whether add source CAS")
    parser.add_argument('--k', type=int, default=3, help="k for knn")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size should be samples * classes")
    parser.add_argument('--n_samples', type=int, default=2, help='number of samples from each src class')
    parser.add_argument('--sim_coeff', type=float, default=1.0, help="coeff for similarity loss")
    parser.add_argument('--only_da_iter', type=int, default=0,
                        help="number of iterations when only DA loss works and sim doesn't")
    parser.add_argument('--simi_func', type=str, default='euclidean', choices=['cosine', 'euclidean', "gaussian"])
    parser.add_argument('--method', type=str, nargs='?', default='DANN', choices=['DANN', 'CDAN', 'CDAN+E'])
    parser.add_argument('--knn_method', type=str, nargs='?', default='ranking', choices=['ranking', 'classic'])
    parser.add_argument('--ranking_k', type=int, default=12, help="use number of samples per class")
    parser.add_argument('--top_ranked_n', type=int, default=40,
                        help="these many target samples are used finally, 1/3 of batch")
    parser.add_argument('--n_sub_classes', type=int, default=10, help="number of classes sampled in batch")
    parser.add_argument('--queue_size', type=int, default=1124, help="size of queue")
    parser.add_argument('--momentum', type=float, default=0.999, help="momentum value")
    parser.add_argument('--tau', type=float, default=0.07, help="temperature value")
    parser.add_argument('--total_classes', type=int, default=31, help="total # classes in the dataset")
    parser.add_argument('--dataset', default="birds31", help="Name of the dataset")
    parser.add_argument('--test_10crop', action="store_true", help="10 crop testing")
    parser.add_argument('--data_dir', type=str, default="./data", help="Path for data directory")
    parser.add_argument('--multi_gpu', type=bool, default=0)
    # parser.add_argument('--n_selectLabels', type=int, default=-1, help="# of subset classes to train on. -1 for all classes.")
    # parser.add_argument('--selectType', type=str)
    # parser.add_argument('--labelLevel', default=0, type=int, choices=[0,1,2,3])

    args = parser.parse_args() 
    out_dir = os.path.join("snapshot" , args.dataset , args.out_dir )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "log.txt")
    log_acc = os.path.join(out_dir, "logAcc.txt")
    print("Writing log to" , out_file)
    out_file = open(out_file, "w")
    best_file = os.path.join(out_dir, "best.txt")
    args.multi_gpu = bool(args.multi_gpu)
    args.source_subset_sampler = bool(args.sim_net)

    ##### TensorBoard & Misc Setup #####
    writer_loc = os.path.join(out_dir , 'tensorboard_logs')
    writer = SummaryWriter(writer_loc)

    if args.source_subset_sampler:
        print('Using SIM NET')
    else:
        print('NOT using sim net')

    if args.dataset in ["birds123" , "cub2011" , "domainNet"]:
        if args.dataset == "domainNet":
            file_path = {
                "real": "./data/visDA_126/real.txt",
                "sketch": "./data/visDA_126/sketch.txt" ,
                "painting": "./data/visDA_126/painting.txt" ,
                "clipart": "./data/visDA_126/clipart.txt",
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
        print("Source: " , args.source)
        print("Target" , args.target)
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target]
    elif args.dataset == "domainNet_full":
        file_path = {
            "real": "./data/visDA_full/real_train.txt" ,
            "sketch": "./data/visDA_full/sketch_train.txt" ,
            "painting": "./data/visDA_full/painting_train.txt" ,
            "clipart": "./data/visDA_full/clipart_train.txt"}
        print("Source: " , args.source)
        print("Target" , args.target)
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target].replace("train" , "test")
    else:
        raise NotImplementedError

    batch_size = {"train": args.batch_size, "val": args.batch_size}
    
    # if args.n_selectLabels > 0:
    #     if args.selectType == "1":
    #         subsetClasses = list(range(args.n_selectLabels))
    #     elif args.selectType == "2":
    #         subsetClasses = list(range(args.total_classes-1 , args.total_classes - args.n_selectLabels-1, -1))
    #     elif args.selectType == "3":
    #         start = (args.total_classes - args.n_selectLabels)//2
    #         end = (args.total_classes + args.n_selectLabels)//2
    #         subsetClasses = list(range(start , end))
    #     elif args.selectType == "4":
    #         subsetClasses = sorted(np.random.choice(range(args.total_classes), args.n_selectLabels))
    #     else:
    #         raise NotImplementedError
    #     args.total_classes = len(subsetClasses)
    # else:
    #     subsetClasses = list(range(args.total_classes))
    subsetClasses = None# args.labelLevel
    # classdict = {0:200, 1:122, 2:38, 3:14}
    # args.total_classes = classdict[args.labelLevel]


    out_file.write('all args = {}\n'.format(args))
    out_file.flush()

    dataset_loaders = {}
    print(dataset_source)

    dataset_list = ImageList(args.data_dir, open(dataset_source).readlines(), subsetClasses=subsetClasses,
                             transform=prep.image_train(resize_size=256, crop_size=224))
    
    print(f"{len(dataset_list)} source samples")

    if args.source_subset_sampler:
        src_train_sampler = sampler.get_sampler({
            'path'              : dataset_source,
            'n_classes'         : args.total_classes,
            'n_samples'         : args.n_samples,
            'n_sub_classes'     : args.n_sub_classes,
        })
        dataset_loaders["source"] = torch.utils.data.DataLoader(dataset_list, batch_sampler=src_train_sampler, \
                                                               shuffle=False, num_workers=16)
    else:
        dataset_loaders["source"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'],
                                                               shuffle=True, num_workers=16,
                                                               drop_last=True)


    dataset_list = ImageList(args.data_dir, open(dataset_target).readlines(), subsetClasses=subsetClasses,
                             transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["target"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'], shuffle=True,
                                                         num_workers=16, drop_last=True)
    print(f"{len(dataset_list)} target samples")


    dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(), subsetClasses=subsetClasses,
                                transform=prep.image_test(resize_size=256, crop_size=224))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['val'], shuffle=False,
                                                          num_workers=16)
    print(f"{len(dataset_list)} target test samples")

    # network construction
    base_network = Encoder(args.total_classes)
    base_network = base_network.cuda()

    if args.method == 'DANN':
        my_discriminator = discriminatorDANN(256)
    elif args.method == 'CDAN':
        my_discriminator = discriminatorCDAN(256, args.total_classes)
    else:
        raise Exception('{} not implemented'.format(args.method))

    # domain discriminator
    # my_discriminator = my_discriminator.cuda()
    # my_discriminator.train(True)

    # gradient reversal layer
    # my_grl = ad.AdversarialLayer()
    import network
    ad_net = network.AdversarialNetwork(256* 200, 1024)
    ad_net = ad_net.cuda()

    # criterion and optimizer
    criterion = {
        "classifier" : nn.CrossEntropyLoss(),
        "adversarial": nn.BCELoss()
    }

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, base_network.model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, base_network.bottleneck_0.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, base_network.classifier_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, ad_net.parameters()), "lr": 1},
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

    best_acc = 0
    test_interval = 2000
    train_sim_loss = 0.0
    train_cls_loss = 0.0
    train_transfer_loss = 0.0
    train_entropy_loss_source = 0.0
    train_entropy_loss_target = 0.0
    train_total_loss = 0.0
    
    memory_network = queue_ila(256, K=args.queue_size, m=args.momentum, T=args.tau, knn=args.k, top_ranked_n=args.top_ranked_n, similarity_func=args.simi_func, batch_size=batch_size["train"], ranking_k=args.ranking_k)
    memory_network = memory_network.cuda()
    # memory_network.copy_params(base_network)

    len_source = len(dataset_loaders["source"]) - 1
    len_target = len(dataset_loaders["target"]) - 1
    iter_source = iter(dataset_loaders["source"])
    iter_target = iter(dataset_loaders["target"])

    with open(os.path.join(out_dir , "best.txt"), "w") as fh:
        fh.write("\nBest Accuracy file\n")
    
    for iter_num in range(1, args.max_iteration + 1):
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
        # domain_labels = torch.tensor([[1], ] * len(inputs_source)+ [[0], ] * len(inputs_target), device=torch.device('cuda:0'), dtype=torch.float)

        features, logits = base_network(inputs)
        logits_source = logits[:len(inputs_source)]
        classifier_loss = criterion["classifier"](logits_source, labels_source)

        # domain_predicted = my_discriminator(my_grl.apply(features), torch.softmax(logits, dim=1).detach())
        # transfer_loss = nn.BCELoss()(domain_predicted, domain_labels)
        softmax_out = nn.Softmax(dim=1)(logits)
        entropy = loss.Entropy(softmax_out)
        transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, calc_coeff(iter_num), None)
        total_loss = classifier_loss + transfer_loss

        sim_coeff = args.sim_coeff if iter_num > args.only_da_iter else 0.
        sim_loss= memory_network(features, labels_source)
        writer.add_scalar("Loss/sim_loss" , sim_loss.detach(), iter_num)

        # if iter_num > 50:
        #     labels_target = labels_target.float()
        #     psuedo_target_labels = psuedo_target_labels.float()
        #     correct_psuedo_labels = torch.mean(torch.eq(psuedo_target_labels, labels_target).float())
        #     writer.add_scalar("Psuedo/before_filter", correct_psuedo_labels.detach(), iter_num)

        #     psuedo_target_labels = psuedo_target_labels[target_indices]
        #     labels_target = labels_target[target_indices]
        #     correct_psuedo_labels = torch.mean(torch.eq(psuedo_target_labels, labels_target).float())
        #     writer.add_scalar("Psuedo/after_filter", correct_psuedo_labels.detach(), iter_num)

        total_loss += (sim_coeff * sim_loss)
          

        total_loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/classifier_loss" , classifier_loss.detach(), iter_num)
        writer.add_scalar("Loss/transfer" , transfer_loss.detach(), iter_num)
        
        # writer.add_scalar("Loss/psuedo_label_loss", target_classifier_loss.detach(), iter_num)

        # train_cls_loss += classifier_loss.item()
        # train_transfer_loss += transfer_loss.item()
        # train_sim_loss += sim_loss.item()      
        # train_total_loss += total_loss.item()

        # test
        test_interval = 2000
        if iter_num % test_interval == 0:
            base_network.eval()
            test_acc = test_target(dataset_loaders, base_network)
            writer.add_scalar("Acc/test" , test_acc , iter_num)
            print_str1 = '\niter: {:05d}, test_acc:{:.4f}\n'.format(iter_num, test_acc)
            print(print_str1)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = base_network.state_dict()
                with open(os.path.join(out_dir , "best.txt"), "a") as fh:
                    fh.write("\nBest Accuracy : {:.4f} at iter: {:05d}\n".format(best_acc, iter_num))
                torch.save(best_model , os.path.join(out_dir , "best_model.pth.tar"))

            # print_str2 = "(train loss)iter {:05d}, average classifier loss: {:.4f}; \t average transfer loss: {:.4f};" \
            #              " \t average entropy loss source: {:.4f}; \t average entropy loss target: {:.4f}; " \
            #              "\t average training loss: {:.4f}; \t average similarity loss: {:.4f};\n".format(
            #         iter_num,
            #         train_cls_loss / float(test_interval),
            #         train_transfer_loss / float(test_interval),
            #         train_entropy_loss_source / float(test_interval),
            #         train_entropy_loss_target / float(test_interval),
            #         train_total_loss / float(test_interval),
            #         train_sim_loss / float(test_interval))

            # out_file.write(print_str1)
            # out_file.flush()

            # out_file.write(print_str2)
            # out_file.flush()

            # train_cls_loss = 0.0
            # train_transfer_loss = 0.0
            # train_entropy_loss_source = 0.0
            # train_entropy_loss_target = 0.0
            # train_total_loss = 0.0
            # train_sim_loss = 0.0
