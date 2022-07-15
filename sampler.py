import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch
import csv

class SelectedBalancedBatchSampler(BatchSampler):
    def __init__(self, file_path, n_total_classes, n_samples, n_sub_classes):
        self.n_extend = 2
        self.labels_list = []
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                self.labels_list.append(int(row[1]))

        self.labels = np.array(self.labels_list)
        self.labels_set = list(set(self.labels))
        assert len(self.labels_set) == n_total_classes
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set} #get indices for each class from the csv file

        ####Shuffle and extend
        #extend: #the list of indices are repeated twice in the list for each pointer move and including all the samples
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
            temp_arr = self.label_to_indices[l].copy()
            for ci in range(self.n_extend - 1):
                self.label_to_indices[l] = np.append(self.label_to_indices[l], temp_arr)

        self.used_label_indices_count = {label: 0 for label in self.labels_set} #acts as pointer for tracking used indices
        self.count = 0
        self.n_total_classes = n_total_classes
        self.n_sub_classes = n_sub_classes #number of subset of classes in a batch
        self.n_samples = n_samples #samples per class
        self.len_dataset = len(self.labels_list)
        self.batch_size = self.n_samples * self.n_sub_classes
        self.all_classes = [i for i in range(0, self.n_total_classes)]

        np.random.shuffle(self.all_classes)
        self.idx_class = 0

    def __iter__(self):
        self.count = 0  #to track if dataset has been explored
        while self.count + self.batch_size < self.len_dataset:
            if self.idx_class + self.n_sub_classes > self.n_total_classes: #acts as pointer for classses
                np.random.shuffle(self.all_classes)
                self.idx_class = 0
            classes = self.all_classes[self.idx_class: self.idx_class+self.n_sub_classes] #take subset of classes for a batch
            self.idx_class = self.idx_class + self.n_sub_classes
            indices = [] #collects the batch
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples #move pointer for a particular class

                # reset pointer if all indices visited for that class
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    indices_set = list(set(self.label_to_indices[class_].copy()))
                    np.random.shuffle(indices_set)
                    for ci in range(self.n_extend - 1):
                        self.label_to_indices[class_] = np.append(self.label_to_indices[class_], indices_set)
                    self.used_label_indices_count[class_] = 0

            assert len(indices) % self.n_samples == 0, 'yielded indices are not in multiple of n_samples'
            assert len(indices) % self.n_sub_classes == 0, 'yielded indices in batch-sampler are not in multiple of n_sub_classes'
            assert len(set(classes)) == self.n_sub_classes, 'batch doesnt contain all subset classes requested'
            self.count += self.n_sub_classes * self.n_samples
            np.random.shuffle(indices)
            yield indices

    def __len__(self):
        return self.len_dataset // self.batch_size

def get_sampler(config_data):
    # only used for office as of now, have same train and eval files for src and tgt
    file_src_train = config_data['path']
    n_classes = config_data['n_classes']
    n_samples = config_data['n_samples']
    n_sub_classes = config_data['n_sub_classes']
    print("file_src: %s \t n_classes: %d \t n_samples_per_class: %d \t n_sub_classes_per_iter: %d"%(
                    file_src_train, n_classes, n_samples, n_sub_classes))

    sampler_src_train = SelectedBalancedBatchSampler(file_src_train, n_classes, n_samples, n_sub_classes)

    return sampler_src_train


