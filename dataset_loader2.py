import os
import math
import torch
import random
import argparse
import numpy as np

from sklearn.preprocessing import normalize
from torch.utils.data import ConcatDataset, DataLoader

random.seed(42)

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

class UnimodalDataset():
    def __init__(self, node_id, modality_str):

        self.folder_path = "../alzheimer/node_{}/".format(node_id)
        self.modality = modality_str
        self.data_path = self.folder_path + "{}/".format(self.modality)

        y = np.load(self.folder_path + "label.npy")

        self.labels = y.tolist()
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = np.load(self.data_path + "{}.npy".format(idx))

        if modality_str == "audio":
            x = normalize(x)

        self.data = x.tolist()
        self.data = torch.tensor(self.data)

        sensor_data = self.data
        if self.modality == 'depth':
            sensor_data = torch.unsqueeze(sensor_data, 0)

        activity_label = self.labels[idx]

        return sensor_data, activity_label

class MultimodalDataset():
    def __init__(self, node_id):

        self.folder_path = "../../alzheimer/node_{}/".format(node_id)
        y = np.load(self.folder_path + "label.npy")

        self.labels = y.tolist()
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x1 = np.load(self.folder_path + "audio/" + "{}.npy".format(idx))
        x2 = np.load(self.folder_path + "depth/" + "{}.npy".format(idx))
        x3 = np.load(self.folder_path + "radar/" + "{}.npy".format(idx))

        x1 = normalize(x1)

        # print(np.shape(x2))
        v_min = x2.min(axis=(1, 2), keepdims=True)
        v_max = x2.max(axis=(1, 2), keepdims=True)
        
        x2 = (x2-v_min)/(v_max - v_min+1)

        self.data1 = x1.tolist() #concate and tolist
        self.data2 = x2.tolist()
        self.data3 = x3.tolist()
        
        sensor_data1 = torch.tensor(self.data1)
        sensor_data2 = torch.tensor(self.data2)
        sensor_data3 = torch.tensor(self.data3)

        # sensor_data2 = torch.unsqueeze(sensor_data2, 0)

        activity_label = self.labels[idx]

        return (sensor_data1, sensor_data2, sensor_data3), activity_label

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--usr_id', type=int, default=16,
                        help='user id')
    parser.add_argument('--local_modality', type=str, default='audio',
                        choices=['audio', 'depth', 'radar', 'all', 'AD', 'DR', 'AR'], help='local_modality')
    parser.add_argument('--fl_epoch', type=int, default=10,
                    help='communication to server after the epoch of local training')
    parser.add_argument('--incomplete_flag', type=int, default=1,
                        help='incomplete_flag')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='MyMMmodel')
    parser.add_argument('--dataset', type=str, default='AD',
                        choices=['MHAD', 'FLASH', 'AD'], help='dataset')
    parser.add_argument('--num_class', type=int, default=11,
                        help='num_class')
    parser.add_argument('--num_of_train', type=int, default=50,
                        help='num_of_train')
    parser.add_argument('--num_of_test', type=int, default=200,
                        help='num_of_test')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--dim_enc_0', type=int, default = 1496256)
    parser.add_argument('--dim_enc_1', type=int, default = 2221056)
    parser.add_argument('--dim_enc_2', type=int, default = 626240)
    parser.add_argument('--dim_enc_multi', type=int, default = 4343552)
    parser.add_argument('--dim_cls_0', type=int, default = 2651)
    parser.add_argument('--dim_cls_1', type=int, default = 2827)
    parser.add_argument('--dim_cls_2', type=int, default = 3531)
    parser.add_argument('--dim_cls_multi', type=int, default = 8987)
    parser.add_argument('--dim_all_0', type=int, default = 1498907)
    parser.add_argument('--dim_all_1', type=int, default = 2223883)
    parser.add_argument('--dim_all_2', type=int, default = 629771)
    parser.add_argument('--dim_all_multi', type=int, default=4352539)

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.incomplete_flag == 1:
        opt.model_path = './save_uniFL/{}_models/'.format(opt.dataset)
        opt.result_path = './save_fedfuse_results/node_{}/{}_results/'.format(opt.usr_id, opt.local_modality)

    opt.load_folder = os.path.join(opt.model_path)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt

def load_datasets(opt, client_num, batch_size, train_test_ratio): # data is the dataset

    # load labeled train and test data
    print("Loading Dataset...")

    dataset_list = []
    if opt.usr_id == 16:
        for idx in range(16):
            tmp_dataset = MultimodalDataset(idx)
            # print(np.shape(tmp_dataset[0][0][0]))
            dataset_list.append(tmp_dataset)
        total_dataset = ConcatDataset(dataset_list)
    else:
        if opt.local_modality in ["all", 'AD', 'DR', 'AR']: # three or two modalities 
            total_dataset = MultimodalDataset(opt.usr_id)
        else:
            total_dataset = UnimodalDataset(opt.usr_id, opt.local_modality) # one modality
    
    dataset_len = len(total_dataset)

    """
        sample_index_path = "./sample_index/node_{}/".format(opt.usr_id)
        all_train_sample_index = np.loadtxt(sample_index_path + "train_sample_index.txt")
        all_test_sample_index = np.loadtxt(sample_index_path + "test_sample_index.txt")
        # all_incomplete_sample_index = np.loadtxt(sample_index_path + "incomplete_sample_index.txt")

        if len(all_train_sample_index) < opt.num_of_train:    
            opt.num_of_train = len(all_train_sample_index)
        if len(all_test_sample_index) < opt.num_of_test:
            opt.num_of_test = len(all_test_sample_index)

        train_sample_index = all_train_sample_index[0:opt.num_of_train]
        test_sample_index = all_test_sample_index[0:opt.num_of_test]
        print(len(train_sample_index))
        print(len(test_sample_index))

        train_sample_index = list(map(int, train_sample_index))
        test_sample_index = list(map(int, test_sample_index))
    """

    # here the train_sample_index control which samples are used in the training and testing
    # for each node, each modality, only 678 samples
    # replace the above code

    trainset, testset = random_split(total_dataset, [train_test_ratio, 1-train_test_ratio], torch.Generator().manual_seed(42))

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // client_num
    lengths = [partition_size] * client_num
    trainset, _ = random_split(trainset, [partition_size*client_num, len(trainset)-partition_size*client_num], torch.Generator().manual_seed(42))
    # print(lengths, len(trainset))
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    server_trainset, _ = random_split(trainset, [0.1, 0.9], torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []

    for ds in datasets:
        len_val = len(ds) // 10  # 10% validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        # print(len(ds_train), len(ds_val), len(ds))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    server_trainloader = DataLoader(server_trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader, server_trainloader
