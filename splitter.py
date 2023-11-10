from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils as u
import node_cls_tasker as nct
import node_anomaly_tasker as nat
from collections import OrderedDict
from torch.utils.data import SubsetRandomSampler
import os
import json


class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''

    def __init__(self, args, tasker):

        if tasker.is_static:  # For static datsets
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            # only the training one requires special handling on start, the others are fine with the split IDX.

            random_perm = False
            indexes = tasker.data.nodes_with_label

            if random_perm:
                perm_idx = torch.randperm(indexes.size(0))
                perm_idx = indexes[perm_idx]
            else:
                print('tasker.data.nodes', indexes.size())
                perm_idx, _ = indexes.sort()
            # print ('perm_idx',perm_idx[:10])

            self.train_idx = perm_idx[:int(
                args.train_proportion*perm_idx.size(0))]
            self.dev_idx = perm_idx[int(args.train_proportion*perm_idx.size(0)): int(
                (args.train_proportion+args.dev_proportion)*perm_idx.size(0))]
            self.test_idx = perm_idx[int(
                (args.train_proportion+args.dev_proportion)*perm_idx.size(0)):]
            # print ('train,dev,test',self.train_idx.size(), self.dev_idx.size(), self.test_idx.size())

            train = static_data_split(tasker, self.train_idx, test=False)
            train = DataLoader(train, shuffle=True, **args.data_loading_params)

            dev = static_data_split(tasker, self.dev_idx, test=True)
            dev = DataLoader(dev, shuffle=False, **args.data_loading_params)

            test = static_data_split(tasker, self.test_idx, test=True)
            test = DataLoader(test, shuffle=False, **args.data_loading_params)

            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test

        else:  # For datsets with time
            if isinstance(tasker, nct.Node_Cls_Tasker):
                assert args.train_proportion + args.dev_proportion < 1, \
                    'there\'s no space for test samples'
                # only the training one requires special handling on start, the others are fine with the split IDX.
                start = tasker.data.min_time + args.num_hist_steps  # -1 + args.adj_mat_time_window
                end = args.train_proportion

                end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
                train = data_split(tasker, start, end, test=False)
                train = DataLoader(train, **args.data_loading_params)

                start = end
                end = args.dev_proportion + args.train_proportion
                end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
                if args.task == 'link_pred':
                    dev = data_split(tasker, start, end,
                                     test=True, all_edges=True)
                else:
                    dev = data_split(tasker, start, end, test=True)

                dev = DataLoader(
                    dev, num_workers=args.data_loading_params['num_workers'])

                start = end

                # the +1 is because I assume that max_time exists in the dataset
                end = int(tasker.max_time) + 1
                if args.task == 'link_pred':
                    test = data_split(tasker, start, end,
                                      test=True, all_edges=True)
                else:
                    test = data_split(tasker, start, end, test=True)

                test = DataLoader(
                    test, num_workers=args.data_loading_params['num_workers'])

                print('Dataset splits sizes:  train', len(
                    train), 'dev', len(dev), 'test', len(test))

            elif isinstance(tasker, nat.Anomaly_Detection_Tasker):
                # train, dev, test = self.anomaly_split(tasker, args)
                train = AnomalyDataset(
                    tasker=tasker,
                    path=tasker.data.dataset_path,
                    split='train')
                dev = AnomalyDataset(tasker=tasker,
                                     path=tasker.data.dataset_path,
                                     split='val')
                test = AnomalyDataset(tasker=tasker,
                                      path=tasker.data.dataset_path,
                                      split='test')

                train = DataLoader(
                    train, num_workers=args.data_loading_params['num_workers'])
                dev = DataLoader(
                    dev, num_workers=args.data_loading_params['num_workers'])
                test = DataLoader(
                    test, num_workers=args.data_loading_params['num_workers'])

                print('Dataset splits sizes:  train', len(
                    train), 'dev', len(dev), 'test', len(test))

            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test

    def anomaly_split(self, tasker, args):
        """_summary_

        Returns:
            _type_: _description_
        """
        # for each capture compute the max number of graphs
        dataset = tasker.data
        graph_list = dataset.graph_list
        train_graphs = OrderedDict()
        val_graphs = OrderedDict()
        for capture in graph_list.keys():
            train_graphs[capture] = OrderedDict()
            val_graphs[capture] = OrderedDict()
            for representation in graph_list[capture].keys():
                for graph_type in graph_list[capture].keys():
                    graph_number = len(
                        graph_list[capture][graph_type])
                    train_final_indx = int(graph_number*args.train_val)
                    train_graphs[capture][graph_type] = graph_list[capture][graph_type][:train_final_indx]
                    val_graphs[capture][graph_type] = graph_list[capture][graph_type][train_final_indx:]

        train_data = AnomalyDataset(graph_list=train_graphs)
        val_data = AnomalyDataset(graph_list=val_graphs)

        train_data, val_data, None


class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end-self.start

    def __getitem__(self, idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test=self.test, **self.kwargs)
        return t


class static_data_split(Dataset):
    def __init__(self, tasker, indexes, test):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.indexes = indexes
        self.test = test
        self.adj_matrix = tasker.adj_matrix

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return self.tasker.get_sample(idx, test=self.test)


class AnomalyDataset(Dataset):

    def __init__(self, tasker: nat.Anomaly_Detection_Tasker, path: str = None, split: str = 'train', representation: str = 'tdg_graph'):
        self.data_dict = None
        self.total_number_graphs = 0
        self.number_of_captures = 0
        self.indx_to_graph = OrderedDict()
        self.capture_start_end_indx = OrderedDict()
        self.tasker = tasker
        self.representation = representation
        self.mode = split
        if split == "train":
            self.data_dict_path = os.path.join(path, "train.json")
        elif split == "val":
            self.data_dict_path = os.path.join(path, "val.json")
        else:
            self.data_dict_path = os.path.join(path, "test_mixed.json")

        with open(self.data_dict_path) as data_file:
            self.data_dict = json.load(data_file)

        self.number_of_captures = len(list(self.data_dict.keys()))

        # create a map between graph path and index
        graph_indx = 0
        for capture in self.data_dict.keys():
            if "Honeypot" in capture:
                self.total_number_graphs += len(
                    self.data_dict[capture])
                start_indx = graph_indx
                for graph in self.data_dict[capture]:
                    self.indx_to_graph[graph_indx] = (capture, graph)
                    graph_indx += 1
                self.capture_start_end_indx[capture] = [
                    start_indx, graph_indx-1]

    def __len__(self):
        return self.total_number_graphs

    def __getitem__(self, idx):

        # map indx to capture
        capture = self.indx_to_graph[idx][0]
        start_indx, end_indx = self.capture_start_end_indx[capture]
        if self.mode == "train":
            graph_type = "full_benign"
        elif self.mode == "val":
            graph_type = "full_benign"
        elif self.mode == "test":
            graph_type = "mixed"
        t = self.tasker.get_sample(
            idx=idx,
            start_indx=start_indx,
            end_indx=end_indx,
            graph_list=self.indx_to_graph,
            capture_name=capture,
            graph_type=graph_type)
        return t
