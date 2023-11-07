import taskers_utils as tu
import torch
import utils as u
from collections import OrderedDict
import pickle
import numpy as np
import scipy.sparse as sp
import os


class Anomaly_Detection_Tasker():
    def __init__(self, args, dataset):
        self.data = dataset
        self.is_static = False
        self.feats_per_node = dataset.feats_per_node
        self.num_classes = dataset.num_classes
        self.num_hist_steps = args.num_hist_steps
        self.adj_mat_time_window = args.adj_mat_time_window
        self.prepare_node_feats = self.build_prepare_node_feats(args, dataset)

    def build_get_node_feats(self, args, dataset):
        pass

    def build_prepare_node_feats(self, args, dataset):
        if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
            def prepare_node_feats(node_feats):
                return u.sparse_prepare_tensor(node_feats,
                                               torch_size=[dataset.num_nodes,
                                                           self.feats_per_node])
        # elif args.use_1_hot_node_feats:

        else:
            def prepare_node_feats(node_feats):
                return node_feats[0]  # I'll have to check this up

        return prepare_node_feats

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def get_adj_matrix(self, graph, weight=False):
        adj_mat = graph['adj']
        adj_mat = torch.LongTensor(adj_mat)
        if not weight:
            vals = torch.ones(adj_mat.size(0), dtype=torch.long)

        out = torch.sparse.FloatTensor(
            adj_mat.t(), vals).coalesce()
        idx = out._indices().t()

        return {'idx': idx, 'vals': vals}

    def open_graph(self, capture_name, graph_type, graph_name):
        graph_path = os.path.join(
            self.data.graph_base_folder, capture_name, self.data.representation, graph_type, graph_name)
        with open(graph_path, 'rb') as f:
            # print(f"Considering graph {graph_path.split('/')[-1]}")
            graph = pickle.load(f)
        return graph

    def get_sample(self, idx, start_indx, end_indx, graph_list, capture_name, graph_type):

        hist_adj_list = []
        hist_ndFeats_list = []
        hist_mask_list = []
        hist_node_labels = []

        # check if there are at least self.adj_mat_time_window graphs
        if end_indx - start_indx < self.adj_mat_time_window:
            time_window = end_indx - start_indx
        else:
            time_window = self.adj_mat_time_window
        if (end_indx - idx) < time_window:
            start = idx - time_window
        else:
            start = idx

        for i in range(time_window):
            graph_data = self.open_graph(
                capture_name=capture_name,
                graph_type=graph_type,
                graph_name=graph_list[start+i][1])

            # 1. Create adj matrix
            # all edgess included from the beginning
            cur_adj = self.get_adj_matrix(graph=graph_data)

            # 2. Create node mask
            node_mask = tu.get_node_mask(cur_adj=cur_adj,
                                         num_nodes=graph_data['n_nodes'])

            # 3. Create node features
            node_feats = graph_data['node_features']

            # 4. Create node labels
            node_labels = self.get_node_labels(graph_data['node_labels'], idx)

            # 5. Normalize matrix
            cur_adj = tu.normalize_adj(
                adj=cur_adj,
                num_nodes=graph_data['n_nodes'])

            hist_adj_list.append(cur_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)
            hist_node_labels.append(node_labels)

        return {'idx': idx,
                'hist_adj_list': hist_adj_list,
                'hist_ndFeats_list': hist_ndFeats_list,
                'label_sp': hist_node_labels,
                'node_mask_list': hist_mask_list,
                'n_nodes': graph_data['n_nodes']}

    def get_node_labels(self, labels, idx):
        labels = torch.LongTensor(labels)
        label_idx = labels[:, 0]
        label_vals = labels[:, 1]
        return {'idx': label_idx,
                'vals': label_vals}


if __name__ == '__main__':
    fraud_times = torch.tensor([10, 5, 3, 6, 7, -1, -1])
    idx = 6
    non_fraudulent = ((fraud_times > idx) + (fraud_times == -1)) > 0
    print(non_fraudulent)
    exit()