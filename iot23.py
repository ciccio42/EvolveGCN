import utils as u
import os
import torch
# erase
import time
import tarfile
import itertools
import numpy as np
from collections import OrderedDict
import glob
import pickle


class IoT23_Dataset():

    def __init__(self, args):
        args.iot23_args = u.Namespace(args.iot23_args)
        self.dataset_path = args.iot23_args.folder
        self.only_benign = args.iot23_args.only_benign
        self.representation = args.iot23_args.representation
        self.feats_per_node = args.iot23_args.feats_per_node
        self.graph_base_folder = args.iot23_args.graph_base_folder
        if args.iot23_args.one_class:
            self.num_classes = 1
        else:
            self.num_classes = 2

    def create_graph_list(self):
        capture_paths = glob.glob(f"{self.dataset_path}/CTU-Honeypot-*")

        for capture in capture_paths:
            print(f"Loading capture {capture}")
            capture_name = capture.split('/')[-1]
            self.graph_list[capture_name] = OrderedDict()
            self.node_ip_port_to_id[capture_name] = dict()
            self.node_id_to_ip_port[capture_name] = dict()

            representation_path = os.path.join(self.dataset_path,
                                               capture,
                                               f"{self.representation}_graph")
            for graph_type in os.listdir(representation_path):
                print(f"Loading graph type {graph_type.split('/')[-1]}")
                graph_type = graph_type.split('/')[-1]
                self.graph_list[capture_name][graph_type] = []
                self.node_ip_port_to_id[capture_name][graph_type] = dict()
                self.node_id_to_ip_port[capture_name][graph_type] = dict()
                if self.only_benign and "full_benign" in graph_type:
                    graph_pkl_files = glob.glob(
                        f"{representation_path}/{graph_type}/full_*.pkl")

                    # keeps track of the indices for each node
                    for graph_file in graph_pkl_files:
                        self.graph_list[capture_name][graph_type.split(
                            '/')[-1]].append(graph_file)


if __name__ == '__main__':
    pass
