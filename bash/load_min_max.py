import pickle as pk
import os
import numpy as np

MIN_MAX_BASE_PATH = "/home/rsofnc000/Anomaly_Detection/anomaly_detection_dataset/min_max_benign"

snap_list = ['60k'] #['150k', '120k', '90k', '60k']

dataset_min_max = "IoT23_min_max_benign"

representation_min_max_list = ["min_max_etdg_graph", "min_max_tdg_graph"]

if __name__ == '__main__':

    for snap in snap_list:
        print("Snap: ", snap)
        for representation_min_max in representation_min_max_list:
            
            min_vector_etdg = np.load(os.path.join(MIN_MAX_BASE_PATH, snap, dataset_min_max, "min_max_etdg_graph", "min.npz"))['arr_0'][:57]
            # print(min_vector_etdg)
            max_vector_etdg = np.load(os.path.join(MIN_MAX_BASE_PATH, snap, dataset_min_max, "min_max_etdg_graph", "max.npz"))['arr_0'][:57]
            # print(max_vector_etdg)
            
            min_vector_tdg = np.load(os.path.join(MIN_MAX_BASE_PATH, snap, dataset_min_max, "min_max_tdg_graph", "min.npz"))['arr_0'][:57]
            # print(min_vector_tdg)
            max_vector_tdg = np.load(os.path.join(MIN_MAX_BASE_PATH, snap, dataset_min_max, "min_max_tdg_graph", "max.npz"))['arr_0'][:57]
            # print(max_vector_tdg)
            
            min_difference = min_vector_etdg - min_vector_tdg
            # cnt number of elements that are not zero
            cnt = np.count_nonzero(min_difference)
            print("\t Min Number of elements that are not zero: ", cnt)
            
            non_zero_indices = np.nonzero(min_difference)
            print("\t Non zero elements: ", min_difference[non_zero_indices])
            
            max_difference = max_vector_etdg - max_vector_tdg
            # cnt number of elements that are not zero
            cnt = np.count_nonzero(max_difference)
            print("\t Max Number of elements that are not zero: ", cnt)
            
            non_zero_indices = np.nonzero(max_difference)
            print("\t Non zero elements: ", max_difference[non_zero_indices])