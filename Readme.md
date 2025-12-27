EvolveGCN
=====
This is a fork from the original EvolveGCN [repo](https://github.com/IBM/EvolveGCN.git).

This repository has been used as it implements the time-dependent methods used in the paper *Graph Neural Networks for IoT Security: A Comparative Study*, 

DOI: https://doi.org/10.1016/j.iot.2025.101863

# Setup
To setup the workspace (Conda env and Dataset) follow the instructions reported [here](https://github.com/MiviaLab/Graph-Neural-Networks-for-IoT-Security-A-Comparative-Study.git).

# Organization

The **experiments** folder contains the `.yaml` configuration files that define the parameters for each experiment and model.

For example:
- **experiments/60k_IoT23_etdg** contains the model configuration files used to run experiments with a 1M snapshot size and the ETDG representation.
- **parameters_egcn_h_anomaly_norm.yaml** contains the configuration file for the EGCN-H model.

For `.yaml` files, the **most important parameters** are:
- **folder**: path to the IoT23 split JSON file  
- **folder_iot_traces**: path to the IoT Traces test split JSON file  
- **folder_iot_id20**: path to the IoTID20 split JSON file  
- **graph_base_folder**: path to the IoT23 graphs directory  
- **graph_base_iot_traces_folder**: path to the IoT Traces graphs directory  
- **graph_base_iot_id20_folder**: path to the IoTID20 graphs directory  
- **normalize**: set to `True` to normalize node embeddings, `False` otherwise  
- **path_min_max_vectors**: path to the folder containing the `.npz` file with minimum and maximum values  
- **save_folder**: path where model checkpoints will be saved 
- **train**: True whether you want to train, False otherwise
- **test**: True whether you want to perform test at the end of training
- **off_line_test**: True whether you want to run test after trainining, otherwise False
- **compute_threshold**: True whether you want to compute the threshold for testing, False otherwise


**Important**: You can use the values in the existing files as a reference, but you must modify them according to your directory structure.

# How to run

```bash
# Train
# In .yaml file set: 
# train: True
# test: False
# off_line_test: False
# test_epoch: -1
# compute_threshold: False
cd bash
sbatch run_exp_evolve_[TIMESTAMP]K_[MODEL_NAME]_[REPRESENTATION].sh

# Test
# In .yaml file set: 
# train: False
# test: True
# off_line_test: True
# test_epoch: -1
# compute_threshold: True
cd bash
sbatch run_exp_evolve_[TIMESTAMP]K_[MODEL_NAME]_[REPRESENTATION].sh

```