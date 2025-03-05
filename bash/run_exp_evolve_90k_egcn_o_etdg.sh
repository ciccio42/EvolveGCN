#!/bin/bash

#SBATCH --exclude=tnode[01-17]
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL

BASE_PATH="/home/rsofnc000/Anomaly_Detection/anomaly_detection_code/gnn-network-analysis/dynamic_graphs/EvolveGCN"

configurations=(

    "90k_IoT23_etdg"
    #"90k_IoT23_tdg"
)

cd "$BASE_PATH"
# Iterate over configurations

for ((i = 0; i < ${#configurations[@]}; i += 1)); do
    DATA_PATH="${configurations[i]}"

    echo "EGCN-O norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_egcn_o_anomaly_norm.yaml"
    echo "$YAML_FILE"
    srun  python run_exp_anomaly.py --config_file "$YAML_FILE"

done

cd bash
