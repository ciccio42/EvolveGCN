#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

BASE_PATH="/user/apaolillo/dynamic_graphs/EvolveGCN/"

configurations=(
    
    "90k_IoT23_etdg"
    "90k_IoT23_tdg"
)

cd "$BASE_PATH"
# Iterate over configurations

for ((i=0; i<${#configurations[@]}; i+=1)); do
    DATA_PATH="${configurations[i]}"
    

    echo "GCN norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_gcn_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"
done

cd bash