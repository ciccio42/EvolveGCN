#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

BASE_PATH="/user/apaolillo/gnn-network-analysis/dynamic_graphs/EvolveGCN"

configurations=(
    
    "90k_IoT23_only_etdg"
)

cd "$BASE_PATH"
# Iterate over configurations

for ((i=0; i<${#configurations[@]}; i+=1)); do
    DATA_PATH="${configurations[i]}"
    
    echo "EGCN-O norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_egcn_o_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"
    
    echo "EGCN-H norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_egcn_h_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"

    echo "LSTM_A norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_lstmA_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"

    echo "GRU-A norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_gruA_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"

    echo "GCN norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_gcn_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"
done

cd bash