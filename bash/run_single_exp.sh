#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

BASE_PATH="/user/frosa/anomaly_detection_code/dynamic_graphs/EvolveGCN"

cd "$BASE_PATH"
# Iterate over configurations

#
SNAP=120
# YAML_FILE="./experiments/${SNAP}k_IoT23_tdg/parameters_egcn_o_anomaly_norm.yaml"
# python run_exp_anomaly.py --config_file "$YAML_FILE"
YAML_FILE="./experiments/${SNAP}k_IoT23_etdg/parameters_egcn_o_anomaly_norm.yaml"
python run_exp_anomaly.py --config_file "$YAML_FILE"

cd bash
