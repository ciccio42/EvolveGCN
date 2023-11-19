#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

cd ../
echo "EGCNO norm"
python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_o_anomaly_norm.yaml
echo "EGCNO no norm"
python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_o_anomaly_no_norm.yaml
echo "LSTM norm"
python run_exp_anomaly.py --config_file ./experiments/parameters_lstm_anomaly_norm.yaml
echo "LSTM no norm"
python run_exp_anomaly.py --config_file ./experiments/parameters_lstm_anomaly_no_norm.yaml

cd bash