#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd ../
#---- EGCN-O ----#

python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_o_anomaly.yaml 

# 

#---- LSTM ----#
# python run_exp_anomaly.py --config_file ./experiments/parameters_lstm_anomaly.yaml 


cd bash