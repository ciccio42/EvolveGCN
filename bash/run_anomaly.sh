#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
cd ../
python run_exp_anomaly.py --config_file ./experiments/parameters_elliptic_egcn_o_anomaly.yaml --debug

cd bash