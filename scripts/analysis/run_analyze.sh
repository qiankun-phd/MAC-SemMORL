#!/bin/bash
set -e
PYTHON=${COLA_PYTHON:-/home/qiankun/.conda/envs/RA_DI/bin/python}
cd /home/qiankun/CommRL/COLA
$PYTHON experiments/analysis/analyze_results.py > /home/qiankun/CommRL/COLA/analysis_output.txt 2>&1
echo "=== Process Status ===" >> /home/qiankun/CommRL/COLA/analysis_output.txt
ps aux | grep main_uav | grep python | grep -v grep | wc -l >> /home/qiankun/CommRL/COLA/analysis_output.txt
echo "=== GPU ===" >> /home/qiankun/CommRL/COLA/analysis_output.txt
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> /home/qiankun/CommRL/COLA/analysis_output.txt 2>&1
