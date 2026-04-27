#!/bin/bash

#====================================================
# Hyperparameter Grid Search Script
#====================================================
DATASET="benchpress"

# 迴圈設定：CFG 從 0 到 10，每次加 2, seq 0 5 30
for CFG in $(seq 10 10 30); do
    # 迴圈設定：STEP 從 100 到 300，每次加 50, 50 50 350
    for STEP in $(seq 100 50 300); do
        echo "=========================================================="
        echo "🚀 Running Inference & Evaluation -> CFG_Scale: $CFG, Total_Step: $STEP"
        echo "=========================================================="
        
        # 1. 執行推論 (Inference)
        python myinfer.py -d "$DATASET" --cfg_scale $CFG --total_step $STEP 
        
        # 2. 執行評估 (Evaluation)
        python myevaluation.py -d "$DATASET" --cfg_scale $CFG --total_step $STEP --method_list C-FID,NND
        
        echo "✅ Finished CFG: $CFG, STEP: $STEP"
        echo ""
    done
done
