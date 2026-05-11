#!/bin/bash

# ==============================================================================
# T2S (Text-to-Sequence) Model Evaluation Pipeline
# 
# 這個腳本整合了論文研究中的四大核心實驗。
# 執行前請確保您的環境已經安裝所有相依套件，並設定好資料夾位址。
# ==============================================================================

DATASET="benchpress"

# # ------------------------------------------------------------------------------
# # 🧪 實驗一 & 二：超參數二維搜尋 (Grid Search) & 條件生成評估 (C-FID / NND)
# # 目的：掃描 CFG 與 Step 的組合，並計算每一組資料的 CFID 與 NND 分數，並產出熱力圖
# # ------------------------------------------------------------------------------
echo "🚀 啟動實驗一 & 二：超參數掃描與 C-FID/NND 分析"

# 定義要掃描的參數範圍 (依據需求可調整，因為 Drop Rate 降至 10%，CFG 通常在 1~10 之間最佳)
for CFG in 1 3 5 7 10; do
    for STEP in 50 100 200; do
        echo "----------------------------------------------------------"
        echo "Grid Search -> CFG_Scale: $CFG, Total_Step: $STEP"
        echo "----------------------------------------------------------"
        
        # 1. 執行並行/隨機生成 (預設 run_time=10)
        python myinfer.py -d "$DATASET" --cfg_scale $CFG --total_step $STEP --run_time 10
        
        # 2. 進行 C-FID 與 NND 的計算 (預設 run_time=10)，會自動把結果合併繪製成 Heatmap
        python myevaluation.py -d "$DATASET" --cfg_scale $CFG --total_step $STEP --n_folds 5
    done
done
echo "✅ 實驗一 & 二 完成！ (可至 ./heatmaps/ 檢查結果)"
echo ""


# ------------------------------------------------------------------------------
# 🧪 實驗三：模型創造力與抗死背能力 (Novelty Score)
# 目的：證明產出並非死背。
# *這部分已經包含在實驗二的 python myevaluation.py 產生的 .json 與 Heatmap 中
# (指標為: Novelty-Score (Gen) vs Novelty-Score (Test))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 🧪 實驗四：條件生成的多樣性消融驗證 (Fixed vs Random Noise)
# 目的：驗證 Diffusion 從雜訊 $z$ 中汲取多樣性，避免 Mode Collapse。
# 這裡我們挑定從「實驗一」中找出的最佳參數來跑對照組，假設最佳參數為 CFG=5, STEP=250
# ------------------------------------------------------------------------------
# BEST_CFG=3
# BEST_STEP=100

# echo "🚀 啟動實驗四： Fixed vs Random Noise 多樣性消融測試"
# echo "使用最佳參數進行對照：CFG = $BEST_CFG, Step = $BEST_STEP"

# # [對照組：Fixed Noise] 鎖死初始隨機種子，沒有自然多樣性，理論上 C-FID 應該會變高。
# # 為了節省算力，固定雜訊只跑 1 次 (run_time 1) 即可。
# echo ">> 正在生成 Fixed Noise 實驗組 (run_time=1)..."
# python myinfer.py -d "$DATASET" --cfg_scale $BEST_CFG --total_step $BEST_STEP --fixed_noise --run_time 10

# echo ">> 正在計算 Fixed Noise C-FID 分數..."
# python myevaluation.py -d "$DATASET" --cfg_scale $BEST_CFG --total_step $BEST_STEP --fixed_noise --n_folds 5

# echo ""
# echo "🎉 所有實驗腳本執行完畢！"
# echo "您現在可以到 ./results/denoiser_results/evaluation_random 與 evaluation_fixed 比較最佳參數的 JSON C-FID 數值差異了！"
