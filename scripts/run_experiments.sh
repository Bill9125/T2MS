#!/bin/bash

# ==============================================================================
# T2S (Text-to-Sequence) Model Evaluation Pipeline
# 
# 這個腳本整合了論文研究中的四大核心實驗。
# 執行前請確保您的環境已經安裝所有相依套件，並設定好資料夾位址。
# ==============================================================================

DATASET="deadlift"

# # # ------------------------------------------------------------------------------
# # # 🧪 實驗一 & 二：超參數二維搜尋 (Grid Search) & 條件生成評估 (C-FID / LDS)
# # # 目的：掃描 CFG 與 Step 的組合，並計算每一組資料的 CFID 與 LDS 分數，並產出熱力圖
# # # ------------------------------------------------------------------------------
# echo "🚀 啟動實驗一 & 二：超參數掃描與 C-FID/LDS 分析"

# # 定義要掃描的參數範圍 (依據需求可調整，因為 Drop Rate 降至 10%，CFG 通常在 1~10 之間最佳)
for CFG in 1 3 5 7 10; do
    for STEP in 50 100 200; do
        echo "----------------------------------------------------------"
        echo "Grid Search -> CFG_Scale: $CFG, Total_Step: $STEP"
        echo "----------------------------------------------------------"
        
        # 1. 執行並行/隨機生成 (預設 run_time=10)
        python myinfer.py -d "$DATASET" --cfg_scale $CFG --total_step $STEP --run_time 10 --subject isolated --caption explain
        
        # 2. 進行 C-FID 與 NND 的計算 (預設 run_time=10)，會自動把結果合併繪製成 Heatmap
        python myevaluation.py -d "$DATASET" --cfg_scale $CFG --total_step $STEP --n_folds 5 --subject isolated --method_list C-FID,LDS --caption explain
    done
done
echo "✅ 實驗一 & 二 完成！ (可至 ./heatmaps/ 檢查結果)"
echo ""

# ------------------------------------------------------------------------------
# 🧪 實驗三：條件生成的多樣性消融驗證 (Fixed vs Random Noise)
# 目的：驗證 Diffusion 從雜訊 $z$ 中汲取多樣性，避免 Mode Collapse。
# 這裡我們挑定從「實驗一」中找出的最佳參數來跑對照組，假設最佳參數為 CFG=5, STEP=250
# ------------------------------------------------------------------------------
# BEST_CFG=3
# BEST_STEP=100

# echo "🚀 啟動實驗三： Fixed vs Random Noise 多樣性消融測試"
# echo "使用最佳參數進行對照：CFG = $BEST_CFG, Step = $BEST_STEP"

# python myinfer.py -d "$DATASET" --cfg_scale $BEST_CFG --total_step $BEST_STEP --subject mix --caption explain --run_time 20

# # [對照組：Fixed Noise] 鎖死初始隨機種子，沒有自然多樣性，理論上 C-FID 應該會變高。
# echo ">> 正在生成 Fixed Noise 實驗組 (run_time=1)..."
# python myinfer.py -d "$DATASET" --cfg_scale $BEST_CFG --total_step $BEST_STEP --subject mix --caption explain --fixed_noise --run_time 20

# echo ">> 正在計算 Fixed Noise C-FID 分數..."
# python myevaluation.py -d "$DATASET" --cfg_scale $BEST_CFG --total_step $BEST_STEP --subject mix --caption explain --fixed_noise --n_folds 5


# echo ""
# echo "🎉 所有實驗腳本執行完畢！"
# echo "您現在可以到 ./results/denoiser_results/evaluation_random 與 evaluation_fixed 比較最佳參數的 JSON C-FID 數值差異了！"

# ------------------------------------------------------------------------------
# 🧪 實驗四：全新 Caption 推論與 t-SNE 視覺化分佈
# 目的：使用 style_new 作為全新的文字提示生成動作，並繪製 t-SNE 以檢查 Training, Test, Generated Set 的多樣性與對齊情況。
# ------------------------------------------------------------------------------
# BEST_CFG=3
# BEST_STEP=100

# echo "🚀 啟動實驗四：推論全新 Caption (style_new) 並進行 t-SNE 視覺化"
# echo ">> 正在生成 style_new 實驗組..."
# python myinfer.py -d "$DATASET" --cfg_scale $BEST_CFG --total_step $BEST_STEP --subject mix --caption style_new --run_time 20

# echo ">> 正在進行 t-SNE 分析 (每類最多抽樣 2000 筆)..."
# python myevaluation.py -d "$DATASET" --cfg_scale $BEST_CFG --total_step $BEST_STEP --subject mix --caption style_new --method_list TSNE

# echo "✅ 實驗四 完成！ (可至 ./heatmaps/ 檢查 t-SNE 繪圖結果)"
# echo ""
# echo "🎉 所有實驗腳本執行完畢！"
