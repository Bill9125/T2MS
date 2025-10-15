CFG_SCALE=(3 7 10)
TOTAL_STEP=(100 1000 10000)
for i in "${!CFG_SCALE[@]}"; do
    for j in "${!TOTAL_STEP[@]}"; do
        python myinfer.py --cfg_scale "${CFG_SCALE[$i]}" --total_step "${TOTAL_STEP[$j]}" --run_time 10
    done
done