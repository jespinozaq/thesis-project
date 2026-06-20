#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

rm -rf packaged_results_temp
mkdir -p packaged_results_temp

DATASET="All_Beauty"

for csv_file in results_A.csv results_B_partial.csv swap_experiment_results.csv; do
    if [ -f "$csv_file" ]; then
        echo "Processing $csv_file ..."
        if head -1 "$csv_file" | grep -q "Dataset"; then
            awk -v ds="$DATASET" 'BEGIN{FS=OFS=","} NR==1 {print $0} NR>1 { $(NF)=ds; print }' "$csv_file" > "packaged_results_temp/$csv_file"
        else
            awk -v ds="$DATASET" 'BEGIN{FS=OFS=","} NR==1 {print $0, "Dataset"} NR>1 {print $0, ds}' "$csv_file" > "packaged_results_temp/$csv_file"
        fi
        echo "  -> packaged_results_temp/$csv_file saved."
    else
        echo "Skipping $csv_file (not found)"
    fi
done

ZIPNAME="beauty_results_temp_$(date +%Y%m%d_%H%M%S).zip"
zip -j "$ZIPNAME" packaged_results_temp/*.csv
echo ""
echo "Temporary package ready: $ZIPNAME"
echo "Download it with SCP (see instructions)."
