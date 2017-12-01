#!/bin/bash
folder=$1

if [ -f $folder ]; then
	echo "Usage: ../evaluate_scripts/evaluate_logs_folder <folder_path>"
	exit
fi

for epoch in $(ls $folder/*.json | cut -d'_' -f4 | cut -d'.' -f1 | sort -n); do
	echo -n "Epoch $epoch: ";
	python ../evaluate_scripts/evaluate-v1.1.py ../../data/dev-v1.1.json $folder/dev_predictions_${epoch}.json
done
