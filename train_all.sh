#!/bin/sh

training_data='data/all_underscored_sent.txt'
eval_file='data/questions-words.txt'
pretrain_model='models/GoogleNews-vectors-negative300.bin'

for window in 3 4 5; do
	for dim in 300; do
		for e in 5 10; do
			for alpha in 0.025; do
    	    	echo "python train.py -i ${training_data} -o models/finetuned_${dim}d_w${window}_a${alpha}_${e}iter -a ${alpha} -e ${e} -d ${dim} -w ${window} -p ${pretrain_model}"
       			python train.py -i ${training_data} -o models/hyphened/finetuned_${dim}d_w${window}_a${alpha}_${e}iter -a ${alpha} -e ${e} -d ${dim} -w ${window} -p ${pretrain_model}
			done
    	done
	done
done
