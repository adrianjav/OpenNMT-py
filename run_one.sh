#!/bin/bash

type=$1; shift
encoder=$1; shift
seed=$1; shift

echo "Training type ${type} encoder ${encoder} seed ${seed}"

onmt_train 	-seed ${seed} -data data/text-norm-${type}/model -save_model outputs/text-norm/models/${type}/model-${encoder}-seed-${seed} \
			-encoder_type ${encoder} -optim adam -adam_beta2 0.999 \
			-learning_rate 2.0 -world_size 1 -gpu_ranks 0 -batch_size 32 -input_feed 0 \
			--keep_checkpoint 1 --train_steps 100 -global_attention dot -layers 3 -rnn_size 256 \
			-cnn_kernel_width 11 -receptive_field 2 -param_init 0.5 -normalization tokens \
			-max_grad_norm 21 -dropout 0.05 -learning_rate_decay 0.847 -decay_method noam > outputs/text-norm/logs/${type}/${encoder}-seed-${seed}.txt 2>&1 

echo "Testing type ${type} encoder ${encoder} seed ${seed}"
onmt_translate 	-gpu 0 -batch_size 32 -beam_size 10 \
				-model outputs/text-norm/models/${type}/model-${encoder}-seed-${seed}* \
				-src data/text-norm-${type}/test.txt.src \
				-tgt data/text-norm-${type}/test.txt.tgt \
				-out outputs/text-norm/predictions/${type}/${encoder}-seed-${seed}.txt \
				-share_vocab > outputs/text-norm/results/${type}/${encoder}-seed-${seed}.txt