#!/bin/bash

mkdir -p outputs/text-norm/
for i in results models logs predictions; do
	mkdir -p outputs/text-norm/${i}/char
	mkdir -p outputs/text-norm/${i}/words
done

cp submission.sub.template submission.sub

for type in char words; do
	for encoder in cfe conv brnn; do
		for seed in $(seq 1 5); do
			echo "arguments = ${type} ${encoder} ${seed}" >> submission.sub
			echo "queue" >> submission.sub
		done
	done
done