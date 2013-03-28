#!/usr/bin/env bash
echo "When done running validate.yaml, I will make validate_best.txt" > $1/new_protocol.txt
train.py $1/validate.yaml
if [[ "$?" == 0 ]]
then
    print_monitor.py $1/validate_best.pkl > $1/validate_best.txt
    echo "Training succeeded."
else
    echo "Training failed."
    echo "Training failed." > $1/fail.txt
fi
