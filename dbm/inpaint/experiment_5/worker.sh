#!/usr/bin/env bash
echo "Generative phase"
train.py $1/inpaint.yaml
if [[ "$?" == 0 ]]
then
    echo "Discriminative phase"
    echo "When done running validate.yaml, I will make validate_best.txt" > $1/new_protocol.txt
    train.py $1/validate.yaml
    if [[ "$?" == 0 ]]
    then
	    echo "Training succeeded."
	    print_monitor.py $1/inpaint_best.pkl > $1/inpaint_best.txt
	    print_monitor.py $1/validate_best.pkl > $1/validate_best.txt
    else
	    echo "Discriminative training failed."
	    echo "Discriminative training failed." > $1/fail.txt
    fi
else
    echo "Generative training failed."
    echo "Generative training failed." > $1/fail.txt
fi
