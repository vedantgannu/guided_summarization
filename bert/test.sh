#!/bin/sh  
# Example command   
# bash test.sh ./highlighted_sentence_data_dir_experiment/oracle_test/cnndm ./highlighted_sentence_data_dir_experiment/model_dir/model_step_2000.pt ./highlighted_sentence_data_dir_experiment/results
BERT_DATA_PATH=$1
MODEL_PATH=$2
RESULT_PATH=$3

# python -W ignore z_train.py \
#     -task abs \
#     -mode test \
#     -batch_size 3000 \
#     -test_batch_size 1500 \
#     -bert_data_path $BERT_DATA_PATH \
#     -log_file logs/test.logs \
#     -test_from $MODEL_PATH \
#     -sep_optim true \
#     -use_interval true \
#     -visible_gpus 0 \
#     -max_pos 512 \
#     -max_length 200 \
#     -alpha 0.95 \
#     -min_length 50 \
#     -result_path $RESULT_PATH
python -W ignore z_train.py \
    -task abs \
    -mode test \
    -batch_size 3000 \
    -test_batch_size 1500 \
    -bert_data_path $BERT_DATA_PATH \
    -log_file ./log_file \
    -test_from $MODEL_PATH \
    -sep_optim true \
    -use_interval true \
    -visible_gpus 0 \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length 50 \
    -result_path $RESULT_PATH
