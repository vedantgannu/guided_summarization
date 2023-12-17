#!/bin/sh
# bash train.sh ./highlighted_sentence_data_dir/cnndm ./model_dir ./log_file
DATA_PATH=$1
MODEL_PATH=$2
LOG_PATH=$3
# CHECK_POINT=$4

# Author's parameters 
# python z_train.py  -task abs -mode train -bert_data_path $DATA_PATH -dec_dropout 0.2  -model_path $MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file $LOG_PATH
# python z_train.py  -task abs -mode train -bert_data_path $DATA_PATH -dec_dropout 0.2  -model_path $MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file $LOG_PATH

# python z_train.py  -task abs -mode train -bert_data_path $DATA_PATH -dec_dropout 0.2  -model_path $MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 100 -batch_size 140 -train_steps 2000 -report_every 10 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 200 -warmup_steps_dec 100 -max_pos 512 -visible_gpus 0,1  -log_file $LOG_PATH
# python z_train.py  -task abs -mode train -bert_data_path $DATA_PATH -dec_dropout 0.2  -model_path $MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 100 -batch_size 140 -train_steps 2000 -report_every 10 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 200 -warmup_steps_dec 100 -max_pos 512 -visible_gpus 0,1  -log_file $LOG_PATH
# python z_train.py  -task abs -mode train -bert_data_path $DATA_PATH -dec_dropout 0.2  -model_path $MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 100 -batch_size 140 -train_steps 4000 -report_every 10 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 200 -warmup_steps_dec 100 -max_pos 512 -visible_gpus 0,1,2  -log_file $LOG_PATH -train_from $CHECK_POINT
python z_train.py  -task abs -mode train -bert_data_path $DATA_PATH -dec_dropout 0.2  -model_path $MODEL_PATH -sep_optim true -lr_bert 0.2 -lr_dec 0.2 -save_checkpoint_steps 100 -batch_size 200 -train_steps 5000 -report_every 10 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20 -warmup_steps_dec 10 -max_pos 512 -visible_gpus 0,1,2  -log_file $LOG_PATH -train_from $CHECK_POINT
# python z_train.py  -task abs -mode train -bert_data_path $DATA_PATH -dec_dropout 0.2  -model_path $MODEL_PATH -sep_optim true -lr_bert 0.7 -lr_dec 0.7 -save_checkpoint_steps 100 -batch_size 500 -train_steps 1000 -report_every 10 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20 -warmup_steps_dec 10 -max_pos 512 -visible_gpus 0,1,2  -log_file $LOG_PATH
