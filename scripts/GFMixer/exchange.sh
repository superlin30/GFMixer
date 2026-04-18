if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=720
model_name=GFMixer

root_path_name=./dataset/
data_path_name=exchange_rate.csv
model_id_name=exchange
data_name=custom
random_seed=2021


for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.25 \
    --fc_dropout 0.15 \
    --kernel_list 3 7 11 \
    --period 24\
    --patch_len 1\
    --stride 1\
    --des Exp \
    --pct_start 0.2 \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

