cd ./src/
python3 train.py \
 --exp_id default \
 --transformer_type fpn --rd_type clause --readout average \
 --window_pooling linear --level_pooling max \
 --windows_size 4 --head_num 8 --TF_depth 4 \
 --dataset random --min_n 3 --max_n 10 --n_pairs 10000 \
 --spc --spc_kl \
 --binary_loss_weight 1 --clause_loss_weight 1 \
 --num_rounds 26 \
 --batch_size 8 --gpu --num_epochs 500 --lr 1e-4 --shuffle --lr_step 15 \
 --resume