EXPID=default
SR=40
CV=-1
RD=26


cd ./src/
python3 test.py \
 --exp_id ${EXPID} \
 --transformer_type fpn --rd_type clause --readout average \
 --window_pooling linear --level_pooling max \
 --windows_size 4 --head_num 8 --TF_depth 4 \
 --dataset random --min_n ${SR} --max_n ${SR} --cv_ratio ${CV} --n_pairs 10 \
 --spc --spc_kl \
 --binary_loss_weight 1 --clause_loss_weight 1 \
 --num_rounds ${RD} \
 --batch_size 8 --gpu --num_epochs 500 --lr 1e-4 --shuffle --lr_step 15 \
 --disable_core \
 --resume
