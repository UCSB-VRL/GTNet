CUDA_VISIBLE_DEVICES=0 python2 main.py \
--first_word soa_vcoco \
--batch_size 8 \
--number_of_epochs 20 \
--learning_rate 0.001 \
--check_point gtnet2best --inference --resume  