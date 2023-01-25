CUDA_VISIBLE_DEVICES=0 python3 main.py \
--first_word soa_paper \
--batch_size 8 \
--number_of_epochs 20 \
--learning_rate 0.001 \
--check_point best --inference --resume  