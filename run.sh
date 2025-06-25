export CUDA_VISIBLE_DEVICES=6

python inference.py --output_dir ./results \
    --model_finetune_path ckpt/unitime-full \
    --data_path data/test.json \
    --clip_length 32