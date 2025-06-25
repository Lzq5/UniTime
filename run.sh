export CUDA_VISIBLE_DEVICES=7

python inference.py --model_local_path  path_to_qwen2vl7B \
    --model_finetune_path ckpt/unitime-full \
    --data_path data/test.json