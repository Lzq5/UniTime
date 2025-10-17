# export CUDA_VISIBLE_DEVICES=0,1,2,3
export DECORD_EOF_RETRY_MAX=20480

python inference.py --model_local_path path_to_qwen2vl7B \
    --model_finetune_path ./checkpoints/RUN_NAME \
    --video_root path_to_video_root \
    --feat_folder path_to_feat_folder \
    --data_path path_to_test_data \
    --output_dir ./results/RUN_NAME \
    --nf_short 128