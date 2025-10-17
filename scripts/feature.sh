data_paths=(
  "path_to_train_data"
  "path_to_val_data"
  "path_to_test_data"
)

export DECORD_EOF_RETRY_MAX=20480
gpu_list=(4 5 6 7)
part_list=(0 1 2 3)
model_local_path=path_to_qwen2vl7B [ToModify]
feat_root=path_to_feature_root [ToModify]
video_root=path_to_video_root [ToModify]
# You can adjust the parallelism by modifying the num_parts parameter, and update gpu_list and part_list to assign GPUs to each process
for data_path in "${data_paths[@]}"; do
  for i in ${!gpu_list[@]}; do
    python feature_offline.py --data_path $data_path --part ${part_list[$i]} --gpu ${gpu_list[$i]} --num_parts 4 --model_local_path $model_local_path --feat_root $feat_root --video_root $video_root &
  done
  wait
done