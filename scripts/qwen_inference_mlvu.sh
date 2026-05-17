CUDA_VISIBLE_DEVICES=1 python3 qwen_inference_ovu_prototrack.py \
   --dataset mlvu   
   --output_dir results/ovu   \
   --exp_tag 7B_32_24_prototrack-kv   \
   --use_block_processing   \
   --block_size 32   \
   --compress_frame_num 24   \
   --model_path Qwen/Qwen2-VL-7B-Instruct   \
   --compression_method prototrack-kv