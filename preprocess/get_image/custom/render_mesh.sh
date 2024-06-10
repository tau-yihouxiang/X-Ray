CUDA_VISIBLE_DEVICES=3 python3 blender/distributed.py \
                              --blender_path /hdd/taohu/Data/Objaverse/objaverse-rendering/blender-3.3.15-linux-x64/blender \
                              --num_gpus 1 \
                              --workers_per_gpu 2 \
                              --input_models_path custom/mesh_list.json \
                              --output_dir Data/Custom/images