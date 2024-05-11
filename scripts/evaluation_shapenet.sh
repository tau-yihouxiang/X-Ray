# # state 1: gen low-resolution X-Ray
# python evaluate_diffusion.py --data_root Data/ShapeNetV2_Car \
#                           --exp_name ShapeNetV2_Car \

# state 2: upsample to high-resolution X-Ray
python evaluate_full.py --data_root Data/ShapeNetV2_Car \
                          --exp_diffusion ShapeNetV2_Car \
                          --exp_upsampler ShapeNetV2_Car_upsampler