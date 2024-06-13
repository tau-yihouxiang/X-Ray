# # state 1: gen low-resolution X-Ray
# python evaluate_diffusion.py --data_root Data/ShapeNetV2_Car \
#                           --exp ShapeNetV2_Car_1 \

# # state 2: upsample to high-resolution X-Ray
python evaluate_upsampler.py --exp_diffusion ShapeNetV2_Car_1 \
                             --exp_upsampler ShapeNetV2_Car_upsampler_large_loss_conv3d