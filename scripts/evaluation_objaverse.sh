# state 1: gen low-resolution X-Ray
python evaluate_diffusion.py --data_root Data/Objaverse_XRay \
                          --exp Objaverse_XRay \

# # state 2: upsample to high-resolution X-Ray
# python evaluate_upsampler.py --exp_diffusion Objaverse_XRay \
#                           --exp_upsampler Objaverse_XRay_upsampler