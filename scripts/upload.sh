# Description: Upload the dataset to the Hugging Face Hub

huggingface-cli upload yihouxiang/X-Ray Objaverse_XRay/Objaverse_XRay.zip.part-00 Objaverse_Xay/Objaverse_XRay.zip.part-00 --repo-type dataset
for i in {00..17}; do
    huggingface-cli upload yihouxiang/X-Ray Objaverse_XRay/Objaverse_XRay.zip.part-$i Objaverse_Xay/Objaverse_XRay.zip.part-$i --repo-type dataset
done