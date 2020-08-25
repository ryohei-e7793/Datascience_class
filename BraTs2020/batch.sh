# Output folder is '' となってしまう
#python nii2png.py -i ../../Class/MICCAI_BraTS2020_TrainingData/*/*.nii.gz -o /Users/ryoheieguchi/Desktop/Class/BraTS2020_png_training/T1/

#!/bin/sh
dir_path="/Users/ryoheieguchi/Desktop/Class/MICCAI_BraTS2020_ValidationData/*"
dirs=`find $dir_path -maxdepth 1 -type f -name *_t2.nii.gz`

for dir in $dirs;
do
    python nii2png.py -i $dir -o /Users/ryoheieguchi/Desktop/Class/BraTS2020_png_validation/T2/
done
