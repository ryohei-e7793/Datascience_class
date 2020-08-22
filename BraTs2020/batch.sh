#python nii2png.py -i ../../Class/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz -o ../../Class/BraTS2020_png_training/T1/
#python nii2png.py -i ../../Class/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_t1.nii.gz -o ../../Class/BraTS2020_png_training/T1/

# Output folder is '' となってしまう

python3 nii2png.py -i ../../Class/MICCAI_BraTS2020_TrainingData/*/*_t1.nii.gz -o /Users/ryoheieguchi/Desktop/
