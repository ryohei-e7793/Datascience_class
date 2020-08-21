import subprocess

for i in range(1, 9):
    cmd = 'python nii2png.py -i ../../Class/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00' + i + '/BraTS20_Training_00' + i + '_flair.nii.gz -o ../../Class/BraTS2020_png/Training_flair_00'+ i + '/'
    process = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')
