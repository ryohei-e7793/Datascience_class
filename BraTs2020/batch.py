'''
import subprocess

for i in range(1, 9):
    cmd = 'python nii2png.py -i ../../Class/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00' + i + '/BraTS20_Training_00' + i + '_flair.nii.gz -o ../../Class/BraTS2020_png/Training_flair_00'+ i + '/'
    process = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')

'''

import os.path
import glob
import sys
import subprocess

files = glob.glob("/Users/ryoheieguchi/Desktop/Class/MICCAI_BraTS2020_TrainingData/*/*.nii.gz")

T1 = []
T1C = []
T2 = []
Flair = []

for i in range(0, len(files)):
    if "_t1.nii.gz" in files[i]:
        T1.append(files[i])
    if "_t2.nii.gz" in files[i]:
        T2.append(files[i])
    if "_t1ce.nii.gz" in files[i]:
    	T1C.append(files[i])
    if "_flair.nii.gz" in files[i]:
    	Flair.append(files[i])


for j in range(0, len(T2)):
    cmd = 'python nii2png.py -i ' + T2[j] + ' -o /Users/ryoheieguchi/Desktop/Class/BraTS2020_png_training/T2/'
    process = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')
    

# python nii2png.py -i ../../Class/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz -o ../../Class/BraTS2020_png_training/Training_seg_001/

#['/Users/ryoheieguchi/Desktop/Class/MICCAI_BraTS2020_TrainingData/BraTS20_Training_082/BraTS20_Training_082_t2.nii.gz',... ]


