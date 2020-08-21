import os.path
import glob
import sys
print(sys.version_info) 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.interpolate as interpolate
import nibabel as nib 
import SimpleITK as sitk
from matplotlib import gridspec


files = glob.glob("/Users/ryoheieguchi/Desktop/Class/BRATS2015_Training/HGG/*/*/*.mha")

T1 = [] 
T1C = []
T2 = []
Flair = []

for i in range(0, len(files)):
    if "VSD.Brain.XX.O.MR_T1." in files[i]:
        T1.append(files[i])
    if "VSD.Brain.XX.O.MR_T2." in files[i]:
        T2.append(files[i])
    if "VSD.Brain.XX.O.MR_T1c." in files[i]:
    	T1C.append(files[i])
    if "VSD.Brain.XX.O.MR_Flair." in files[i]:
    	Flair.append(files[i])


T1_lastpath = []
T1C_lastpath = []
T2_lastpath = []
Flair_lastpath = []

for a in range(0, len(T1)):
    moziretu = T1[a].split("/")
    T1_lastpath.append(moziretu[9])

for b in range(0, len(T2)):
    moziretu2 = T2[b].split("/")
    T2_lastpath.append(moziretu2[9])

for c in range(0, len(T1C)):
    moziretu3 = T1C[c].split("/")
    T1C_lastpath.append(moziretu3[9])

for d in range(0, len(Flair)):
    moziretu4 = Flair[d].split("/")
    Flair_lastpath.append(moziretu4[9])

#niiファイル読み込み

for j in range(0, len(Flair)):
    nii_path = "/Users/ryoheieguchi/Desktop/Class/BRATS2015/HGG_nii/" + Flair_lastpath[j].replace('.mha', '.nii')
    mha_path = Flair[j]
    img = sitk.ReadImage(mha_path)
    sitk.WriteImage(img, nii_path)


