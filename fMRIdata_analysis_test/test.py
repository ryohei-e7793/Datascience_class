#fmri_filename = "/Users/ryoheieguchi/Downloads/subj2/bold.nii.gz"

from nilearn import plotting
from nilearn.image import mean_img

#plotting.view_img(mean_img(fmri_filename), threshold=None)


from nilearn import datasets
haxby_dataset = datasets.fetch_haxby()

fmri_filename = haxby_dataset.func[0]

print('First subject functional nifti images (4D) are at: %s' %fmri_filename)
plotting.view_img(mean_img(fmri_filename), threshold=None)

