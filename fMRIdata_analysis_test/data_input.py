import nipype.interfaces.io as nio
datasource1 = nio.DataGrabber()
datasource1.inputs.base_directory = '/data/ds000114'
datasource1.inputs.template = 'sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz'
datasource1.inputs.sort_filelist = True
results = datasource1.run()
results.outputs
