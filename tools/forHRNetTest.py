import os
import subprocess
from shutil import copyfile
from distutils.dir_util import copy_tree

test_file = 'test.lst'
with open(test_file, "w") as the_file:
    for i in range(200):
        the_file.write( '/data/new_disk/luoxi/NEURAL/data/Mars2_mv/out/add_D/baseline_zoom_mono_all/forHRNet_test/vis_img/img_' +str(i).zfill(4) + '.jpg' +'\n')