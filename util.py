import os
import numpy as np
import shutil

def get_subjlist(path):
    subjlist=np.genfromtxt(path,dtype=str)
    return subjlist


def mv_img_from_subjlist(subjlist,src_fp,dst_fp):
    src_list=os.listdir(src_fp)
    for src_file in src_list:
        temp=src_file.replace(".txt","")
        if temp in subjlist:
            shutil.copy(os.path.join(src_fp,src_file),os.path.join(dst_fp,src_file))
