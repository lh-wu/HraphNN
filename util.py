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



if __name__ == '__main__':
    # subj_path=r"./data/NC-lMCI/subject_IDs.txt"
    # src_fp=r"./data/NC-elMCI/aal/timeseries"
    # dst_fp = r"./data/NC-lMCI/aal/timeseries"
    # subjlist=get_subjlist(subj_path)
    # mv_img_from_subjlist(subjlist,src_fp,dst_fp)
    pass