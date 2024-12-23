import pandas as pd
from feature_sel import *
import numpy as np
import csv
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
import os
# CN-MCI
IDS_PATH=r"./data/ADNI/subject_IDs.txt"         # list for subjId
DEMOPATH=r"./data/ADNI/CN_MCI.csv"              # basic demographic information including age, sex, label, id
NIMGPATH=r"./data/ADNI/DEMO.csv"                # download Subject Demographics [ADNI1,GO,2,3] from ADNI; need to add another row——PTID (format: xxx_S_xxxx)
FILE_PATH=r"./data/ADNI/aal"                    # path for imaging data

def get_ids():
    subject_IDs = np.genfromtxt(IDS_PATH, dtype=str)
    return subject_IDs

def get_subject_score(subject_list, score)->dict:
    scores_dict = {}
    phenotype = DEMOPATH
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Subject ID'] in subject_list:
                scores_dict[row['Subject ID']] = row[score]
    return scores_dict

def get_subject_score_ABIDE(subject_list,score):
    scores_dict = {}
    phenotype = NIMGPATH
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['PTID'] in subject_list:
                scores_dict[row['PTID']] = row[score]
    return scores_dict


def get_data(kind="BOLD"):
    subj_list=get_ids()

    labels=get_subject_score(subj_list,"Research Group")
    ages = get_subject_score(subj_list, score='Age')
    genders = get_subject_score(subj_list, score='Sex')

    EDU=get_subject_score2(subj_list,score="PTEDUCAT")
    MARRY=get_subject_score2(subj_list,score="PTMARRY")
    HOME=get_subject_score2(subj_list,score="PTHOME")
    ETH=get_subject_score2(subj_list,score="PTETHCAT")
    RACE=get_subject_score2(subj_list,score="PTRACCAT")

    subj_num=len(labels)

    y = np.zeros([subj_num])
    age = np.zeros(subj_num, dtype=np.float32)
    gender = np.zeros(subj_num, dtype=np.int32)
    edu=np.zeros(subj_num, dtype=np.int32)
    marry=np.zeros(subj_num, dtype=np.int32)
    home=np.zeros(subj_num, dtype=np.int32)
    eth=np.zeros(subj_num, dtype=np.int32)
    race=np.zeros(subj_num, dtype=np.int32)

    pd_dict = {}

    for i in range(subj_num):
        y[i] = int(labels[str(subj_list[i])])
        age[i] = float(ages[str(subj_list[i])])
        gender[i] = genders[str(subj_list[i])]
        edu[i]=int(EDU[str(subj_list[i])])
        marry[i] = int(MARRY[str(subj_list[i])])
        home[i] = int(HOME[str(subj_list[i])])
        eth[i] = int(ETH[str(subj_list[i])])
        race[i] = int(RACE[str(subj_list[i])])

    phonetic_data = np.zeros([subj_num, 7], dtype=np.float32)
    phonetic_data[:, 0] = age
    phonetic_data[:, 1] = gender
    phonetic_data[:, 2] = edu
    phonetic_data[:, 3] = marry
    phonetic_data[:, 4] = home
    phonetic_data[:, 5] = eth
    phonetic_data[:, 6] = race

    pd_dict['Age'] = np.copy(phonetic_data[:, 0])
    pd_dict['Sex'] = np.copy(phonetic_data[:, 1])

    networks=[]
    data_path=os.path.join(FILE_PATH,kind)
    if kind=="BOLD":
        for subj in subj_list:
            network=np.loadtxt(os.path.join(data_path,subj+".txt"),dtype=np.float32)
            networks.append(network)

    networks=np.array(networks)
    _,roi_num,_=networks.shape
    return networks,y,phonetic_data,pd_dict

def get_fc(atlas="aal",do_zscore = True):
    subj_list = get_ids()
    networks = []
    connexion_path = os.path.join(FILE_PATH, atlas)  # ./data/ADNI/aal
    for subj in subj_list:
        network = sio.loadmat(os.path.join(connexion_path, str(subj), str(subj) + "_" + atlas + "_correlation.mat"))[
            "connectivity"]
        networks.append(network)
    if do_zscore == True:
        for idx, network in enumerate(networks):
            with np.errstate(divide='ignore', invalid='ignore'):
                matrix = np.arctanh(network)
                matrix[matrix == float('inf')] = 0
                networks[idx] = matrix
    networks = np.array(networks)
    return networks

def data_split(features, y, n_folds):
    # split data by k-fold CV
    skf = StratifiedKFold(n_splits=n_folds)
    cv_splits = list(skf.split(features, y))
    return cv_splits

if __name__ == '__main__':
    atlas2roi={"aal":116,"bna":246,"cc":200,"ho":112}
    feature, y, phonetic_data, pd_dict = get_data(kind="BOLD")
    n_folds=10
    lmd=0.005
    cv_splits = data_split(feature, y, n_folds)
    ten_fold_fea_count=np.zeros((10),dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))
        train_eval_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]
        inner_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        train_eval_x, train_eval_y = feature[train_eval_ind], y[train_eval_ind]
        inner_splits = list(skf.split(train_eval_x, train_eval_y))
        train_ind = inner_splits[-1][0]

        one_fold_fea_count = lasso_sel(feature, y, train_ind,lamd=lmd)
        ten_fold_fea_count=ten_fold_fea_count+one_fold_fea_count
    print("*"*100)
    print(ten_fold_fea_count)
    fea_sort=np.argsort(ten_fold_fea_count)[::-1]
    np.savetxt(r"./featureSelection/aal/BOLD/lasso/ans.txt",fea_sort,fmt="%i")
    # print("BOLD selection done")