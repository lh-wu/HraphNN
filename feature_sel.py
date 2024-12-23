import os
from sklearn import datasets
from scipy.stats import kendalltau
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifier,Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold



def data_split(features, y, n_folds):
    # split data by k-fold CV
    skf = StratifiedKFold(n_splits=n_folds)
    cv_splits = list(skf.split(features, y))
    return cv_splits

def svm_sel_roi(feature,y,train_ind,eval_ind):
    train_feature=feature[train_ind]
    eval_feature=feature[eval_ind]
    train_label=y[train_ind]
    eval_label=y[eval_ind]
    subjnum,roinum,feanum=feature.shape
    roi_score=np.zeros(roinum,dtype=np.float32)

    for roi in range(roinum):
        clf=svm.SVC()
        train_roi=train_feature[:,roi,:].reshape(train_label.shape[0],-1)
        eval_roi = eval_feature[:, roi, :].reshape(eval_label.shape[0], -1)
        clf.fit(train_roi,train_label)
        preds=clf.predict(eval_roi)
        score=np.equal(preds,eval_label).sum()/eval_label.shape[0]
        roi_score[roi]=score

    return roi_score

def lda_sel(feature,y,train_ind,eval_ind,fold,sel_roi=90):
    train_feature=feature[train_ind]
    eval_feature=feature[eval_ind]
    train_label=y[train_ind]
    eval_label=y[eval_ind]
    subjnum,roinum,feanum=feature.shape
    roi_score=np.zeros(roinum,dtype=np.float32)

    for roi in range(roinum):
        clf=LDA()
        train_roi = train_feature[:, roi, :].reshape(train_label.shape[0], -1)
        eval_roi = eval_feature[:, roi, :].reshape(eval_label.shape[0], -1)
        clf.fit(train_roi, train_label)
        pred_label=clf.predict(eval_roi)
        acc=np.equal(pred_label,eval_label).sum()
        acc=acc/eval_label.shape[0]
        roi_score[roi]=acc
    print(roi_score)
    return roi_score

def svm_sel_signal(feature,y,train_ind,eval_ind):
    train_feature=feature[train_ind]
    eval_feature=feature[eval_ind]
    train_label=y[train_ind]
    eval_label=y[eval_ind]
    subjnum,roinum,feanum=feature.shape
    fea_score=np.zeros(feanum,dtype=np.float32)

    for fea in range(feanum):
        clf=svm.SVC()
        train_fea=train_feature[:, :, fea].reshape(train_label.shape[0],-1)
        eval_fea = eval_feature[:, :, fea].reshape(eval_label.shape[0], -1)
        clf.fit(train_fea,train_label)
        preds=clf.predict(eval_fea)
        score=np.equal(preds,eval_label).sum()/eval_label.shape[0]
        fea_score[fea]=score

    return fea_score

def lda_sel_signal(feature,y,train_ind,eval_ind):
    train_feature=feature[train_ind]
    eval_feature=feature[eval_ind]
    train_label=y[train_ind]
    eval_label=y[eval_ind]
    subjnum,roinum,feanum=feature.shape
    fea_score=np.zeros(feanum,dtype=np.float32)

    for fea in range(feanum):
        clf=LDA()
        train_fea = train_feature[:, :, fea].reshape(train_label.shape[0], -1)
        eval_fea = eval_feature[:, :, fea].reshape(eval_label.shape[0], -1)
        clf.fit(train_fea, train_label)
        pred_label=clf.predict(eval_fea)
        acc=np.equal(pred_label,eval_label).sum()
        acc=acc/eval_label.shape[0]
        fea_score[fea]=acc
    # print(fea_score)
    return fea_score

def lasso_sel(feature,y,train_ind,lamd):
    '''lasso for BOLD'''
    featureX = feature[train_ind]
    featureY = y[train_ind]
    subjnum,roinum,feanum=feature.shape
    count_list=np.zeros((roinum,feanum),dtype=np.int32)
    for roi in range(roinum):
        lasso = Lasso(alpha=lamd)
        ss=MinMaxScaler()
        roi_feature=featureX[:,roi,:].reshape((featureY.shape[0],-1))
        #roi_feature = ss.fit_transform(roi_feature)

        lasso.fit(roi_feature, featureY)
        roi_imp=lasso.coef_!=0
        count_list[roi][np.where(lasso.coef_!=0)]+=1
        print(lasso.coef_!=0)
    sum_list=count_list.sum(axis=0)
    print(sum_list)
    return sum_list


def rf_sel_signal(feature,y,train_ind):
    featureX = feature[train_ind]
    featureY = y[train_ind]
    subjnum, roinum, feanum = feature.shape
    accumulate_imp=np.zeros(feanum,dtype=np.float32)
    # count_list = np.zeros((roinum, feanum), dtype=np.int32)
    for roi in range(roinum):
        rf=RandomForestClassifier()
        roi_feature = featureX[:, roi, :].reshape((featureY.shape[0], -1))
        rf.fit(roi_feature, featureY)
        imp=rf.feature_importances_
        accumulate_imp=accumulate_imp+imp

    print(accumulate_imp)
    return accumulate_imp



def rfe_sel(feature,y,train_ind):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=2, step=1, verbose=0)
    featureX = feature[train_ind, :]
    featureY = y[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(feature)
    print(x_data)


def kendall_sel_roi(feature_train,y):
    subjnum, roinum, feanum = feature_train.shape
    feature_mean=feature_train.mean(axis=2)
    print(feature_mean.shape)
    ans=np.zeros(roinum,np.float32)
    for i in range(roinum):
        temp_feature=feature_mean[:,i].reshape(-1)
        ans[i]=kendalltau(temp_feature,y)[0]
    sort_ans_idx=np.argsort(ans)[::-1]
    np.savetxt("./featureSelection/aal/BOLD/kendall/ans.txt",sort_ans_idx,fmt="%i")
    print("Region selection done")