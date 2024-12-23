import dhg
import torch
import scipy.io as sio
import torch.nn as nn
from dataloader import *
from opt import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.special import softmax
from torch.nn.parameter import Parameter
from iLMNN import LMNN,get_hanming_matrix
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform,pdist
import numpy as np
import math
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


SUBJ_NUM = 184

def make_connexion(edge_map, sel_k=8):
    edge_map = np.abs(edge_map)
    row, col = edge_map.shape
    ans = []
    for idx in range(row):
        temp = list(np.argsort(edge_map[idx])[0:sel_k])
        ans.append(temp)
    return ans


def build_hg_for_intra_subj(feature):
    subj_num, roi_num, fea_num = feature.shape
    hg_list = []
    for idx in range(subj_num):
        temp_feature = torch.tensor(feature[idx], dtype=torch.float32)
        dist_matrix = squareform(pdist(temp_feature))
        hedge_list = make_connexion(dist_matrix, sel_k=opt.rKNN)
        hg_list.append(dhg.Hypergraph(num_v=roi_num, e_list=hedge_list))
        # hg_list.append(dhg.Hypergraph.from_feature_kNN(temp_feature, k=roi_knn))  # 输入k，建立脑区之间的knn超边
    return hg_list


def similar_metric_learn(phonetic_data):
    # phonetic_data——0age 1gender 2edu 3marry 4home 5eth 6race
    idx=[0,1,2]
    X=phonetic_data[:,idx]
    hm_matrix=get_hanming_matrix(X)
    lmnn=LMNN(k=opt.iKNN,learn_rate=1e-6)
    subj_num_list=np.ones(phonetic_data.shape[0])
    subj_num_list[0]=0
    _,L=lmnn.fit(X, subj_num_list, hanming_matrix=hm_matrix)
    x=X.dot(L.T)
    dist_matrix = squareform(pdist(x))
    return dist_matrix


def build_hg_for_inter_subj(feature, adjust_matrix):
    Tensor_feature = torch.tensor(feature)
    temp = torch.flatten(Tensor_feature, 1, -1)
    dist_matrix=squareform(pdist(temp))
    dist_matrix=1.0*dist_matrix+1.0*adjust_matrix
    hedge_list=make_connexion(dist_matrix,sel_k=opt.pKNN)
    group_hg=dhg.Hypergraph(num_v=feature.shape[0],e_list=hedge_list)
    # group_hg = dhg.Hypergraph.from_feature_kNN(temp, k=opt.pKNN)
    return group_hg


def get_group_laplace(hg_list, feature_map):
    ans = torch.zeros((feature_map.shape[0], feature_map.shape[1], feature_map.shape[1]), dtype=torch.float32)
    for i in range(feature_map.shape[0]):
        ans[i] = hg_list[i].smoothing_with_HGNN(torch.eye(feature_map.shape[1]))
    return ans          #.detach()


class net(nn.Module):
    def __init__(self, roi_num, in_dim):
        super(net, self).__init__()
        size = in_dim
        self.hidden = opt.hidden
        self.weight = Parameter(torch.FloatTensor(size, self.hidden), requires_grad=True)
        self.weight1 = Parameter(torch.FloatTensor(self.hidden, self.hidden), requires_grad=True)
        self.weight2 = Parameter(torch.FloatTensor(self.hidden, self.hidden), requires_grad=True)
        self.relu = nn.ReLU(inplace=False)
        self.linear = nn.Sequential(nn.Linear(roi_num * self.hidden, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 2)
                                    )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)


    def forward(self, feature_map, hg_list, group_hg,_group_laplace):
        '''   '''
        group_laplace=_group_laplace.to(opt.device)
        drop=opt.dropout if self.training else 0.0
        Y = feature_map
        Y = torch.matmul(feature_map.permute(2, 1, 0),
                         group_hg.smoothing_with_HGNN(torch.eye(SUBJ_NUM).to(opt.device), drop_rate=drop).T.to(opt.device)).permute(2, 1, 0)
        Y = self.relu(Y)
        Y = torch.matmul(group_laplace, Y)
        Y = torch.matmul(Y, self.weight)
        Y = self.relu(Y)

        Y = torch.matmul(Y.permute(2, 1, 0),
                         group_hg.smoothing_with_HGNN(torch.eye(SUBJ_NUM).to(opt.device), drop_rate=drop).T.to(opt.device)).permute(2, 1, 0)
        Y = self.relu(Y)
        temp1 = Y.clone()
        Y = torch.matmul(group_laplace, Y)
        Y = torch.matmul(Y, self.weight1)
        Y += temp1
        Y = self.relu(Y)

        Y = torch.matmul(group_laplace, Y)
        Y = torch.matmul(Y, self.weight2)
        Y = self.relu(Y)

        Y = torch.flatten(Y, 1, -1)
        Y = self.linear(Y)

        return Y


def accuracy(preds, labels):
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)


def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  '''
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:, 1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out


def prf(preds, labels, is_logit=True):
    ''' input: logits, labels  '''
    pred_lab = np.argmax(preds, 1)
    p, r, f, s = precision_recall_fscore_support(labels, pred_lab, average='binary')
    return [p, r, f]

def numeric_score(pred, gt):
    FP = np.float32(np.sum((pred == 1) & (gt == 0)))
    FN = np.float32(np.sum((pred == 0) & (gt == 1)))
    TP = np.float32(np.sum((pred == 1) & (gt == 1)))
    TN = np.float32(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN

def metrics(preds, labels):
    preds = np.argmax(preds, 1)
    FP, FN, TP, TN = numeric_score(preds, labels)
    sen = TP / (TP + FN + 1e-10)  # recall sensitivity
    spe = TN / (TN + FP + 1e-10)
    return sen, spe



def train():
    print("\tNumber of training samples %d" % len(train_ind))
    print("\tStart training...\r\n")
    acc = 0.0

    for epoch in range(EPOCH):
        HGnn.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            x = HGnn(Tsel_feature, hg_list, group_hg,group_lplace)
            loss = loss_fn(x[train_ind], label[train_ind])
            loss.backward()
            optimizer.step()
        correct_train, acc_train = accuracy(x[train_ind].detach().cpu().numpy(), label[train_ind].detach().cpu().numpy())
        # print("Epoch: {},\ttrain loss: {:.4f},\ttrain acc: {:.4f}".format(epoch, loss.item(), acc_train.item()))

        HGnn.eval()
        with torch.set_grad_enabled(False):
            eval_x = HGnn(Tsel_feature, hg_list, group_hg,group_lplace)
        logits_eval = eval_x[eval_ind].detach().cpu().numpy()
        label_eval=label[eval_ind].detach().cpu().numpy()

        correct_eval, acc_eval = accuracy(logits_eval, label_eval)
        eval_auc = auc(logits_eval, label_eval)
        prf_eval = prf(logits_eval, label_eval)
        sesp=metrics(logits_eval, label_eval)

        if acc_eval > acc and epoch > 5:
            correct = correct_eval
            acc = acc_eval
            aucs[fold] = eval_auc
            prfs[fold] = prf_eval
            detail_acc[i][fold] = acc_eval
            sesps[fold]=sesp
            torch.save(HGnn, save_path+"/{}-{}.pkl".format(cur,fold))

    corrects[fold] = correct
    accs[fold] = acc
    print("\r\n => Fold {} eval accuacry {:.2f}%,correct num:{}".format(fold, acc * 100, correct))
    print("\r\n => Fold {} eval auc {:.2f}%".format(fold, aucs[fold] * 100))






if __name__ == '__main__':
    opt = OptInit().initialize()

    feature, y, phonetic_data, pd_dict = get_data(kind="BOLD")
    demo_sim = similar_metric_learn(phonetic_data)
    save_path=opt.ckpt_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    EPOCH = opt.num_iter
    loop_number = opt.loopNumber if opt.train else 1
    n_folds = opt.Kfold
    cv_splits = data_split(feature, y, n_folds)

    multi_accs = np.zeros(loop_number, dtype=np.float32)
    multi_aucs = np.zeros(loop_number, dtype=np.float32)
    multi_sens = np.zeros(loop_number, dtype=np.float32)
    multi_spes = np.zeros(loop_number, dtype=np.float32)
    multi_f1s = np.zeros(loop_number, dtype=np.float32)

    detail_acc = np.zeros((loop_number, n_folds), dtype=np.float32)

    eval_acc_auc_sen_spe_f1 = np.zeros((loop_number, 5), dtype=np.float32)

    for i in range(loop_number):
        cur=i
        accs = np.zeros(n_folds, dtype=np.float32)
        aucs = np.zeros(n_folds, dtype=np.float32)
        prfs = np.zeros([n_folds, 3], dtype=np.float32)
        sesps = np.zeros([n_folds, 2], dtype=np.float32)
        corrects = np.zeros(n_folds, dtype=np.int32)
        for fold in range(n_folds):
            print("\r\n========================== Fold {} ==========================".format(fold))
            train_eval_ind = cv_splits[fold][0]
            test_ind = cv_splits[fold][1]

            inner_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
            train_eval_x, train_eval_y = feature[train_eval_ind], y[train_eval_ind]
            inner_splits = list(skf.split(train_eval_x, train_eval_y))
            train_ind = inner_splits[-1][0]
            eval_ind = inner_splits[-1][1]

            roi_sel = np.loadtxt("./featureSelection/aal/BOLD/kendall/ans{}.txt".format(fold), dtype=np.int32)
            roi_sel=roi_sel[0:opt.fRegion]
            roi_sel=np.sort(roi_sel)
            sel_feature=feature[:,roi_sel,:]
            fea_sel = np.loadtxt("./featureSelection/aal/BOLD/lasso/ans{}.txt".format(fold), dtype=np.int32)
            fea_sel=fea_sel[0:opt.fBOLD]
            fea_sel=np.sort(fea_sel)
            sel_feature=sel_feature[:,:,fea_sel]
            hg_list = build_hg_for_intra_subj(sel_feature)
            group_hg = build_hg_for_inter_subj(sel_feature, adjust_matrix=demo_sim)
            group_lplace=get_group_laplace(hg_list,sel_feature)

            if opt.train:
                HGnn = net(sel_feature.shape[1], sel_feature.shape[2]).to(opt.device)
            else:
                HGnn=torch.load("./<path for model>".format(fold),map_location=torch.device('cpu')).to(opt.device)

            loss_fn = nn.CrossEntropyLoss().to(opt.device)
            optimizer = torch.optim.Adam(HGnn.parameters(), lr=opt.lr, weight_decay=opt.wd)
            Tsel_feature = torch.tensor(sel_feature, dtype=torch.float32).to(opt.device)
            label = torch.tensor(y, dtype=torch.long).to(opt.device)

            def eval():
                print("\tNumber of test samples %d" % len(test_ind))
                print("\tStart test...\r\n")
                HGnn.eval()
                test_x = HGnn(Tsel_feature, hg_list, group_hg, group_lplace)
                logits_test = test_x[test_ind].detach().cpu().numpy()

                correct_test, acc_test = accuracy(logits_test, label[test_ind].detach().cpu().numpy())
                test_auc = auc(logits_test, label[test_ind].detach().cpu().numpy())
                prf_test = prf(logits_test, label[test_ind].detach().cpu().numpy())
                sesp = metrics(logits_test, label[test_ind].detach().cpu().numpy())

                print("fold{},acc={} auc={}".format(fold, acc_test, test_auc))

                eval_acc_auc_sen_spe_f1[fold, 0] = acc_test
                eval_acc_auc_sen_spe_f1[fold, 1] = test_auc
                eval_acc_auc_sen_spe_f1[fold, 2] = sesp[0]
                eval_acc_auc_sen_spe_f1[fold, 3] = sesp[1]
                eval_acc_auc_sen_spe_f1[fold, 4] = prf_test[2]

            if opt.train:
                train()
            else:
                eval()
        acc_nfold = np.sum(corrects) / feature.shape[0]
        auc_nfold = np.mean(aucs)
        print("=> Value format: mean(std)")
        print("=> Average eval accuracy in {}-fold CV: {:.4f}({:.4f})".format(n_folds, acc_nfold,np.std(accs)))
        print("=> Average eval auc in {}-fold CV: {:.4f}({:.4f})".format(n_folds, auc_nfold,np.std(aucs)))
        pr, rc, f1 = np.mean(prfs, axis=0)
        std_p, std_r, std_f1 = np.std(prfs, axis=0)
        se,sp=np.mean(sesps,axis=0)
        std_se, std_sp = np.std(sesps, axis=0)
        print("=> Average eval precision {:.4f}({:.4f}), recall {:.4f}({:.4f}), F1-score {:.4f}({:.4f})".format(pr,std_p, rc,std_r, f1,std_f1))
        print("=> Average eval sen {:.4f}({:.4f}), spe {:.4f}({:.4f})".format(se,std_se,sp,std_sp))
        multi_accs[i] = acc_nfold
        multi_aucs[i] = auc_nfold
        multi_sens[i] = se
        multi_spes[i] = sp
        multi_f1s[i] = f1
    print("{} Loop ACC for {}-CV =".format(opt.loopNumber,opt.Kfold), multi_accs)
    print("{} Loop AUC for {}-CV =".format(opt.loopNumber,opt.Kfold), multi_aucs)



    if not opt.train:
        eval_metrix=eval_acc_auc_sen_spe_f1.mean(axis=0)
        std_metrix=np.std(eval_acc_auc_sen_spe_f1,axis=0)
        print("=> Value format: mean(std)")
        print("acc{:.4f}({:.4f}),auc{:.4f}({:.4f}),sen{:.4f}({:.4f}),spe{:.4f}({:.4f}),f1{:.4f}({:.4f})".format(eval_metrix[0],std_metrix[0],eval_metrix[1],std_metrix[1],eval_metrix[2],std_metrix[2],eval_metrix[3],std_metrix[3],eval_metrix[4],std_metrix[4]))

