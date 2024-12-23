import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of HraphNN')
        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--Kfold', type=int, default=10, help='k-fold CrossValidation')
        parser.add_argument('--loopNumber', type=int, default=10, help='loop for running CrossValidation')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
        parser.add_argument('--num_iter', default=300, type=int, help='number of epochs for training')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')

        parser.add_argument('--rKNN', type=int, default=8, help='knn for region hypergraph construction')
        parser.add_argument('--iKNN', type=int, default=8, help='params for iLMNN module')
        parser.add_argument('--pKNN', type=int, default=8, help='knn for population hypergraph construction')

        parser.add_argument('--fRegion', type=int, default=40, help='number of selected Region')
        parser.add_argument('--fBOLD', type=int, default=10, help='number of selected BOLD')

        parser.add_argument('--hidden', type=int, default=32, help='hidden units of graph conv layer')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--ckpt_path', type=str, default='./models', help='checkpoint path to save trained models')

        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("*" * 60)
            if args.device=="cuda":
                print(" Using GPU in torch")
            else:
                print(" Using CPU in torch")
            print("*" * 60)

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        # self.set_seed(12)
        # self.logging_init()
        self.print_args()
        return self.args

    def set_seed(self, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False