import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from models_seq import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Classification',
                    choices=['Classification','Regression'])
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
# parser.add_argument('--hidden', type=int, default=256,
                    # help='Number of hidden units.')
parser.add_argument('--src_vocab_size', type=int, default=21) # number of amino acids + 'Empty'
parser.add_argument('--src_len', type=int, default=24)
# parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--model', type=str, default='Bi-LSTM',choices=['RNN','LSTM','Bi-LSTM','Transformer'])

args = parser.parse_args()

# Transformer Parameters
if args.model == 'Transformer':
    args.d_model = 512  # Embedding size
    args.d_ff = 2048  # FeedForward dimension
    args.d_k = args.d_v = 64  # dimension of K(=Q), V
    args.n_layers = 6  # number of Encoder and Decoder Layer
    args.n_heads = 8  # number of heads in Multi-Head Attention
else:
    args.d_model = 512  # Embedding size

device = 'cuda' if torch.cuda.is_available() else 'cpu'    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)  
    
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

def main():

    for some_seed in seed_list:
        for test_number in range(1, 6):

            if args.task_type == 'Classification':
                df_train = pd.read_csv('Sequential_Peptides/mine/no_val/seed_{}/test_{}/train_val_seqs_seed_{}_test_{}.csv'.format(some_seed, test_number, some_seed, test_number))
                df_test = pd.read_csv('Sequential_Peptides/mine/no_val/seed_{}/test_{}/test_seqs_seed_{}_test_{}.csv'.format(some_seed, test_number, some_seed, test_number))
                train_label = torch.Tensor(np.array(df_train["Label"])).long()
                test_label = torch.Tensor(np.array(df_test["Label"])).long().to(device)
            elif args.task_type == 'Regression':
                df_train = pd.read_csv('Sequential_Peptides/mine/no_val/seed_{}/test_{}/train_val_seqs_seed_{}_test_{}.csv'.format(some_seed, test_number, some_seed, test_number))
                df_test = pd.read_csv('Sequential_Peptides/mine/no_val/seed_{}/test_{}/test_seqs_seed_{}_test_{}.csv'.format(some_seed, test_number, some_seed, test_number))
                train_label = torch.Tensor(np.array(df_train["Label"])).unsqueeze(1).float()
                test_label = torch.Tensor(np.array(df_test["Label"])).unsqueeze(1).float().to(device)
            
            args.max = train_label.max().item()
            args.min = train_label.min().item()

            train_feat = np.array(df_train["Feature"])
            test_feat = np.array(df_test["Feature"])

            train_enc_inputs = make_data(train_feat,args.src_len)
            test_enc_inputs = make_data(test_feat,args.src_len).to(device)

            train_loader = Data.DataLoader(MyDataSet(train_enc_inputs,train_label), args.batch_size, True)

            valid_mse_saved = 100
            valid_acc_saved = 0
            loss_mse = torch.nn.MSELoss()

            if args.model == 'Transformer':
                model = Transformer(args).to(device)
                optimizer = optim.SGD(model.parameters(), lr=0.2)
            elif args.model == 'LSTM':
                model = LSTM(args).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            elif args.model == 'Bi-LSTM':
                model = BidirectionalLSTM(args).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            elif args.model == 'RNN':
                model = RNN(args).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(args.epochs):
                model.train()
                for enc_inputs,labels in train_loader:
                    enc_inputs = enc_inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(enc_inputs)

                    if args.task_type == 'Classification':
                        loss = F.nll_loss(outputs, labels)
                    elif args.task_type == 'Regression':
                        loss = loss_mse(outputs, labels)

                    # print('Epoch:','%04d' % (epoch+1), 'loss =','{:.6f}'.format(loss))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if args.task_type == 'Classification':
                if not os.path.isdir('models/no_val/'):
                    os.makedirs('models/no_val/')
                torch.save(model.state_dict(),'models/no_val/seed_{}/test_{}/cla_seed_{}_test_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, some_seed, test_number, args.model,args.lr,args.batch_size))
            elif args.task_type == 'Regression':
                if not os.path.isdir('models/no_val/'):
                    os.makedirs('models/no_val/')
                torch.save(model.state_dict(),'models/no_val/seed_{}/test_{}/reg_seed_{}_test_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, some_seed, test_number, args.model,args.lr,args.batch_size))
            
            predict = []

            if args.model == 'Transformer':
                model_load = Transformer(args).to(device)
            elif args.model == 'LSTM':
                model_load = LSTM(args).to(device)
            elif args.model == 'Bi-LSTM':
                model_load = BidirectionalLSTM(args).to(device)
            elif args.model == 'RNN':
                model_load = RNN(args).to(device)
            if args.task_type == 'Classification':
                checkpoint = torch.load('models/no_val/seed_{}/test_{}/cla_seed_{}_test_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, some_seed, test_number, args.model,args.lr,args.batch_size))
            elif args.task_type == 'Regression':
                checkpoint = torch.load('models/no_val/seed_{}/test_{}/reg_seed_{}_test_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, some_seed, test_number, args.model,args.lr,args.batch_size))
            model_load.load_state_dict(checkpoint)
            model_load.eval()
            
            outputs = model_load(test_enc_inputs)

            if args.task_type == 'Classification':
                
                predict_test = outputs.max(1)[1].type_as(test_label)
                predict = predict + predict_test.cpu().detach().numpy().tolist()

                df_test_save = pd.DataFrame()
                labels = test_label.tolist()
                df_test_seq = pd.read_csv('Sequential_Peptides/mine/no_val/seed_{}/test_{}/test_seqs_seed_{}_test_{}.csv'.format(some_seed, test_number, some_seed, test_number))
                df_test_save['feature'] = df_test_seq['Feature']
                df_test_save['predict'] = predict
                df_test_save['label'] = labels
                test_acc = accuracy(model_load(test_enc_inputs),test_label).item()
                df_test_save['Acc'] = test_acc

                from sklearn.metrics import precision_score, recall_score, f1_score
                df_test_save['Precision'] = precision_score(test_label.cpu(),predict_test.cpu().detach())
                df_test_save['Recall'] = recall_score(test_label.cpu(),predict_test.cpu().detach())
                df_test_save['F1-score'] = f1_score(test_label.cpu(),predict_test.cpu().detach())

                if not os.path.isdir('results_seq/mine/no_val/seed_{}/test_{}/'.format(some_seed, test_number)):
                    os.makedirs('results_seq/mine/no_val/seed_{}/test_{}/'.format(some_seed, test_number))
                
                df_test_save.to_csv('results_seq/mine/no_val/seed_{}/test_{}/Test_cla_seed_{}_test_{}_reg_{}_Acc_{}_lr_{}_bs_{}.csv'.format(some_seed, test_number, some_seed, test_number, args.model,test_acc,args.lr,args.batch_size))

            if args.task_type == 'Regression':

                predict = predict + outputs.squeeze(1).cpu().detach().numpy().tolist()

                df_test_save = pd.DataFrame()
                labels = test_label.squeeze(1).tolist()
                df_test_seq = pd.read_csv('Sequential_Peptides/mine/no_val/seed_{}/test_{}/test_seqs_seed_{}_test_{}.csv'.format(some_seed, test_number, some_seed, test_number))
                df_test_save['feature'] = df_test_seq['Feature']
                df_test_save['predict'] = predict
                df_test_save['label'] = labels
                error = []
                for i in range(len(labels)):
                    error.append(labels[i]-predict[i])
                absError = []
                squaredError = []
                for val in error:
                    absError.append(abs(val))
                    squaredError.append(val*val)

                from sklearn.metrics import r2_score

                MSE = sum(squaredError)/len(squaredError)
                MAE = sum(absError)/len(absError)
                R2 = r2_score(test_label.cpu(),outputs.cpu().detach())

                df_test_save['MSE'] = squaredError
                df_test_save['MAE'] = absError
                df_test_save['MSE_ave'] = MSE
                df_test_save['MAE_ave'] = MAE
                df_test_save['R2'] = R2
                
                if not os.path.isdir('results_seq/mine/no_val/seed_{}/test_{}/'.format(some_seed, test_number)):
                    os.makedirs('results_seq/mine/no_val/seed_{}/test_{}/'.format(some_seed, test_number))
                
                df_test_save.to_csv('results_seq/mine/no_val/seed_{}/test_{}/Test_reg_seed_{}_test_{}_reg_{}_MAE_{}_lr_{}_bs_{}.csv'.format(some_seed, test_number, some_seed, test_number, args.model,MAE,args.lr,args.batch_size))

if __name__ == '__main__':
    main()