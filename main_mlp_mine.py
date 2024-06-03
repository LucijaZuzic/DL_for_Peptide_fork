import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from models_mlp import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Classification',
                    choices=['Classification','Regression'])
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
# parser.add_argument('--hidden', type=int, default=256,
                    # help='Number of hidden units.')
parser.add_argument('--src_vocab_size', type=int, default=21) # number of amino acids + 'Empty'
parser.add_argument('--src_len', type=int, default=10)
# parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--model', type=str, default='MLP')

args = parser.parse_args()
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
            for fold_nr in range(1, 6):

                if args.task_type == 'Classification':
                    df_train = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/train_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                    df_valid = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/val_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                    df_test = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/test_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                    train_label = torch.Tensor(np.array(df_train["Label"])).long()
                    valid_label = torch.Tensor(np.array(df_valid["Label"])).long().to(device)
                    test_label = torch.Tensor(np.array(df_test["Label"])).long().to(device)
                elif args.task_type == 'Regression':
                    df_train = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/train_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                    df_valid = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/val_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                    df_test = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/test_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                    train_label = torch.Tensor(np.array(df_train["Label"])).unsqueeze(1).float()
                    valid_label = torch.Tensor(np.array(df_valid["Label"])).unsqueeze(1).float().to(device)
                    test_label = torch.Tensor(np.array(df_test["Label"])).unsqueeze(1).float().to(device)
                
                train_feat = np.array(df_train["Feature"])
                valid_feat = np.array(df_valid["Feature"])
                test_feat = np.array(df_test["Feature"])

                train_enc_inputs = make_data(train_feat,args.src_len)
                valid_enc_inputs = make_data(valid_feat,args.src_len).to(device)
                test_enc_inputs = make_data(test_feat,args.src_len).to(device)

                train_loader = Data.DataLoader(MyDataSet(train_enc_inputs,train_label), args.batch_size, True)

                valid_mse_saved = 100
                valid_acc_saved = 0
                loss_mse = torch.nn.MSELoss()

                model = MLP(args).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.01)

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

                    if (epoch+1) % 1 == 0:
                        model.eval()
                        predict = model(valid_enc_inputs)
                        
                        if args.task_type == 'Classification':
                            valid_acc = accuracy(predict, valid_label)
                            if valid_acc_saved < valid_acc:
                                valid_acc_saved = valid_acc
                                print('Epoch:',epoch+1)
                                print('Valid Performance:',valid_acc.item())
                                if not os.path.isdir('models/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr)):
                                    os.makedirs('models/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr))
                                torch.save(model.state_dict(),'models/seed_{}/test_{}/fold_{}/cla_seed_{}_test_{}_fold_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, fold_nr, some_seed, test_number, fold_nr, args.model,args.lr,args.batch_size))

                        elif args.task_type == 'Regression':
                            valid_mse = loss_mse(predict,valid_label).item()
                            if valid_mse_saved > valid_mse:
                                valid_mse_saved = valid_mse
                                print('Epoch:',epoch+1)
                                print('Valid Performance:',valid_mse)
                                if not os.path.isdir('models/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr)):
                                    os.makedirs('models/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr))
                                torch.save(model.state_dict(),'models/seed_{}/test_{}/fold_{}/reg_seed_{}_test_{}_fold_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, fold_nr, some_seed, test_number, fold_nr, args.model,args.lr,args.batch_size))

                predict = []

                model_load = MLP(args).to(device)
                if args.task_type == 'Classification':
                    checkpoint = torch.load('models/seed_{}/test_{}/fold_{}/cla_seed_{}_test_{}_fold_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, fold_nr, some_seed, test_number, fold_nr, args.model,args.lr,args.batch_size))
                elif args.task_type == 'Regression':
                    checkpoint = torch.load('models/seed_{}/test_{}/fold_{}/reg_seed_{}_test_{}_fold_{}_reg_{}_lr_{}_bs_{}.pt'.format(some_seed, test_number, fold_nr, some_seed, test_number, fold_nr, args.model,args.lr,args.batch_size))
                model_load.load_state_dict(checkpoint)
                model_load.eval()
                
                outputs = model_load(test_enc_inputs)

                if args.task_type == 'Classification':
                    
                    predict_test = outputs.max(1)[1].type_as(test_label)
                    predict = predict + predict_test.cpu().detach().numpy().tolist()

                    df_test_save = pd.DataFrame()
                    labels = test_label.tolist()
                    df_test_seq = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/test_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                    df_test_save['feature'] = df_test_seq['Feature']
                    df_test_save['predict'] = predict
                    df_test_save['label'] = labels
                    test_acc = accuracy(model_load(test_enc_inputs),test_label).item()
                    df_test_save['Acc'] = test_acc
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    df_test_save['Precision'] = precision_score(test_label.cpu(),predict_test.cpu().detach())
                    df_test_save['Recall'] = recall_score(test_label.cpu(),predict_test.cpu().detach())
                    df_test_save['F1-score'] = f1_score(test_label.cpu(),predict_test.cpu().detach())

                    if not os.path.isdir('results_seq/mine/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr)):
                        os.makedirs('results_seq/mine/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr))
                    
                    df_test_save.to_csv('results_seq/mine/seed_{}/test_{}/fold_{}/Test_cla_seed_{}_test_{}_fold_{}_reg_{}_Acc_{}_lr_{}_bs_{}.csv'.format(some_seed, test_number, fold_nr, some_seed, test_number, fold_nr, args.model,test_acc,args.lr,args.batch_size))

                if args.task_type == 'Regression':

                    predict = predict + outputs.squeeze(1).cpu().detach().numpy().tolist()

                    df_test_save = pd.DataFrame()
                    labels = test_label.squeeze(1).tolist()
                    df_test_seq = pd.read_csv('Sequential_Peptides/mine/seed_{}/test_{}/test_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
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
                    MSE = sum(squaredError)/len(squaredError)
                    MAE = sum(absError)/len(absError)
                    from sklearn.metrics import r2_score
                    R2 = r2_score(test_label.cpu(),outputs.cpu().detach())
                    df_test_save['MSE'] = squaredError
                    df_test_save['MAE'] = absError
                    df_test_save['MSE_ave'] = MSE
                    df_test_save['MAE_ave'] = MAE
                    df_test_save['R2'] = R2

                    if not os.path.isdir('results_seq/mine/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr)):
                        os.makedirs('results_seq/mine/seed_{}/test_{}/fold_{}/'.format(some_seed, test_number, fold_nr))
                    
                    df_test_save.to_csv('results_seq/mine/seed_{}/test_{}/fold_{}/Test_reg_seed_{}_test_{}_fold_{}_reg_{}_MAE_{}_lr_{}_bs_{}.csv'.format(some_seed, test_number, fold_nr, some_seed, test_number, fold_nr, args.model,MAE,args.lr,args.batch_size))

if __name__ == '__main__':
    main()