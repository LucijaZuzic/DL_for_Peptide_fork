import pandas as pd
import numpy as np

model_list_old = ['Transformer', 'RNN', 'LSTM', 'Bi-LSTM', 'MLP']
model_list_new = ["AP", "SP", "AP-SP", "t-SNE SP", "t-SNE AP-SP"]
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

def merge_format_long(model_list, dirname, mini, maxi):
    dfdict = dict()
    for model in model_list:
        for seed in seed_list:
            pd_file = pd.read_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_" + str(seed) + "_preds.csv")
            dfdict["preds_" + str(model) + "_" + str(seed)] = pd_file["preds"]
            dfdict["labels_" + str(model) + "_" + str(seed)] = pd_file["labels"]
    df_new = pd.DataFrame(dfdict)
    df_new.to_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_preds.csv")

def merge_format(model_list, dirname, mini, maxi):
    dfdict = dict()
    for model in model_list:
        pd_file = pd.read_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_preds.csv")
        dfdict["preds_" + str(model)] = pd_file["preds"]
        dfdict["labels_" + str(model)] = pd_file["labels"]
    df_new = pd.DataFrame(dfdict)
    df_new.to_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_preds.csv")

merge_format_long(model_list_old, "results_processed_merged_seq_no_val", 3, 24)
merge_format_long(model_list_old, "results_processed_20_seq_no_val", 5, 5)
merge_format(model_list_old, "results_processed_seq_no_test", 5, 5)
merge_format(model_list_old, "results_processed_seq_long", 5, 10)