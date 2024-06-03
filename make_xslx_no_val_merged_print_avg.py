import os
import pandas as pd
import numpy as np
model_list = ["Bi-LSTM", "LSTM", "MLP", "Transformer", "RNN"]
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
mini = 3
maxi = 24
dict_new = dict()
for model_name in model_list:
    dict_new[model_name] = dict()
    path_csv = 'results_processed_merged_seq_no_val/' + str(mini) + "_" + str(maxi) + "/" + model_name + "/" + str(mini) + "_" + str(maxi) + "_" + model_name + ".csv"
    file_csv = pd.read_csv(path_csv)
    for c_ix in range(len(file_csv.columns)):
        c = file_csv.columns[c_ix]
        if model_name in c:
            for r_ix in range(len(file_csv[c])):
                metric_name = file_csv["Metric"][r_ix]
                metric_val = file_csv[c][r_ix]
                if metric_name not in dict_new[model_name]:
                    dict_new[model_name][metric_name] = []
                dict_new[model_name][metric_name].append(metric_val)
dict_csv_new = dict()
dict_csv_new["Metric"] = []
for metric_name in dict_new["RNN"]:
    dict_csv_new["Metric"].append(metric_name)
for model_name in dict_new:
    dict_csv_new[model_name] = []
for model_name in dict_new:
    for metric_name in dict_new["RNN"]:
        dict_csv_new[model_name].append(np.average(dict_new[model_name][metric_name]))
df_new = pd.DataFrame(dict_csv_new)
df_new.to_csv('results_processed_merged_seq_no_val/a_result.csv')

mini = 5
maxi = 5
dict_new = dict()
for model_name in model_list:
    dict_new[model_name] = dict()
    path_csv = 'results_processed_20_seq_no_val/' + str(mini) + "_" + str(maxi) + "/" + model_name + "/" + str(mini) + "_" + str(maxi) + "_" + model_name + ".csv"
    file_csv = pd.read_csv(path_csv)
    for c_ix in range(len(file_csv.columns)):
        c = file_csv.columns[c_ix]
        if model_name in c:
            for r_ix in range(len(file_csv[c])):
                metric_name = file_csv["Metric"][r_ix]
                metric_val = file_csv[c][r_ix]
                if metric_name not in dict_new[model_name]:
                    dict_new[model_name][metric_name] = []
                dict_new[model_name][metric_name].append(metric_val)
dict_csv_new = dict()
dict_csv_new["Metric"] = []
for metric_name in dict_new["RNN"]:
    dict_csv_new["Metric"].append(metric_name)
for model_name in dict_new:
    dict_csv_new[model_name] = []
for model_name in dict_new:
    for metric_name in dict_new["RNN"]:
        dict_csv_new[model_name].append(np.average(dict_new[model_name][metric_name]))
df_new = pd.DataFrame(dict_csv_new)
df_new.to_csv('results_processed_20_seq_no_val/a_result.csv')