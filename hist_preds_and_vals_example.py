import pandas as pd
import matplotlib.pyplot as plt
from example import survey

model_list = ['Transformer', 'RNN', 'LSTM', 'Bi-LSTM', 'MLP']
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

def merge_format_long(dirname, mini, maxi):
    results = dict()
    for model in model_list:
        tp = 0
        fn = 0
        tn = 0
        fp = 0
        for seed in seed_list:
            pd_file = pd.read_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_" + str(seed) + "_preds.csv")
            preds = [int(x) for x in pd_file["preds"]]
            labs = [int(x) for x in pd_file["labels"]]
            for ix in range(len(preds)):
                if labs[ix] == 1:
                    if preds[ix] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if preds[ix] == 0:
                        tn += 1
                    else:
                        fp += 1
        print(tn, fp, fn, tp)
        results[model] = [tn, fp, fn, tp]
    survey(results)
    plt.savefig(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models_new.png", bbox_inches = "tight")
    plt.close()

def merge_format_long_seed(dirname, mini, maxi):
    results = dict()
    for model in model_list:
        for seed in seed_list:
            tp = 0
            fn = 0
            tn = 0
            fp = 0
            pd_file = pd.read_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_" + str(seed) + "_preds.csv")
            preds = [int(x) for x in pd_file["preds"]]
            labs = [int(x) for x in pd_file["labels"]]
            for ix in range(len(preds)):
                if labs[ix] == 1:
                    if preds[ix] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if preds[ix] == 0:
                        tn += 1
                    else:
                        fp += 1
            print(tn, fp, fn, tp)
            results[model + "\n(seed " + str(seed) + ")"] = [tn, fp, fn, tp]
    survey(results)
    plt.savefig(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models_seeds_new.png", bbox_inches = "tight")
    plt.close()

def merge_format(dirname, mini, maxi):
    results = dict()
    for model in model_list:
        tp = 0
        fn = 0
        tn = 0
        fp = 0
        pd_file = pd.read_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_preds.csv")
        preds = [int(x) for x in pd_file["preds"]]
        labs = [int(x) for x in pd_file["labels"]]
        for ix in range(len(preds)):
            if labs[ix] == 1:
                if preds[ix] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if preds[ix] == 0:
                    tn += 1
                else:
                    fp += 1
        print(tn, fp, fn, tp)
        results[model] = [tn, fp, fn, tp]
    survey(results)
    plt.savefig(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models_new.png", bbox_inches = "tight")
    plt.close()

merge_format_long("results_processed_merged_seq_no_val", 3, 24)
merge_format_long_seed("results_processed_merged_seq_no_val", 3, 24)
merge_format_long("results_processed_20_seq_no_val", 5, 5)
merge_format_long_seed("results_processed_20_seq_no_val", 5, 5)
merge_format("results_processed_seq_no_test", 5, 5)
merge_format("results_processed_seq_long", 5, 10)

model_list = ['Random Forest']
merge_format_long("results_processed_random_forest", 3, 24)
merge_format_long_seed("results_processed_random_forest", 3, 24)
merge_format_long("results_processed_random_forest_20", 5, 5)
merge_format_long_seed("results_processed_random_forest_20", 5, 5)