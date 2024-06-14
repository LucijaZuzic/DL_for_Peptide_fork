import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
)

new = [ 
    "PTPCY",
    "PPPHY",
    "SYCGY",
    "KWMDF",
    "FFEKF",
    "KWEFY",
    "FKFEF",
    "RWLDY",
    "WKPYY",
    "VVVVV",
    "FKIDF",
    "VKVFF",
    "KFFFE",
    "KFAFD",
    "VKVEV",
    "RVSVD", 
    "KKFDD",
    "VKVKV",
    "KVKVK",
    "DPDPD",
]
model_list = ["Random Forest"]

def return_seq_predictions_labels(some_seed):
    labels_dict = dict()
    for test_number in range(1, 6):
        dir_name = 'results_seq/mine/no_val/seed_{}/test_{}/'.format(some_seed, test_number)
        used_file_name_part = 'Test_cla_seed_{}_test_{}_reg_{}'.format(some_seed, test_number, "RNN")
        for one_file in os.listdir(dir_name):
            if used_file_name_part in one_file:
                one_file_read = pd.read_csv(dir_name + one_file)
                print(one_file)
                old_feature, old_predict, old_label = one_file_read['feature'], one_file_read['predict'], one_file_read['label']
                for ix in range(len(old_feature)):
                    labels_dict[old_feature[ix]] = old_label[ix]
                break
        
    dir_name = 'rf_predictions.csv'
    one_file_read = pd.read_csv(dir_name)
    print(dir_name)
    features, predictions, seeds = one_file_read['sequence'], one_file_read['prediction'], one_file_read['seed']
    features_filter, predictions_filter, labels_filter = [], [], []
    for ix in range(len(features)):
        if seeds[ix] == some_seed:
            features_filter.append(features[ix])
            predictions_filter.append(predictions[ix])
            labels_filter.append(labels_dict[features[ix]])
    return features_filter, predictions_filter, labels_filter, one_file
        
def weird_division(n, d):
    return n / d if d else 0

# Convert probability to class based on the threshold of probability
def convert_to_binary(model_predictions, threshold=0.5):
    model_predictions_binary = []

    for x in model_predictions:
        if x >= threshold:
            model_predictions_binary.append(1.0)
        else:
            model_predictions_binary.append(0.0)

    return model_predictions_binary

# Count correct predictions based on a custom threshold of probability
def my_accuracy_calculate(test_labels, model_predictions, threshold=0.5):
    score = 0

    model_predictions = convert_to_binary(model_predictions, threshold)

    for i in range(len(test_labels)):
        if model_predictions[i] == test_labels[i]:
            score += 1

    return score / len(test_labels) * 100

def returnGMEAN(actual, pred):
    tn = 0
    tp = 0
    apo = 0
    ane = 0
    for i in range(len(pred)):
        a = actual[i]
        p = pred[i]
        if a == 1:
            apo += 1
        else:
            ane += 1
        if p == a:
            if a == 1:
                tp += 1
            else:
                tn += 1

    return np.sqrt(weird_division(tp, apo) * weird_division(tn, ane))

def read_data(test_labels, model_predictions, lines_dict): 
    
    #lines_dict["AUC = "] = auc(test_labels, model_predictions)
    lines_dict["gmean = "] = returnGMEAN(test_labels, model_predictions)
    lines_dict["F1 = "] = f1_score(test_labels, model_predictions)
    lines_dict["Accuracy = "] = my_accuracy_calculate(test_labels, model_predictions, 0.5)

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

def return_lens():
    lens = dict()
    seqs, predict, labs, one_file = return_seq_predictions_labels(seed_list[0])
    for seq_ix in range(len(seqs)):
        if len(seqs[seq_ix]) not in lens:
            lens[len(seqs[seq_ix])] = 0
        lens[len(seqs[seq_ix])] += 1
    return(lens)
        
def count_classes_seed_test(seed_val, minlen, maxlen):
    classes = dict() 
    seqs, predict, labs, one_file = return_seq_predictions_labels(seed_val)
    for seq_ix in range(len(seqs)):
        if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
            continue
        if labs[seq_ix] not in classes:
            classes[labs[seq_ix]] = 0
        classes[labs[seq_ix]] += 1
    return(classes)
    
def count_classes(minlen, maxlen):
    classes = dict()
    seqs, predict, labs, one_file = return_seq_predictions_labels(seed_list[0])
    for seq_ix in range(len(seqs)):
        if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
            continue
        if labs[seq_ix] not in classes:
            classes[labs[seq_ix]] = 0
        classes[labs[seq_ix]] += 1
    return(classes)

def filter_dict(minlen, maxlen):
    models_line_dicts = dict()
    for model_name in model_list:
        models_line_dicts[model_name] = dict()
        for seed_val in seed_list:
            models_line_dicts[model_name][seed_val] = dict()
            lines_dict = dict()
            predict_filter = []
            seqs_filter = []
            labs_filter = []
            seqs, predict, labs, one_file = return_seq_predictions_labels(seed_val)
            for seq_ix in range(len(seqs)):
                if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
                    continue
                predict_filter.append(predict[seq_ix])
                seqs_filter.append(seqs[seq_ix])
                labs_filter.append(labs[seq_ix])
            read_data(labs_filter, predict_filter, lines_dict)
            if not os.path.isdir('results_processed_random_forest/' + str(minlen) + "_" + str(maxlen) + "/" + model_name):
                os.makedirs('results_processed_random_forest/' + str(minlen) + "_" + str(maxlen) + "/" + model_name)
            df_new = pd.DataFrame({"preds": predict_filter, "labels": labs_filter, "feature": seqs_filter})
            df_new.to_csv('results_processed_random_forest/' + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_" + str(seed_val) + "_preds.csv")
            models_line_dicts[model_name][seed_val] = lines_dict
    return models_line_dicts
 
def print_dict(dicti, mini, maxi):
    colnames = ["Metric"]
    for model_name in model_list:
        for seed_val in seed_list:
            colnames.append('{} (seed {})'.format(model_name, seed_val))
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    for metric in dicti[model_list[0]][seed_list[0]]:
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for model_name in model_list:
            for seed_val in seed_list:
                dict_csv_data['{} (seed {})'.format(model_name, seed_val)].append(dicti[model_name][seed_val][metric])
    if not os.path.isdir('results_processed_random_forest/' + str(mini) + "_" + str(maxi)):
        os.makedirs('results_processed_random_forest/' + str(mini) + "_" + str(maxi))
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv('results_processed_random_forest/' + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + ".csv")
 
def print_dict_model(dicti, model_name, mini, maxi):
    colnames = ["Metric"]
    for seed_val in seed_list:
        colnames.append('{} (seed {})'.format(model_name, seed_val))
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    for metric in dicti[model_list[0]][seed_list[0]]:
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for seed_val in seed_list:
            dict_csv_data['{} (seed {})'.format(model_name, seed_val)].append(dicti[model_name][seed_val][metric])
    if not os.path.isdir('results_processed_random_forest/' + str(mini) + "_" + str(maxi) + "/" + model_name):
        os.makedirs('results_processed_random_forest/' + str(mini) + "_" + str(maxi) + "/" + model_name)
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv('results_processed_random_forest/' + str(mini) + "_" + str(maxi) + "/" + model_name + "/" + str(mini) + "_" + str(maxi) + "_" + model_name + ".csv")

lens = return_lens()
for lena in sorted(lens.keys()):
    print(lena, lens[lena])
larger = []
for lena in sorted(lens.keys()):
    print(lena, count_classes(lena, lena), lens[lena])
    if len(count_classes(lena, lena)) > 1:
        larger.append(lena)
print(larger)

for a in sorted(lens.keys()):
    for b in sorted(lens.keys()):
        if b < a:
            continue
        rnge = list(range(a, b + 1))
        is_range_ok = False
        for r in rnge:
            if r in larger:
                is_range_ok = True
                break
        if not is_range_ok:
            continue

        dict_use = filter_dict(a, b)
        print_dict(dict_use, a, b)
        for model_name in model_list:
            print_dict_model(dict_use, model_name, a, b)
        print(min(lens), a - 1, count_classes(min(lens), a - 1), a, b, count_classes(a, b), b + 1, max(lens), count_classes(b + 1, max(lens)))