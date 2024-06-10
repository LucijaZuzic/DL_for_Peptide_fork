import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
)

model_list = ["Bi-LSTM", "LSTM", "MLP", "Transformer", "RNN"]

usable_csv = pd.read_csv("text_strong_all.csv")
usable_pep = list(usable_csv["Feature"])

def return_seq_predictions_labels(use_model):
    features_filter, predictions_filter, labels_filter = [], [], []
    dir_name = 'results_seq/mine/genetic/'
    used_file_name_part = 'Test_cla_reg_{}'.format(use_model)
    for one_file in os.listdir(dir_name):
        if used_file_name_part in one_file:
            one_file_read = pd.read_csv(dir_name + one_file)
            print(one_file)
            features, predictions, labels = one_file_read['feature'], one_file_read['predict'], one_file_read['label']
            for ix in range(len(features)):
                if features[ix] in usable_pep:
                    features_filter.append(features[ix])
                    predictions_filter.append(predictions[ix])
                    labels_filter.append(labels[ix])

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

def return_lens():
    lens = dict()
    seqs, predict, labs, one_file = return_seq_predictions_labels(model_list[0])
    for seq_ix in range(len(seqs)):
        if len(seqs[seq_ix]) not in lens:
            lens[len(seqs[seq_ix])] = 0
        lens[len(seqs[seq_ix])] += 1
    return(lens)
    
def count_classes(minlen, maxlen):
    classes = dict() 
    seqs, predict, labs, one_file = return_seq_predictions_labels(model_list[0])
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
        seqs, predict, labs, one_file = return_seq_predictions_labels(model_name)
        lines_dict = dict()
        predict_filter = []
        seqs_filter = []
        labs_filter = []
        for seq_ix in range(len(seqs)):
            if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
                continue
            predict_filter.append(predict[seq_ix])
            seqs_filter.append(seqs[seq_ix])
            labs_filter.append(labs[seq_ix])
        read_data(labs_filter, predict_filter, lines_dict)
        models_line_dicts[model_name] = lines_dict
        if not os.path.isdir('results_processed_seq_genetic_strong/' + str(minlen) + "_" + str(maxlen) + "/" + model_name):
            os.makedirs('results_processed_seq_genetic_strong/' + str(minlen) + "_" + str(maxlen) + "/" + model_name)
        df_new = pd.DataFrame({"preds": predict_filter, "labels": labs_filter})
        df_new.to_csv('results_processed_seq_genetic_strong/' + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_preds.csv")
    return models_line_dicts
 
def print_dict(dicti, mini, maxi):
    colnames = ["Metric"]
    for model_name in model_list:
        colnames.append('{}'.format(model_name))
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    for metric in dicti[model_list[0]]:
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for model_name in model_list:
            dict_csv_data['{}'.format(model_name)].append(dicti[model_name][metric])
    if not os.path.isdir('results_processed_seq_genetic_strong/' + str(mini) + "_" + str(maxi)):
        os.makedirs('results_processed_seq_genetic_strong/' + str(mini) + "_" + str(maxi))
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv('results_processed_seq_genetic_strong/' + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + ".csv")

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
        dict_use = filter_dict(a, b)
        rnge = list(range(a, b + 1))
        is_range_ok = False
        for r in rnge:
            if r in larger:
                is_range_ok = True
                break
        if not is_range_ok:
            continue
         
        print_dict(dict_use, a, b) 
        
        print(min(lens), a - 1, count_classes(min(lens), a - 1), a, b, count_classes(a, b), b + 1, max(lens), count_classes(b + 1, max(lens)))