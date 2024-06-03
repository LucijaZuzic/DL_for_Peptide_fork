import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)

# Count correct predictions based on a custom threshold of probability
def my_accuracy_calculate(test_labels, model_predictions, threshold=0.5):
    score = 0

    model_predictions = convert_to_binary(model_predictions, threshold)

    for i in range(len(test_labels)):
        if model_predictions[i] == test_labels[i]:
            score += 1

    return score / len(test_labels) * 100

# Convert probability to class based on the threshold of probability
def convert_to_binary(model_predictions, threshold=0.5):
    model_predictions_binary = []

    for x in model_predictions:
        if x >= threshold:
            model_predictions_binary.append(1.0)
        else:
            model_predictions_binary.append(0.0)

    return model_predictions_binary


def weird_division(n, d):
    return n / d if d else 0,

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

def read_ROC(test_labels, model_predictions, lines_dict, prthr, rocthr):
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)

    model_predictions_binary_thrROC = convert_to_binary(
        model_predictions, rocthr
    )

    # Get recall and precision.
    precision, recall, thresholdsPR = precision_recall_curve(
        test_labels, model_predictions
    )

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ixPR = np.argmax(fscore)

    model_predictions_binary_thrPR = convert_to_binary(
        model_predictions, prthr
    )

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    lines_dict["ROC thr = "].append(rocthr)
    lines_dict["gmean (0.5) = "].append(
        returnGMEAN(test_labels, model_predictions_binary)
    )
    lines_dict["gmean (PR thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrPR)
    )
    lines_dict["gmean (ROC thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrROC)
    )
    lines_dict["ROC AUC = "].append(roc_auc_score(test_labels, model_predictions))
    lines_dict["Accuracy (ROC thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, rocthr)
    )


def read_PR(test_labels, model_predictions, lines_dict, prthr, rocthr):
    # Get recall and precision.
    precision, recall, thresholds = precision_recall_curve(
        test_labels, model_predictions
    )

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ix = np.argmax(fscore)
    model_predictions_binary_thrPR = convert_to_binary(
        model_predictions, prthr
    )
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ixROC = np.argmax(gmeans)

    model_predictions_binary_thrROC = convert_to_binary(
        model_predictions, rocthr
    )

    lines_dict["PR thr = "].append(prthr)
    lines_dict["PR AUC = "].append(auc(recall, precision))
    lines_dict["F1 (0.5) = "].append(f1_score(test_labels, model_predictions_binary))
    lines_dict["F1 (PR thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR)
    )
    lines_dict["F1 (ROC thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrROC)
    )
    lines_dict["Accuracy (PR thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, prthr)
    )
    lines_dict["Accuracy (0.5) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, 0.5)
    )
params_for_model = {"AP": 1, "SP": 7, "AP_SP": 1, "TSNE_SP": 5, "TSNE_AP_SP": 8}
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
PRthr = {
    "AP": 0.2581173828,
    "SP": 0.31960518800000004,
    "AP_SP": 0.3839948796,
    "TSNE_SP": 0.30602415759999996,
    "TSNE_AP_SP": 0.34321978799999997,
}
ROCthr = {
    "AP": 0.5024869916000001,
    "SP": 0.5674178616000001,
    "AP_SP": 0.5708274524,
    "TSNE_SP": 0.566148754,
    "TSNE_AP_SP": 0.5588111376,
}
vals_in_lines = [
    "ROC thr = ",
    "PR thr = ",
    "ROC AUC = ",
    "gmean (ROC thr) = ",
    "F1 (ROC thr) = ",
    "Accuracy (ROC thr) = ",
    "PR AUC = ",
    "gmean (PR thr) = ",
    "F1 (PR thr) = ",
    "Accuracy (PR thr) = ",
    "gmean (0.5) = ",
    "F1 (0.5) = ",
    "Accuracy (0.5) = ",
]

def return_lens():
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        for seed_val_ix in range(len(seed_list)):
            seed_val = seed_list[seed_val_ix]
            lens = dict()
            for test_num in range(1, 6):
                csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
                seqs = list(csv_seed_test_fold["sequence"])
                for seq_ix in range(len(seqs)):
                    if len(seqs[seq_ix]) not in lens:
                        lens[len(seqs[seq_ix])] = 0
                    lens[len(seqs[seq_ix])] += 1
            return(lens)
        
def count_classes_seed_test(seed_val, test_num, minlen, maxlen):
    classes = dict() 
    csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
    seqs = list(csv_seed_test_fold["sequence"])
    labs = list(csv_seed_test_fold["label"])
    for seq_ix in range(len(seqs)):
        if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
            continue
        if labs[seq_ix] not in classes:
            classes[labs[seq_ix]] = 0
        classes[labs[seq_ix]] += 1
    return(classes)
    
def count_classes(minlen, maxlen):
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        for seed_val_ix in range(len(seed_list)):
            seed_val = seed_list[seed_val_ix]
            classes = dict()
            for test_num in range(1, 6):
                csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
                seqs = list(csv_seed_test_fold["sequence"])
                labs = list(csv_seed_test_fold["label"])
                for seq_ix in range(len(seqs)):
                    if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
                        continue
                    if labs[seq_ix] not in classes:
                        classes[labs[seq_ix]] = 0
                    classes[labs[seq_ix]] += 1
            return(classes)

def filter_dict(minlen, maxlen):
    models_line_dicts = dict()
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        #print(model_name.replace("_model_data", "").replace("_data", ""))
        lines_dict = dict()
        for v in vals_in_lines:
            lines_dict[v] = []
        for seed_val_ix in range(len(seed_list)):
            seed_val = seed_list[seed_val_ix]
            #print(seed_val)
            pred_arr1_filter = []
            pred_arr2_filter = []
            seqs_filter = []
            labs_filter = []
            for test_num in range(1, 6):
                dir_pred = "../seeds/seed_" + str(seed_val) + "/" + model_name + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num))
                csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
                seqs = list(csv_seed_test_fold["sequence"])
                labs = list(csv_seed_test_fold["label"])
                pred_file = open(dir_pred + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_predictions.txt", "r")
                pred_arrs = pred_file.readlines()
                pred_arr1 = eval(pred_arrs[0])
                pred_arr2 = eval(pred_arrs[1])
                for seq_ix in range(len(seqs)):
                    if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
                        continue
                    pred_arr1_filter.append(pred_arr1[seq_ix])
                    pred_arr2_filter.append(pred_arr2[seq_ix])
                    seqs_filter.append(seqs[seq_ix])
                    labs_filter.append(labs[seq_ix])
            read_PR(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
            read_ROC(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        #print(lines_dict)
        models_line_dicts[model_name.replace("_model_data", "").replace("_data", "")] = lines_dict
    return models_line_dicts

def filter_dict_seed(minlen, maxlen, seed_val):
    models_line_dicts = dict()
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        #print(model_name.replace("_model_data", "").replace("_data", ""))
        lines_dict = dict()
        for v in vals_in_lines:
            lines_dict[v] = []
        for test_num in range(1, 6):
            pred_arr1_filter = []
            pred_arr2_filter = []
            seqs_filter = []
            labs_filter = []
            dir_pred = "../seeds/seed_" + str(seed_val) + "/" + model_name + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num))
            csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
            seqs = list(csv_seed_test_fold["sequence"])
            labs = list(csv_seed_test_fold["label"])
            pred_file = open(dir_pred + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_predictions.txt", "r")
            pred_arrs = pred_file.readlines()
            pred_arr1 = eval(pred_arrs[0])
            pred_arr2 = eval(pred_arrs[1])
            for seq_ix in range(len(seqs)):
                if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
                    continue
                pred_arr1_filter.append(pred_arr1[seq_ix])
                pred_arr2_filter.append(pred_arr2[seq_ix])
                seqs_filter.append(seqs[seq_ix])
                labs_filter.append(labs[seq_ix])
            read_PR(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
            read_ROC(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        #print(lines_dict)
        models_line_dicts[model_name.replace("_model_data", "").replace("_data", "")] = lines_dict
    return models_line_dicts

def filter_dict_none(minlen, maxlen):
    models_line_dicts = dict()
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        #print(model_name.replace("_model_data", "").replace("_data", ""))
        lines_dict = dict()
        for v in vals_in_lines:
            lines_dict[v] = []
        for seed_val_ix in range(len(seed_list)):
            seed_val = seed_list[seed_val_ix]
            #print(seed_val)
            for test_num in range(1, 6):
                pred_arr1_filter = []
                pred_arr2_filter = []
                seqs_filter = []
                labs_filter = []
                dir_pred = "../seeds/seed_" + str(seed_val) + "/" + model_name + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num))
                csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
                seqs = list(csv_seed_test_fold["sequence"])
                labs = list(csv_seed_test_fold["label"])
                pred_file = open(dir_pred + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_predictions.txt", "r")
                pred_arrs = pred_file.readlines()
                pred_arr1 = eval(pred_arrs[0])
                pred_arr2 = eval(pred_arrs[1])
                for seq_ix in range(len(seqs)):
                    if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
                        continue
                    pred_arr1_filter.append(pred_arr1[seq_ix])
                    pred_arr2_filter.append(pred_arr2[seq_ix])
                    seqs_filter.append(seqs[seq_ix])
                    labs_filter.append(labs[seq_ix])
                read_PR(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
                read_ROC(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
                #print(lines_dict)
        models_line_dicts[model_name.replace("_model_data", "").replace("_data", "")] = lines_dict
    return models_line_dicts

model_order = {"AP": "AP", "SP": "SP", "AP_SP": "AP-SP", "TSNE_SP": "t-SNE SP", "TSNE_AP_SP": "t-SNE AP-SP"}

def print_dict_seed(mini, maxi, seed):
    dicti = filter_dict_seed(mini, maxi, seed)
    linea = "Metric"
    colnames = ["Metric"]
    for model in model_order:
        for test_num in range(1, 6):
            linea += " & " + model_order[model] + " (test " + str(test_num) + ")"
            colnames.append(model_order[model] + " (test " + str(test_num) + ")")
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    #print(linea + " \\\\ \\hline")
    for metric in dicti["AP"]:
        if "thr" in metric and ")" not in metric:
            continue
        linea = metric.replace(" = ", "")
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for model in model_order:
            ix_test = 1
            for v in dicti[model][metric]:
                rv = 1
                if "Acc" not in metric:
                    rv = 3
                linea += " & " + str(np.round(v, rv))
                if "thr" in metric and ")" not in metric:
                    continue
                dict_csv_data[model_order[model] + " (test " + str(ix_test) + ")"].append(v)
                ix_test += 1
        #print(linea + " \\\\ \\hline")
    if not os.path.isdir("review/seed_" + str(seed)):
        os.makedirs("review/seed_" + str(seed))
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv("review/seed_" + str(seed) +  "/" + str(mini) + "_" + str(maxi) + "_" + str(seed) + ".csv")

def print_dict_seed_all(mini, maxi):
    dicti = filter_dict_none(mini, maxi)
    linea = "Metric"
    colnames = ["Metric"]
    for model in model_order:
        for seed_val in seed_list:
            for test_num in range(1, 6):
                linea += " & " + model_order[model] + " (seed " + str(seed_val) + ") (test " + str(test_num) + ")"
                colnames.append(model_order[model] + " (seed " + str(seed_val) + ") (test " + str(test_num) + ")")
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    #print(linea + " \\\\ \\hline")
    for metric in dicti["AP"]:
        if "thr" in metric and ")" not in metric:
            continue
        linea = metric.replace(" = ", "")
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for model in model_order:
            ix_seed = 0
            ix_test = 1
            for v in dicti[model][metric]:
                rv = 1
                if "Acc" not in metric:
                    rv = 3
                linea += " & " + str(np.round(v, rv))
                if "thr" in metric and ")" not in metric:
                    continue
                dict_csv_data[model_order[model] + " (seed " + str(seed_list[ix_seed]) + ") (test " + str(ix_test) + ")"].append(v)
                if ix_test < 5:
                    ix_test += 1
                else:
                    ix_test = 1
                    ix_seed += 1
        #print(linea + " \\\\ \\hline")
    if not os.path.isdir("review/seed_all"):
        os.makedirs("review/seed_all")
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv("review/seed_all/" + str(mini) + "_" + str(maxi) + "_all.csv")

def print_dict(mini, maxi):
    dicti = filter_dict(mini, maxi)
    linea = "Metric"
    colnames = ["Metric"]
    for model in model_order:
        linea += " & " + model_order[model]
        colnames.append(model_order[model])
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    #print(linea + " \\\\ \\hline")
    for metric in dicti["AP"]:
        if "thr" in metric and ")" not in metric:
            continue
        linea = metric.replace(" = ", "")
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for model in model_order:
            rv = 1
            if "Acc" not in metric:
                rv = 3
            linea += " & " + str(np.round(np.average(dicti[model][metric]), rv))
            dict_csv_data[model_order[model]].append(np.average(dicti[model][metric]))
        #print(linea + " \\\\ \\hline")
    if not os.path.isdir("review/short"):
        os.makedirs("review/short")
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv("review/short/" + str(mini) + "_" + str(maxi) + ".csv")

def print_dict_long(mini, maxi):
    dicti = filter_dict(mini, maxi)
    linea = "Metric"
    colnames = ["Metric"]
    for model in model_order:
        for seed_val in seed_list:
            linea += " & " + model_order[model] + " (seed " + str(seed_val) + ")"
            colnames.append(model_order[model] + " (seed " + str(seed_val) + ")")
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    #print(linea + " \\\\ \\hline")
    for metric in dicti["AP"]:
        if "thr" in metric and ")" not in metric:
            continue
        linea = metric.replace(" = ", "")
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for model in model_order:
            ix_seed = 0
            for v in dicti[model][metric]:
                rv = 1
                if "Acc" not in metric:
                    rv = 3
                linea += " & " + str(np.round(v, rv))
                if "thr" in metric and ")" not in metric:
                    continue
                dict_csv_data[model_order[model] + " (seed " + str(seed_list[ix_seed]) + ")"].append(v)
                ix_seed += 1
        #print(linea + " \\\\ \\hline")
    if not os.path.isdir("review/long"):
        os.makedirs("review/long")
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv("review/long/" + str(mini) + "_" + str(maxi) + ".csv")

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
        all_seeds_test_ok = True
        for seed in seed_list:
            all_test_ok = True
            for tn in range(1, 6):
                lens_seed_test = count_classes_seed_test(seed, tn, a, b)
                if len(lens_seed_test) < 2:
                    all_test_ok = False
                    all_seeds_test_ok = False
            if all_test_ok:
                print_dict_seed(a, b, seed) 
            else:
                print("seed not ok", seed)
        if all_seeds_test_ok:
            print_dict_seed_all(a, b)
        else:
            print("not all ok")
        print_dict(a, b)
        print_dict_long(a, b)
        print(min(lens), a - 1, count_classes(min(lens), a - 1), a, b, count_classes(a, b), b + 1, max(lens), count_classes(b + 1, max(lens)))