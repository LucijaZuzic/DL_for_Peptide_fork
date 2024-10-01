import pandas as pd
import os

data_for_original = pd.read_csv("Sequential_Peptides/mine/all_data.csv")
seq_original = data_for_original["Feature"]
label_original = data_for_original["Label"]
lab_for_seq = dict()
for ix in range(len(seq_original)):
    lab_for_seq[seq_original[ix]] = label_original[ix]

models = []
for m in os.listdir("results_processed_merged_seq_no_val/3_24/"):
    if os.path.isdir("results_processed_merged_seq_no_val/3_24/" + m):
        models.append(m)
print(models)

seeds = []
for s in os.listdir("Sequential_Peptides/mine/"):
    if os.path.isdir("Sequential_Peptides/mine/" + s):
        seeds.append(int(s.replace("seed_", "")))
print(seeds)

seq_seed_test_dict = dict()
for seed in seeds:
    seq_seed_test_dict[seed] = dict()
    for test in range(1, 6):
        seq_seed_test_dict[seed][test] = []
        new_path = "Sequential_Peptides/mine/seed_" + str(seed) + "/test_" + str(test) + "/test_seqs_seed_" + str(seed) + "_test_" + str(test) + ".csv"
        data_for_test = pd.read_csv(new_path, index_col = False)
        seq_seed_test_dict[seed][test] = data_for_test["Feature"]

file_all_original = pd.read_csv("results_processed_merged_seq_no_val/3_24/3_24_all_preds.csv", index_col = False)

dicti_original = dict()
for model in models:
    for seed in seeds:
        colname_preds = file_all_original["preds_" + model + "_" + str(seed)]
        colname_labels = file_all_original["labels_" + model + "_" + str(seed)]
        colname_seqs = file_all_original["feature_" + model + "_" + str(seed)]
        for ix in range(len(colname_seqs)):
            if colname_seqs[ix] not in dicti_original:
                dicti_original[colname_seqs[ix]] = {"Label": colname_labels[ix], "Preds": dict()}
            if model not in dicti_original[colname_seqs[ix]]["Preds"]:
                dicti_original[colname_seqs[ix]]["Preds"][model] = dict()
            dicti_original[colname_seqs[ix]]["Preds"][model][seed] = colname_preds[ix]

file_all_rf = pd.read_csv("results_processed_random_forest/3_24/3_24_all_preds.csv", index_col = False)

for seed in seeds:
    colname_preds = file_all_rf["preds_Random Forest_" + str(seed)]
    colname_labels = file_all_rf["labels_Random Forest_" + str(seed)]
    colname_seqs = file_all_rf["feature_Random Forest_" + str(seed)]
    for ix in range(len(colname_seqs)):
        if colname_seqs[ix] not in dicti_original:
            dicti_original[colname_seqs[ix]] = {"Label": colname_labels[ix], "Preds": dict()}
        if "Random Forest" not in dicti_original[colname_seqs[ix]]["Preds"]:
            dicti_original[colname_seqs[ix]]["Preds"]["Random Forest"] = dict()
        dicti_original[colname_seqs[ix]]["Preds"]["Random Forest"][seed] = colname_preds[ix]

new_files_dict = {"Seed": [], "Test": [], "Sequence": [], "Label": []}
for seq in seq_seed_test_dict[seeds[0]][1]:
    for model in dicti_original[seq]["Preds"]:
        new_files_dict["Pred " + model] = []
    break
for seed in seeds:
    for test in range(1, 6):
        for seq in seq_seed_test_dict[seed][test]:
            new_files_dict["Seed"].append(seed)
            new_files_dict["Test"].append(test)
            new_files_dict["Sequence"].append(seq)
            new_files_dict["Label"].append(lab_for_seq[seq])
            for model in dicti_original[seq]["Preds"]:
                new_files_dict["Pred " + model].append(dicti_original[seq]["Preds"][model][seed])

print(len(new_files_dict["Label"]))

df_new_original = pd.DataFrame(new_files_dict)
df_new_original.to_csv("original_all_models.csv", index = False)

writer = pd.ExcelWriter("Source_Data_ExtendedDataTable1.xlsx", engine = 'openpyxl', mode = "w")
df_new_original.to_excel(writer, sheet_name = "ExtendedDataTable1", index = False)
writer.close()

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

new_files_dict_short = {"Seed": [], "Test": [], "Sequence": [], "Label": []}
for seq in seq_seed_test_dict[seeds[0]][1]:
    for model in dicti_original[seq]["Preds"]:
        new_files_dict_short["Pred " + model] = []
    break

for seed in seeds:
    for test in range(1, 6):
        for seq in seq_seed_test_dict[seed][test]:
            if seq not in new:
                continue
            new_files_dict_short["Seed"].append(seed)
            new_files_dict_short["Test"].append(test)
            new_files_dict_short["Sequence"].append(seq)
            new_files_dict_short["Label"].append(lab_for_seq[seq])
            for model in dicti_original[seq]["Preds"]:
                new_files_dict_short["Pred " + model].append(dicti_original[seq]["Preds"][model][seed])

print(len(new_files_dict_short["Label"]))

df_new_original_short = pd.DataFrame(new_files_dict_short)
df_new_original_short.to_csv("short_all_models.csv", index = False)

file_all_long = pd.read_csv("results_processed_seq_no_test/5_5/5_5_all_preds.csv", index_col = False)

dicti_long = dict()
for model in models:
    colname_preds = file_all_long["preds_" + model]
    colname_labels = file_all_long["labels_" + model]
    colname_seqs = file_all_long["feature_" + model]
    for ix in range(len(colname_seqs)):
        if colname_seqs[ix] not in dicti_long:
            dicti_long[colname_seqs[ix]] = {"Label": colname_labels[ix], "Preds": dict()}
        dicti_long[colname_seqs[ix]]["Preds"][model] = colname_preds[ix]

new_files_dict_long = {"Sequence": [], "Label": []}
for seq in dicti_long:
    for model in dicti_long[seq]["Preds"]:
        new_files_dict_long["Pred " + model] = []
    break

for seq in dicti_long:
    new_files_dict_long["Sequence"].append(seq)
    new_files_dict_long["Label"].append(dicti_long[seq]["Label"])
    for model in dicti_long[seq]["Preds"]:
        new_files_dict_long["Pred " + model].append(dicti_long[seq]["Preds"][model])

print(len(new_files_dict_long["Label"]))

df_new_original_long = pd.DataFrame(new_files_dict_long)
df_new_original_long.to_csv("long_all_models.csv", index = False)

file_all_longest = pd.read_csv("results_processed_seq_long/5_10/5_10_all_preds.csv", index_col = False)

dicti_longest = dict()
for model in models:
    colname_preds = file_all_longest["preds_" + model]
    colname_labels = file_all_longest["labels_" + model]
    colname_seqs = file_all_longest["feature_" + model]
    for ix in range(len(colname_seqs)):
        if colname_seqs[ix] not in dicti_longest:
            dicti_longest[colname_seqs[ix]] = {"Label": colname_labels[ix], "Preds": dict()}
        dicti_longest[colname_seqs[ix]]["Preds"][model] = colname_preds[ix]

new_files_dict_longest = {"Sequence": [], "Label": []}
for seq in dicti_longest:
    for model in dicti_longest[seq]["Preds"]:
        new_files_dict_longest["Pred " + model] = []
    break

for seq in dicti_longest:
    new_files_dict_longest["Sequence"].append(seq)
    new_files_dict_longest["Label"].append(dicti_longest[seq]["Label"])
    for model in dicti_longest[seq]["Preds"]:
        new_files_dict_longest["Pred " + model].append(dicti_longest[seq]["Preds"][model])

print(len(new_files_dict_longest["Label"]))

df_new_original_longest = pd.DataFrame(new_files_dict_longest)
df_new_original_longest.to_csv("longest_all_models.csv", index = False)