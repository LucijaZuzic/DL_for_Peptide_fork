import pandas as pd

model_list = ['Transformer', 'RNN', 'LSTM', 'Bi-LSTM', 'MLP']
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

def merge_format_long(dirname, mini, maxi):
    dfdict = dict()
    for model in model_list:
        for seed in seed_list:
            pd_file = pd.read_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_" + str(seed) + "_preds.csv")
            dfdict["preds_" + str(model) + "_" + str(seed)] = pd_file["preds"]
            dfdict["labels_" + str(model) + "_" + str(seed)] = pd_file["labels"]
            dfdict["feature_" + str(model) + "_" + str(seed)] = pd_file["feature"]
    df_new = pd.DataFrame(dfdict)
    df_new.to_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_preds.csv")

def merge_format(dirname, mini, maxi):
    dfdict = dict()
    for model in model_list:
        pd_file = pd.read_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_preds.csv")
        dfdict["preds_" + str(model)] = pd_file["preds"]
        dfdict["labels_" + str(model)] = pd_file["labels"]
        dfdict["feature_" + str(model)] = pd_file["feature"]
    df_new = pd.DataFrame(dfdict)
    df_new.to_csv(dirname + "/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_preds.csv")

merge_format_long("results_processed_merged_seq_no_val", 3, 24)
merge_format_long("results_processed_20_seq_no_val", 5, 5)
merge_format("results_processed_seq_no_test", 5, 5)
merge_format("results_processed_seq_long", 5, 10)

merge_format("results_processed_seq_genetic_all", 6, 10)

merge_format("results_processed_seq_genetic_low", 6, 10)
merge_format("results_processed_seq_genetic_low_0", 6, 6)
merge_format("results_processed_seq_genetic_low_1", 10, 10)

merge_format("results_processed_seq_genetic_strong", 6, 10)
merge_format("results_processed_seq_genetic_strong_0", 6, 6)
merge_format("results_processed_seq_genetic_strong_1", 7, 10)

merge_format("results_processed_seq_genetic_experiments_all", 9, 10)
merge_format("results_processed_seq_genetic_experimentsA", 9, 10)
merge_format("results_processed_seq_genetic_experimentsB", 9, 10)
merge_format("results_processed_seq_genetic_experimentsC", 9, 10)