import pandas as pd
import numpy as np

model_list = ['Transformer', 'RNN', 'LSTM', 'Bi-LSTM', 'MLP']

def print_latex_format(path_to_file):
    pd_file = pd.read_csv(path_to_file)
    rows_string = "Metric"
    for model in model_list:
        rows_string += " & " + model
    rows_string += "\\\\ \\hline\n"

    metric_ord = list(range(len(pd_file["Metric"])))

    for ix in metric_ord:
        rows_string_one = pd_file["Metric"][ix]
        max_row = -1
        max_part = "be"
        for model in model_list:
            round_val = 3
            if "Acc" in pd_file["Metric"][ix]:
                round_val = 1
            part = str(np.round(pd_file[model][ix], round_val))
            if "Acc" in pd_file["Metric"][ix]:
                part += "\\%"
            rows_string_one += " & " + part
            if pd_file[model][ix] > max_row:
                max_row = pd_file[model][ix]
                max_part = part
        rows_string_one += " \\\\ \\hline\n"
        rows_string += rows_string_one.replace(max_part, "\\textbf{" + max_part + "}")
    print(rows_string)

print_latex_format("results_processed_merged_seq_no_val/a_result.csv")
print_latex_format("results_processed_20_seq_no_val/a_result.csv")
print_latex_format("results_processed_seq_no_test/5_5/5_5.csv")
print_latex_format("results_processed_seq_long/5_10/5_10.csv")