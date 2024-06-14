import pandas as pd
import numpy as np

model_list = ['Transformer', 'RNN', 'LSTM', 'Bi-LSTM', 'MLP']
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

def print_latex_format_long(path_to_file):
    pd_file = pd.read_csv(path_to_file)
    rows_string = "Metric"
    for model in model_list:
        for seed in seed_list:
            rows_string += " & " + model + " (seed " + str(seed) + ")"
    rows_string += "\\\\ \\hline\n"

    metric_ord = list(range(len(pd_file["Metric"])))

    for ix in metric_ord:
        rows_string_one = pd_file["Metric"][ix]
        max_row = -1
        max_part = "be"
        for model in model_list:
            for seed in seed_list:
                round_val = 3
                if "Acc" in pd_file["Metric"][ix]:
                    round_val = 1
                part = str(np.round(pd_file[model + " (seed " + str(seed) + ")"][ix], round_val))
                if "Acc" in pd_file["Metric"][ix]:
                    part += "\\%"
                rows_string_one += " & " + part
                if pd_file[model + " (seed " + str(seed) + ")"][ix] > max_row:
                    max_row = pd_file[model + " (seed " + str(seed) + ")"][ix]
                    max_part = part
        rows_string_one += " \\\\ \\hline\n"
        rows_string += rows_string_one.replace(max_part, "\\textbf{" + max_part + "}")
    print(rows_string)

def print_latex_format(path_to_file):
    pd_file = pd.read_csv(path_to_file)
    rows_string = "Metric"
    for model in model_list:
        rows_string += " & " + model
    rows_string += "\\\\ \\hline\n"

    metric_ord = list(range(len(pd_file["Metric"])))

    new_csv = {"Metric": []}
    for ix in metric_ord:
        new_csv["Metric"].append(pd_file["Metric"][ix] + " (avg.)")
        new_csv["Metric"].append(pd_file["Metric"][ix] + " (std.)")
    for ix in metric_ord:
        rows_string_one = pd_file["Metric"][ix]
        max_row = -1
        max_part = "be"
        for model in model_list:
            if model not in new_csv:
                new_csv[model] = []
            vals_use = []
            for seed in seed_list:
                vals_use.append(pd_file[model + " (seed " + str(seed) + ")"][ix])
            round_val = 3
            if "Acc" in pd_file["Metric"][ix]:
                round_val = 1
            new_csv[model].append(np.average(vals_use))
            new_csv[model].append(np.std(vals_use))
            part_avg = str(np.round(np.average(vals_use), round_val))
            part_std = str(np.round(np.std(vals_use), round_val))
            if "Acc" in pd_file["Metric"][ix]:
                part_avg += "\\%"
                part_std += "\\%"
            part = part_avg + " (" + part_std + ")"
            rows_string_one += " & " + part
            if np.average(vals_use) > max_row:
                max_row = np.average(vals_use)
                max_part = part
        rows_string_one += " \\\\ \\hline\n"
        rows_string += rows_string_one.replace(max_part, "\\textbf{" + max_part + "}")
    new_df = pd.DataFrame(new_csv)
    new_df.to_csv(path_to_file.replace(".csv", "_avg_std.csv"))
    print(rows_string)

print_latex_format("results_processed_merged_seq_no_val/3_24/3_24.csv")
print_latex_format("results_processed_20_seq_no_val/5_5/5_5.csv")

model_list = ['Random Forest']
print_latex_format("results_processed_random_forest/3_24/3_24.csv")
print_latex_format("results_processed_random_forest_20/5_5/5_5.csv")