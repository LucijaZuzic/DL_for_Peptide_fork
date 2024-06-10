
import pandas as pd

def process_txt(txt_name, all_peps, all_labs, label_use = 1):
    file_txt = open(txt_name, "r")
    lines_txt = file_txt.readlines()
    pep_per_line = dict()
    for line in lines_txt:
        peps_in_line = []
        if "\\texttt{" in line:
            new_line = line.split("\\texttt{")
            for pep in new_line:
                if "}" in pep:
                    peps_in_line.append(pep.split("}")[0])
        for ix in range(len(peps_in_line)):
            if ix not in pep_per_line:
                pep_per_line[ix] = []
            pep_per_line[ix].append(peps_in_line[ix])
    all_peps_me = []
    for keyval in pep_per_line:
        df_new = pd.DataFrame()
        df_new['Feature'] = pep_per_line[keyval]
        df_new['Label'] = [label_use for p in pep_per_line[keyval]]
        df_new.to_csv(txt_name.replace(".txt", "_" + str(keyval) + ".csv"))
        for p in pep_per_line[keyval]:
            all_peps.append(p)
            all_labs.append(label_use)
            all_peps_me.append(p)
    df_new = pd.DataFrame()
    df_new['Feature'] = all_peps_me
    df_new['Label'] = [label_use for p in all_peps_me]
    df_new.to_csv(txt_name.replace(".txt", "_all.csv"))
    return all_peps, all_labs

all_peps, all_labs = process_txt("text_strong.txt", [], [])
all_peps, all_labs = process_txt("text_low.txt", all_peps, all_labs, 0)
all_peps, all_labs = process_txt("text_experiments.txt", all_peps, all_labs)
df_new = pd.DataFrame()
df_new['Feature'] = all_peps
df_new['Label'] = all_labs
df_new.to_csv("genetic_peptides.csv")