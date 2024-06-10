
import pandas as pd

def process_txt(txt_name, all_peps, all_labs, label_use = 1):
    file_txt = open(txt_name, "r")
    lines_txt = file_txt.readlines()
    pep_per_line = dict()
    for line in lines_txt:
        peps_in_line = []
        if "&" in line:
            cells = line.split("&")
            for cell_ix in range(len(cells)):
                if "\\texttt{" in cells[cell_ix]:
                    peps_in_line.append(cells[cell_ix].split("\\texttt{")[-1].split("}")[0])
                else:
                    peps_in_line.append("")
        for ix in range(len(peps_in_line)):
            if ix not in pep_per_line:
                pep_per_line[ix] = []
            pep_per_line[ix].append(peps_in_line[ix])
    all_peps_me = []
    for keyval in pep_per_line:
        df_new = pd.DataFrame()
        pf = []
        pl = []
        for p in pep_per_line[keyval]:
            if len(p) > 0:
                pf.append(p)
                pl.append(label_use)
                all_peps.append(p)
                all_labs.append(label_use)
                all_peps_me.append(p)
        if len(pf) > 0:
            df_new['Feature'] = pf
            df_new['Label'] = pl
            df_new.to_csv(txt_name.replace(".txt", "_" + str(keyval // 2) + ".csv"))
            print(txt_name, keyval, len(pf))
    df_new = pd.DataFrame()
    df_new['Feature'] = all_peps_me
    df_new['Label'] = [label_use for p in all_peps_me]
    df_new.to_csv(txt_name.replace(".txt", "_all.csv"))
    print(txt_name, len(all_peps_me))
    return all_peps, all_labs

all_peps, all_labs = process_txt("text_strong.txt", [], [])
all_peps, all_labs = process_txt("text_low.txt", all_peps, all_labs, 0)
all_peps, all_labs = process_txt("text_experiments.txt", all_peps, all_labs)
df_new = pd.DataFrame()
df_new['Feature'] = all_peps
df_new['Label'] = all_labs
df_new.to_csv("genetic_peptides.csv")
print(len(all_peps))