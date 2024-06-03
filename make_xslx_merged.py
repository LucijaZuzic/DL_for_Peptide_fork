import os
import csv
from xlsxwriter.workbook import Workbook

model_list = ["Bi-LSTM", "LSTM", "MLP", "Transformer", "RNN"]
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
workbook = Workbook('xlsx_version_merged/joined_all.xlsx')
mini = 3
maxi = 24
if not os.path.isdir('xlsx_version_merged/'):
    os.makedirs('xlsx_version_merged/')
for model_name in model_list:
    workbook_model = Workbook('xlsx_version_merged/joined_all_' + model_name + '.xlsx')
    path_csv = 'results_processed_merged_seq/' + str(mini) + "_" + str(maxi) + "/" + model_name + "/" + str(mini) + "_" + str(maxi) + "_" + model_name + ".csv"
    worksheet = workbook.add_worksheet(model_name)
    with open(path_csv, 'rt', encoding='utf8') as f:
        reader = csv.reader(f)
        for r, row in enumerate(reader):
            for c, col in enumerate(row):
                worksheet.write(r, c, col)
workbook.close()