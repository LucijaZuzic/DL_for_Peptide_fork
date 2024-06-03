import os
import csv
from xlsxwriter.workbook import Workbook

model_list = ["Bi-LSTM", "LSTM", "MLP", "Transformer", "RNN"]
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
workbook = Workbook('xlsx_version/joined_all.xlsx')
mini = 3
maxi = 24
if not os.path.isdir('xlsx_version/'):
    os.makedirs('xlsx_version/')
for model_name in model_list:
    workbook_model = Workbook('xlsx_version/joined_all_' + model_name + '.xlsx')
    for seed_val in seed_list:
        path_csv = 'results_processed_seq/' + str(mini) + "_" + str(maxi) + "/" + model_name + "/seed_" + str(seed_val) + "/" + str(mini) + "_" + str(maxi) + "_" + model_name + "_seed_" + str(seed_val) + ".csv"
        worksheet = workbook.add_worksheet(model_name + " " + str(seed_val))
        worksheet_model = workbook_model.add_worksheet(model_name + " " + str(seed_val))
        with open(path_csv, 'rt', encoding='utf8') as f:
            reader = csv.reader(f)
            for r, row in enumerate(reader):
                for c, col in enumerate(row):
                    worksheet.write(r, c, col)
                    worksheet_model.write(r, c, col)
    workbook_model.close()
workbook.close()

for seed_val in seed_list:
    workbook_seed = Workbook('xlsx_version/joined_all_' + str(seed_val) + '.xlsx')
    for model_name in model_list:
        path_csv = 'results_processed_seq/' + str(mini) + "_" + str(maxi) + "/" + model_name + "/seed_" + str(seed_val) + "/" + str(mini) + "_" + str(maxi) + "_" + model_name + "_seed_" + str(seed_val) + ".csv"
        worksheet_seed = workbook_seed.add_worksheet(model_name + " " + str(seed_val))
        with open(path_csv, 'rt', encoding='utf8') as f:
            reader = csv.reader(f)
            for r, row in enumerate(reader):
                for c, col in enumerate(row):
                    worksheet_seed.write(r, c, col)
    workbook_seed.close()