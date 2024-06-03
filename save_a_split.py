


from sklearn.model_selection import StratifiedKFold
from utils_general import save_data_train_val_test, data_and_labels_from_indices
import pandas as pd
import os
import numpy as np
N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
for some_seed in seed_list:
         
        SA_data = np.load("Sequential_Peptides/mine/data_SA_updated.npy", allow_pickle=True).item()

        # Define N-fold cross validation test harness for splitting the test data from the train and validation data
        kfold_first = StratifiedKFold(
            n_splits=N_FOLDS_FIRST, shuffle=True, random_state=some_seed
        )

        # Define N-fold cross validation test harness for splitting the validation from the train data
        kfold_second = StratifiedKFold(
            n_splits=N_FOLDS_SECOND, shuffle=True, random_state=some_seed
        )

        all_data, all_labels = save_data_train_val_test(SA_data)

        test_number = 0

        for train_and_validation_data_indices, test_data_indices in kfold_first.split(
            all_data, all_labels
        ):
            test_number += 1

            if not os.path.isdir('Sequential_Peptides/mine/seed_{}/test_{}/'.format(some_seed, test_number)):
                os.makedirs('Sequential_Peptides/mine/seed_{}/test_{}/'.format(some_seed, test_number))

            # Convert train and validation indices to train and validation data and train and validation labels
            (
                train_and_validation_data,
                train_and_validation_labels,
            ) = data_and_labels_from_indices(
                all_data, all_labels, train_and_validation_data_indices
            )

            # Convert test indices to test data and test labels
            test_data, test_labels = data_and_labels_from_indices(
                all_data, all_labels, test_data_indices
            )

            fold_nr = 0

            for train_data_indices, validation_data_indices in kfold_second.split(
                train_and_validation_data, train_and_validation_labels
            ):
                  
                fold_nr += 1

                # Convert train indices to train data and train labels
                (
                    train_data,
                    train_labels,
                ) = data_and_labels_from_indices(
                    train_and_validation_data,
                    train_and_validation_labels,
                    train_data_indices,
                )

                # Convert validation indices to validation data and validation labels
                (
                    val_data,
                    val_labels,
                ) = data_and_labels_from_indices(
                    train_and_validation_data,
                    train_and_validation_labels,
                    validation_data_indices,
                )

                df_train_save = pd.DataFrame()
                df_train_save['Feature'] = train_data
                df_train_save['Label'] = train_labels
                df_train_save.to_csv('Sequential_Peptides/mine/seed_{}/test_{}/train_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))
                
                df_val_save = pd.DataFrame()
                df_val_save['Feature'] = val_data
                df_val_save['Label'] = val_labels
                df_val_save.to_csv('Sequential_Peptides/mine/seed_{}/test_{}/val_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))

                df_test_save = pd.DataFrame()
                df_test_save['Feature'] = test_data
                df_test_save['Label'] = test_labels
                df_test_save.to_csv('Sequential_Peptides/mine/seed_{}/test_{}/test_seqs_seed_{}_test_{}_fold_{}.csv'.format(some_seed, test_number, some_seed, test_number, fold_nr))