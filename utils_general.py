MAX_LEN = 24
import numpy as np

def data_and_labels_from_indices(all_data, all_labels, indices):
    data = []
    labels = []

    for i in indices:
        data.append(all_data[i])
        labels.append(all_labels[i])

    return data, labels

def save_data_train_val_test(SA_data):
    sequences = []
    labels = []
    for peptide in SA_data:
        if len(peptide) > MAX_LEN or SA_data[peptide] == "-1":
            continue
        sequences.append(peptide)
        labels.append(SA_data[peptide])

    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    SA = []
    NSA = []
    for index in range(len(sequences)): 
        if labels[index] == "1":
            SA.append(sequences[index])
        elif labels[index] == "0":
            NSA.append(sequences[index])

    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA) :] *= 0
    return merged_data, merged_labels