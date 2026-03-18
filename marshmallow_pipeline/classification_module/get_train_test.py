import copy
import logging
import pickle
import time
from marshmallow_pipeline.classification_module.classifier import classify
import pandas as pd


def get_train_test_sets(X_temp, y_temp, samples_dict, cell_clustering_df):
    logging.debug("Train-Test set preparation")
    cells_per_cluster = cell_clustering_df["cells_per_cluster"].values[0]
    samples_df = pd.DataFrame(samples_dict)
    X_train, y_train, X_test, y_test, y_cell_ids = [], [], [], [], []
    clusters = samples_df["cell_cluster"].unique().tolist()
    clusters.sort()
    for key in clusters:
        try:
            if key == -1:
                continue
            cell_cluster_samples = samples_df[samples_df["cell_cluster"] == key][
                "samples_indices_global"
            ].values[0]
            cell_cluster_final_label = samples_df[samples_df["cell_cluster"] == key][
                "final_label_to_be_propagated"
            ].values[0]
            if len(cell_cluster_samples) == 0:
                for cell in cells_per_cluster[key]:
                    X_test.append(X_temp[cell])
                    y_test.append(y_temp[cell])
                    y_cell_ids.append(cell)
            else:
                for cell in cells_per_cluster[key]:
                    X_train.append(X_temp[cell])
                    if cell in cell_cluster_samples:
                        y_train.append(y_temp[cell])
                    else:
                        y_train.append(cell_cluster_final_label)
                    X_test.append(X_temp[cell])
                    y_test.append(y_temp[cell])
                    y_cell_ids.append(cell)
        except Exception as e:
            logging.error("Error in get_train_test_sets: %s", e)

    logging.debug("Length of X_train: %s", len(X_train))
    return X_train, y_train, X_test, y_test, y_cell_ids

def get_train_test_sets_per_col(X_temp, y_temp, auto_test_labels, samples_dict, cell_clustering_df, uids, output_path):
    logging.debug("Train-Test set preparation")
    cells_per_cluster = cell_clustering_df["cells_per_cluster"].values[0]
    samples_df = pd.DataFrame(samples_dict)
    uids_per_col = {}
    cols_of_uids = {}
    for uid in uids:
        if (uid[0], uid[1]) not in uids_per_col:
            uids_per_col[(uid[0], uid[1])] = {uids[uid]: uid}
        else:
            uids_per_col[(uid[0], uid[1])][uids[uid]] = uid
        cols_of_uids[uids[uid]] = (uid[0], uid[1])
    X_train_cols = {}
    y_train_cols = {}
    X_test_cols = {}
    y_test_cols = {}
    y_cell_ids_cols = {}
    predicted_cols = {}
    X_train, y_train, X_test, y_test, y_cell_ids, predicted = [], [], [], [], [], []    
    clusters = samples_df["cell_cluster"].unique().tolist()
    logging.debug("Clusters: %s", clusters)
    count_extra_labels_due_to_auto_test = 0
    count_propagation_and_auto_test_agreements = 0
    count_total_dirty_labels = 0
    count_total_clean_labels = 0
    s_time = time.time()
    for key in clusters:
        try:
            if key == -1:
                continue
            cell_cluster_samples = samples_df[samples_df["cell_cluster"] == key][
                "samples_indices_global"
            ].values[0]
            cell_cluster_final_label = samples_df[samples_df["cell_cluster"] == key][
                "final_label_to_be_propagated"
            ].values[0]
            if len(cell_cluster_samples) == 0:
                for cell in cells_per_cluster[key]:
                    cell_col = cols_of_uids[cell]
                    if cell_col not in X_test_cols:
                        X_test_cols[cell_col] = [X_temp[cell]]
                        y_test_cols[cell_col] = [y_temp[cell]]
                        y_cell_ids_cols[cell_col] = [cell]
                    else:
                        X_test_cols[cell_col].append(X_temp[cell])
                        y_test_cols[cell_col].append(y_temp[cell])
                        y_cell_ids_cols[cell_col].append(cell)
            else:
                for cell in cells_per_cluster[key]:
                    cell_col = cols_of_uids[cell]
                    if cell_col not in X_train_cols:
                        X_train_cols[cell_col] = [X_temp[cell]]
                    else:
                        X_train_cols[cell_col].append(X_temp[cell])
                    if cell in cell_cluster_samples:
                        if cell_col not in y_train_cols:
                            y_train_cols[cell_col] = [y_temp[cell]]
                        else:
                            y_train_cols[cell_col].append(y_temp[cell])
                    else:
                        auto_test_label = auto_test_labels[cell]
                        
                        # AUTO-TEST OVERRIDE LOGIC:
                        # If auto-test detected an error (1) but cluster consensus says clean (0),
                        # trust auto-test and label as error. This prevents missing errors.
                        if auto_test_label == 1 and cell_cluster_final_label == 0:
                            label_to_use = 1  # Override with error label
                            count_extra_labels_due_to_auto_test += 1
                        else:
                            if auto_test_label == cell_cluster_final_label and auto_test_label == 1:
                                count_propagation_and_auto_test_agreements += 1
                            label_to_use = cell_cluster_final_label  # Use cluster consensus
                        if label_to_use == 0:
                            count_total_clean_labels += 1
                        else:
                            count_total_dirty_labels += 1
                        # Add propagated label (or auto-test override) to training set
                        if cell_col not in y_train_cols:
                            y_train_cols[cell_col] = [label_to_use]
                        else:
                            y_train_cols[cell_col].append(label_to_use)
                    if cell_col not in X_test_cols:
                        X_test_cols[cell_col] = [X_temp[cell]]
                        y_test_cols[cell_col] = [y_temp[cell]]
                        y_cell_ids_cols[cell_col] = [cell]
                    else:
                        X_test_cols[cell_col].append(X_temp[cell])
                        y_test_cols[cell_col].append(y_temp[cell])
                        y_cell_ids_cols[cell_col].append(cell)
        except Exception as e:
            logging.error("Error in get_train_test_sets: %s", e)
    logging.info("Extra labels added due to Auto-Test overrides: %s", count_extra_labels_due_to_auto_test)
    logging.info("Agreements between cluster consensus and Auto-Test: %s", count_propagation_and_auto_test_agreements)
    logging.info("Total clean labels in training set: %s", count_total_clean_labels)
    logging.info("Total dirty labels in training set: %s", count_total_dirty_labels)
    logging.debug("*******Time for train-test set preparation: %s", time.time() - s_time)
    s_time = time.time()
    logging.debug("Start classification Per Column")
    for col in X_train_cols:
        gbc, predicted_cols[col] = classify(X_train_cols[col], y_train_cols[col], X_test_cols[col])
        if gbc is not None:
            feature_importances = gbc.feature_importances_
            with open(f"{output_path}/feature_importances_{col}.pickle", "wb") as f:
                pickle.dump(feature_importances, f)
    logging.debug("End classification Per Column")
    logging.debug("*******Time for classification Per Column: %s", time.time() - s_time)
    for col in predicted_cols:
        for i in range(len(predicted_cols[col])):
            X_test.append(X_test_cols[col][i])
            y_test.append(y_test_cols[col][i])
            y_cell_ids.append(y_cell_ids_cols[col][i])
            y_train.append(predicted_cols[col][i])
            X_train.append(X_train_cols[col][i])
            predicted.append(predicted_cols[col][i])
    logging.debug("Length of X_train: %s", len(X_train))
    return X_train, y_train, X_test, y_test, y_cell_ids, predicted
