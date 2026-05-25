import configparser
import os
import pickle
import pandas as pd
import re

def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def get_output_path(exec, config, labeling_budget):
    return os.path.join(
        config["DIRECTORIES"]["output_dir"] + f"_{exec}",
        "_"
        + config["EXPERIMENTS"]["exp_name"]
        + "_"
        + config["DIRECTORIES"]["tables_dir"]
        + "_"
        + str(labeling_budget)
        + "_labels",
    )

def get_columns_num(path_to_dataset):
    columns_num = 0
    if not os.path.exists(path_to_dataset):
        return 100
    for table in os.listdir(path_to_dataset):
        # some paths have dirty.csv inside a directory
        table_path = os.path.join(path_to_dataset, table, 'dirty.csv')
        if os.path.exists(table_path) and os.path.isfile(table_path):
            df = pd.read_csv(table_path)
            columns_num += len(df.columns)
    return columns_num

def parse_log_stats(log_path):
    num_quality_folds = 0
    num_overridden_folds = 0
    if not os.path.exists(log_path):
        return None, None

    with open(log_path, 'r') as f:
        for line in f:
            if "get_train_test_sets_per_col: Clusters: [" in line:
                match = re.search(r"Clusters: \[(.*)\]", line)
                if match:
                    clusters_str = match.group(1)
                    if clusters_str.strip():
                        clusters = [c.strip() for c in clusters_str.split(',')]
                        quality_folds = [c for c in clusters if c != '-1' and c != '']
                        num_quality_folds += len(quality_folds)

            if "Variant 2: Overriding cluster user label" in line:
                num_overridden_folds += 1

    return num_quality_folds, num_overridden_folds

def gather_tau_results(tables_dir_override=None):
    """
    Call this function from a Jupyter Notebook to gather all the scores into a Pandas DataFrame.
    """
    config_path = './config.ini'
    config = read_config(config_path)

    if tables_dir_override:
        config.set('DIRECTORIES', 'tables_dir', tables_dir_override)
        dataset_name = os.path.basename(tables_dir_override)
        config.set('DIRECTORIES', 'output_dir', f"tau_exp_output/{dataset_name}")
    else:
        dataset_name = os.path.basename(config.get('DIRECTORIES', 'tables_dir', fallback='DGov_Typo_5'))
        config.set('DIRECTORIES', 'output_dir', f"tau_exp_output/{dataset_name}")

    exp_name = config["EXPERIMENTS"].get("exp_name", "test_edbt")
    taus = [0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.33,0.50]

    dataset_path = os.path.join(config["DIRECTORIES"]["sandbox_dir"], config["DIRECTORIES"]["tables_dir"])
    cols_num = get_columns_num(dataset_path)
    if cols_num == 0:
        cols_num = 100
    multipliers = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

    budgets = []
    unique_multipliers = []
    for m in multipliers:
        b = max(10, round(cols_num * m))
        if b not in budgets:
            budgets.append(b)
            unique_multipliers.append(m)

    results_summary = []

    for idx, budget in enumerate(budgets):
        for tau in taus:
            exec_name = f"tau_{tau}_budget_{budget}"
            out_dir = get_output_path(exec_name, config, budget)

            scores_path = os.path.join(out_dir, "results", "scores_all.pickle")
            if os.path.exists(scores_path):
                with open(scores_path, "rb") as f:
                    scores = pickle.load(f)

                # Parse logs for extra stats
                log_path = os.path.join(out_dir, f"logs_{exp_name}", "app.log")
                num_quality_folds, num_overridden_folds = parse_log_stats(log_path)

                entry = {
                    "dataset": config["DIRECTORIES"]["tables_dir"],
                    "tau": tau,
                    "budget": budget,
                    "budget_multiplier": unique_multipliers[idx],
                    "total_fscore": scores.get("total_fscore", None),
                    "total_precision": scores.get("total_precision", None),
                    "total_recall": scores.get("total_recall", None),
                    "num_quality_folds": num_quality_folds,
                    "num_overridden_folds": num_overridden_folds
                }
                results_summary.append(entry)

    summary_df = pd.DataFrame(results_summary)
    return summary_df

