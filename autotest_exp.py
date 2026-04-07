import configparser
import os
import pickle
import numpy as np
import pipeline
import pandas as pd
import re

def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def update_config(config, section, parameter, new_value):
    config.set(section, parameter, new_value)

def save_config(config, file_path):
    with open(file_path, 'w') as configfile:
        config.write(configfile)

def get_output_path(exec, config):
    return os.path.join(
        config["DIRECTORIES"]["output_dir"] + f"_{exec}",
        "_"
        + config["EXPERIMENTS"]["exp_name"]
        + "_"
        + config["DIRECTORIES"]["tables_dir"]
        + "_"
        + str(int(config["EXPERIMENTS"]["labeling_budget"]))
        + "_labels",
    )

def load_experiment_result(executions: list[str], config):
    result_dfs = {}
    results_per_table = {}
    for execution in executions:
        base_path = os.path.join(f"{config['DIRECTORIES']['output_dir']}_{execution}")
        if f"_test_edbt_{os.path.dirname(config['DIRECTORIES']['tables_dir'])}" in os.listdir(base_path):
            base_path = os.path.join(base_path, f"_test_edbt_{os.path.dirname(config['DIRECTORIES']['tables_dir'])}")
        result_dfs[execution] = {}
        results_per_table[execution] = {}
        for run in os.listdir(base_path):
            result_df_path = os.path.join(base_path, run, "results", "results_df.pickle")
            if os.path.exists(result_df_path):
                with open(result_df_path, "rb") as f:
                    result_dfs[execution][run] = pickle.load(f)
            results_per_table_path = os.path.join(base_path, run, "results", "results_per_table.pickle")
            if os.path.exists(results_per_table_path):
                with open(results_per_table_path, "rb") as f:
                    results_per_table[execution][run] = pickle.load(f)
    return result_dfs, results_per_table

def add_labels_to_result_dfs(result_dfs):
    for execution in result_dfs.keys():
        for run in result_dfs[execution].keys():
            result_df = result_dfs[execution][run]
            result_df = _add_result_label(result_df, "predicted", "result")
            result_df["auto_test_overwrite"] = (result_df["auto_test_label"] == 1) & (result_df["propagated_label"] == 0)
            result_dfs[execution][run] = result_df
    return result_dfs

def _add_result_label(result_df, predicted, result):
    conditions = [
        (result_df[predicted] == 1) & (result_df["label"] == 1),
        (result_df[predicted] == 1) & (result_df["label"] == 0),
        (result_df[predicted] == 0) & (result_df["label"] == 0),
        (result_df[predicted] == 0) & (result_df["label"] == 1),
    ]
    choices = ["TP", "FP", "TN", "FN"]
    result_df[result] = np.select(conditions, choices, default="Unknown")
    return result_df

def add_training_labels_to_result_dfs(result_dfs):
    # Add Training Labels to result_dfs:
    for execution in result_dfs.keys():
        for run in result_dfs[execution].keys():
            df = result_dfs[execution][run]
            if execution == "Integration_Option_0":
                df['training_label'] = df['propagated_label']
            elif execution in ["Integration_Option_1", "Integration_Option_3"]:
                df['training_label'] = (df['propagated_label'] == 1) | (
                            df['auto_test_label'] == 1)
                df['training_label'] = df['training_label'].astype(int)
            if 'training_label' in df:
                result_dfs[execution][run] = _add_result_label(df, "training_label", "training_result")
    return result_dfs

def add_labels_to_results_per_table(results_per_table):
    for execution in results_per_table.keys():
        for run in results_per_table[execution].keys():
            df = pd.DataFrame(results_per_table[execution][run]).T
            df["total_cells"] = df["tp"] + df["fp"] + df["fn"] + df["tn"]
            results_per_table[execution][run] = df.T
    return results_per_table

def get_analysis_df(result_dfs, to_analyse):
    all_analysis = []
    precision, recall, f_score = {}, {}, {}
    for execution in result_dfs.keys():
        for run in result_dfs[execution].keys():
            result_df = result_dfs[execution][run]
            value_counts = result_df[to_analyse].value_counts()
            analysis = pd.Series()
            analysis['execution'] = execution
            analysis['run'] = run
            analysis['labels'] = int(re.findall(r'\d+', run)[0])
            analysis["TP"] = value_counts["TP"] if "TP" in value_counts else 0
            analysis["FP"] = value_counts["FP"] if "FP" in value_counts else 0
            analysis["TN"] = value_counts["TN"] if "TN" in value_counts else 0
            analysis["FN"] = value_counts["FN"] if "FN" in value_counts else 0

            analysis["precision"] = analysis["TP"] / (analysis["TP"] + analysis["FP"])
            analysis["recall"] = analysis["TP"] / (analysis["TP"] + analysis["FN"])
            analysis["f_score"] = 2 * (analysis["precision"] * analysis["recall"]) / (
                        analysis["precision"] + analysis["recall"])

            all_analysis.append(analysis)

    return pd.DataFrame(all_analysis)

def experiment(execution, config_file_path, config):
    result_df_path = os.path.join(
        get_output_path(execution, config),
        "results/results_df.pickle")

    if not os.path.exists(result_df_path):
        old_config = read_config(config_file_path)
        save_config(config, config_file_path)
        pipeline.main(execution)
        save_config(old_config, config_file_path)

    with open(result_df_path, "rb") as f:
        output = pickle.load(f)
    return output

def experiments(pipeline_options, labeling_budget_multipliers, config_file_path):
    config = read_config(config_file_path)
    path_to_dataset = os.path.join(config["DIRECTORIES"]["sandbox_dir"], config["DIRECTORIES"]["tables_dir"])
    print(f"Running experiments on dataset at {path_to_dataset}")
    dataset_num = len(os.listdir(path_to_dataset))
    labeling_budgets = [dataset_num * multiplier for multiplier in labeling_budget_multipliers]
    execs = []
    for pipeline_option in pipeline_options:
        for labeling_budget in labeling_budgets:
            print(f"Running experiment with pipeline option {pipeline_option} and labeling budget {labeling_budget}")
            exec = f"Integration_Option_{pipeline_option}"
            execs.append(exec)
            update_config(config, "AUTO-TEST", "integration_pipeline_option", str(pipeline_option))
            update_config(config, "EXPERIMENTS", "labeling_budget", str(labeling_budget))
            experiment(exec, config_file_path, config)
    return sorted(list(set(execs))), labeling_budgets
"""
config = read_config(config_file_path)


experiment_output_path = get_output_path(0, config)
"""

if __name__ == "__main__":
    config_file_path = './config.ini'

    pipeline_options = [0, 1, 3]
    label_multiplier = [2, 4, 6, 8, 12, 24]
    config = read_config(config_file_path)
    executions, runs = experiments(pipeline_options, label_multiplier, config_file_path)
