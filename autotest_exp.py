import configparser
import os
import pickle
import numpy as np
import pipeline

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
        base_path = os.path.join(f"{config['DIRECTORIES']['output_dir']}_{execution}", f"_test_edbt_{os.path.dirname(config['DIRECTORIES']['tables_dir'])}")
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
        print(f"Adding labels to result dfs for execution {execution}")
        for run in result_dfs[execution].keys():
            print(f"Adding labels to result df for run {run}")
            result_df = result_dfs[execution][run]
            conditions = [
                (result_df["predicted"] == 1) & (result_df["label"] == 1),
                (result_df["predicted"] == 1) & (result_df["label"] == 0),
                (result_df["predicted"] == 0) & (result_df["label"] == 0),
                (result_df["predicted"] == 0) & (result_df["label"] == 1),
            ]
            choices = ["TP", "FP", "TN", "FN"]
            result_dfs[execution][run]["result"] = np.select(conditions, choices, default="Unknown")

            result_dfs[execution][run]["auto_test_overwrite"] = (result_df["auto_test_label"] == 1) & (result_df["propagated_label"] == 0)
    return result_dfs


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
    return list(set(execs)), labeling_budgets
"""
config = read_config(config_file_path)


experiment_output_path = get_output_path(0, config)
"""

if __name__ == "__main__":
    config_file_path = './config.ini'
    dataset_num = 96
    config = read_config(config_file_path)

    exec = "Integration_Option_1"
    update_config(config, "AUTO-TEST", "integration_pipeline_option", "1")
    update_config(config, "EXPERIMENTS", "labeling_budget", f"{dataset_num * 2}")
    experiment(exec, config_file_path, config)

    exec = "Integration_Option_1"
    update_config(config, "AUTO-TEST", "integration_pipeline_option", "1")
    update_config(config, "EXPERIMENTS", "labeling_budget", f"{dataset_num * 6}")
    experiment(exec, config_file_path, config)

    exec = "Integration_Option_1"
    update_config(config, "AUTO-TEST", "integration_pipeline_option", "1")
    update_config(config, "EXPERIMENTS", "labeling_budget", f"{dataset_num * 12}")
    experiment(exec, config_file_path, config)

    exec = "Integration_Option_0"
    update_config(config, "AUTO-TEST", "integration_pipeline_option", "0")
    update_config(config, "EXPERIMENTS", "labeling_budget", f"{dataset_num * 2}")
    experiment(exec, config_file_path, config)

    exec = "Integration_Option_0"
    update_config(config, "AUTO-TEST", "integration_pipeline_option", "0")
    update_config(config, "EXPERIMENTS", "labeling_budget", f"{dataset_num * 6}")
    experiment(exec, config_file_path, config)

    exec = "Integration_Option_0"
    update_config(config, "AUTO-TEST", "integration_pipeline_option", "0")
    update_config(config, "EXPERIMENTS", "labeling_budget", f"{dataset_num * 12}")
    experiment(exec, config_file_path, config)
