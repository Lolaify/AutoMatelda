import configparser
import os
import pickle

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

"""
config = read_config(config_file_path)


experiment_output_path = get_output_path(0, config)
"""