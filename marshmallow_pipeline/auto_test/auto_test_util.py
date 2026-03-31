import subprocess
import pandas as pd
import os
import ast
import pickle
import logging

def load_autotest_df(auto_test_config, output_path: str, table_file_name_santos: str):

    if not auto_test_config["rerun"]:
        detected_df = _load_from_pickle(f"auto_test_detected_cells_{table_file_name_santos}.pickle", output_path)
        if detected_df is not None:
            return detected_df

    auto_test_output_df = _get_autotest_res(auto_test_config, output_path, table_file_name_santos)

    detected_df = _get_detected_cells(auto_test_output_df, output_path, table_file_name_santos)

    return detected_df

def _get_autotest_res(auto_test_config, output_path: str, table_file_name_santos: str):
    mediate_file_path = os.path.join(output_path, "mediate_files", "auto_test", f"auto_test_res_{table_file_name_santos}.pickle")

    if not auto_test_config["rerun"]:
        auto_test_output_df = _load_from_pickle(f"auto_test_res_{table_file_name_santos}.pickle", output_path)
        if auto_test_output_df is not None:
            return auto_test_output_df

    logging.debug("scanning %s with Auto-Test", table_file_name_santos)
    auto_test_output_df = _run_autotest(auto_test_config["auto_test_path"], auto_test_config["sdc_file_name"], output_path, table_file_name_santos)
    os.makedirs(os.path.dirname(mediate_file_path), exist_ok=True)
    with open(mediate_file_path, "wb+") as handle:
        pickle.dump(auto_test_output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return auto_test_output_df


def _run_autotest(auto_test_path: str, sdc_file_name: str, output_path: str, table_file_name_santos: str) -> pd.DataFrame:
    """
    Runs Autotest, returns results as DataFrame
    """
    dirty_file_path = os.path.join(os.getcwd(), output_path, "aggregated_lake", table_file_name_santos)
    res_file_name = f"{os.path.splitext(os.path.basename(sdc_file_name))[0]}_on_{table_file_name_santos}"
    auto_test_output_path = os.path.join(auto_test_path, "results/detected_outliers", res_file_name)
    # If Auto-Test has already scanned this dataset, load results from disk instead of running Auto-Test again
    if not os.path.exists(auto_test_output_path):
        sdc_path = os.path.join("results/SDC", sdc_file_name)
        result = subprocess.run(
            ['conda', 'run', '-n', 'VENV', 'python3', './online_detect.py', dirty_file_path, sdc_path],
            text=True,
            capture_output=True,
            cwd=auto_test_path
        )
        result.check_returncode()  # raises CalledProcessError if autotest failed
    if os.path.exists(auto_test_output_path):
        auto_test_output_df = pd.read_table(auto_test_output_path, dtype=str)
        if all(auto_test_output_df.columns == ',header,outlier,conf,dist_val,SDC') :
            auto_test_output_df = pd.read_csv(auto_test_output_path, dtype=str)
        if all(auto_test_output_df.columns == 'header,outlier'):
            auto_test_output_df = pd.DataFrame(columns=['header', 'outlier', 'conf', 'dist_val', 'SDC'])
    else:
        auto_test_output_df = pd.DataFrame(columns=['header', 'outlier', 'conf', 'dist_val', 'SDC'])
    return auto_test_output_df

def _get_detected_cells(auto_test_output_df: pd.DataFrame, output_path: str, table_file_name_santos: str):
    mediate_file_path = os.path.join(output_path, "mediate_files", "auto_test",
                                     f"auto_test_detected_cells_{table_file_name_santos}.pickle")

    dirty_file_path = os.path.join(output_path, "aggregated_lake", table_file_name_santos)
    dirty_df = pd.read_csv(dirty_file_path, dtype=str)
    detected_errors = _mark_detected_cells(auto_test_output_df, dirty_df)

    with open(mediate_file_path, "wb+") as handle:
        pickle.dump(detected_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return detected_errors

def _mark_detected_cells(auto_test_output_df: pd.DataFrame, dirty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates df with the same shape of dirty_df where every cell that was detected by Auto-Test is a 1 and every other cell is 0
    """
    detected_errors = pd.DataFrame(False, index=dirty_df.index, columns=dirty_df.columns)

    for _, row in auto_test_output_df.iterrows():
        if 'header' not in row or 'outlier' not in row:
            continue

        column_name = row['header']
        error_values = row['outlier']

        if pd.isna(column_name) or pd.isna(error_values):
            continue
        if isinstance(error_values, tuple):
            continue
        # Convert error_values from str to list[str]
        if isinstance(error_values, str):
            parsed = ast.literal_eval(error_values)
            if not isinstance(parsed, list):
                logging.warning(f"Expected a list literal in 'outlier', got: {type(parsed)}: {error_values!r}")
                continue
            for item in parsed:
                if not isinstance(item, str):
                    logging.warning(f"Expected all list elements to be str, got: {type(item)}: {item!r}")
                    continue
        else:
            logging.warning(f"Expected 'outlier' to be a str, got: {type(error_values)}: {error_values!r}")
            continue


        if column_name in dirty_df.columns:
            for error_value in parsed:
                mask = dirty_df[column_name].fillna('') == str(error_value)
                detected_errors.loc[mask, column_name] = True
        else:
            print(f"Warning: Column '{column_name}' not found in data files.")

    return detected_errors.astype(int)

def _load_from_pickle(file_name: str, output_path: str):
    file_path = os.path.join(output_path, "mediate_files", "auto_test", file_name)
    if os.path.exists(file_path):
        logging.debug("loading %s from disk", file_name)
        with open(file_path, "rb") as handle:
            return pickle.load(handle)
    # Check if Matelda has scanned this dataset with a different amount of labels and get Auto-Test results from there
    logging.debug("%s not found in mediate files, checking if Matelda scanned this Dataset with a different amount of labels", file_name)
    folders = os.listdir(os.path.dirname(output_path))
    if len(folders) > 1:
        for folder in folders:
            file_path = os.path.join(os.path.dirname(output_path), folder, "mediate_files", "auto_test", file_name)
            if os.path.exists(file_path):
                logging.debug("Found %s in folder %s, loading from disk..", file_name, folder)
                with open(file_path, "rb") as handle:
                    return pickle.load(handle)
    return None