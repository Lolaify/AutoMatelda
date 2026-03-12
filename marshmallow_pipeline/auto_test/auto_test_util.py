import subprocess
import pandas as pd
import os
import ast
import pickle


def load_autotest_df(auto_test_path: str, sdc_file_name: str, output_path: str, table_file_name_santos: str, rerun: bool):

    auto_test_output_df = _get_autotest_res(auto_test_path, sdc_file_name, output_path, table_file_name_santos, rerun)

    detected_df = _get_detected_cells(auto_test_output_df, output_path, table_file_name_santos, rerun)

    return detected_df

def _get_autotest_res(auto_test_path: str, sdc_file_name: str, output_path: str, table_file_name_santos: str, rerun: bool):
    mediate_file_path = os.path.join(output_path, "mediate_files", "auto_test", f"auto_test_res_{table_file_name_santos}.pickle")

    if not rerun and os.path.exists(mediate_file_path):
        with open(mediate_file_path, "wb+") as handle:
            return pickle.load(handle)

    auto_test_output_df = _run_autotest(auto_test_path, sdc_file_name, output_path, table_file_name_santos)

    with open(mediate_file_path, "wb+") as handle:
        pickle.dump(auto_test_output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return auto_test_output_df


def _run_autotest(auto_test_path: str, sdc_file_name: str, output_path: str, table_file_name_santos: str) -> pd.DataFrame:
    """
    Runs Autotest, returns results as DataFrame
    """
    dirty_file_path = os.path.join(output_path, "aggregated_lake", table_file_name_santos)
    res_file_name = f"{sdc_file_name}_on_{table_file_name_santos}"
    sdc_path = os.path.join("results/SDC")
    result = subprocess.run(
        ['conda', 'run', '-n', 'VENV', 'python3', './online_detect.py', dirty_file_path, sdc_path],
        text=True,
        capture_output=True,
        cwd=auto_test_path
    )
    result.check_returncode()  # raises CalledProcessError if autotest failed
    auto_test_output_path = os.path.join(auto_test_path, "results/detected_outliers", res_file_name)
    auto_test_output_df = pd.read_table(auto_test_output_path, dtype=str)

    return auto_test_output_df

def _get_detected_cells(auto_test_output_df: pd.DataFrame, output_path: str, table_file_name_santos: str, rerun: bool):
    mediate_file_path = os.path.join(output_path, "mediate_files", "auto_test",
                                     f"auto_test_detected_cells_{table_file_name_santos}.pickle")
    if not rerun and os.path.exists(mediate_file_path):
        with open(mediate_file_path, "wb+") as handle:
            return pickle.load(handle)

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

        # Convert error_values from str to list[str]
        if isinstance(error_values, str):
            parsed = ast.literal_eval(error_values)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected a list literal in 'outlier', got: {type(parsed)}: {error_values!r}")
            for item in parsed:
                if not isinstance(item, str):
                    raise ValueError(f"Expected all list elements to be str, got: {type(item)}: {item!r}")
        else:
            raise ValueError(f"Expected 'outlier' to be a list, got: {type(error_values)}: {error_values!r}")


        if column_name in dirty_df.columns:
            for error_value in error_values:
                mask = dirty_df[column_name].fillna('') == str(error_value)
                detected_errors.loc[mask, column_name] = True
        else:
            print(f"Warning: Column '{column_name}' not found in data files.")

    return detected_errors.astype(int)