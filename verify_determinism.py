#!/usr/bin/env python3
"""
Script to verify determinism by running the pipeline twice and comparing results.
"""
import os
import pickle
import configparser
import shutil

import pipeline


def read_config(file_path):
    """Read configuration file."""
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def update_config(config, section, parameter, new_value):
    """Update configuration parameter."""
    config.set(section, parameter, new_value)


def save_config(config, file_path):
    """Save configuration file."""
    with open(file_path, 'w') as configfile:
        config.write(configfile)


def get_output_path(exec_name, config):
    """Get the output path for an execution."""
    return os.path.join(
        config["DIRECTORIES"]["output_dir"] + f"_{exec_name}",
        "_"
        + config["EXPERIMENTS"]["exp_name"]
        + "_"
        + config["DIRECTORIES"]["tables_dir"]
        + "_"
        + str(int(config["EXPERIMENTS"]["labeling_budget"]))
        + "_labels",
    )


def load_results_df(output_path):
    """Load results_df.pickle from output path."""
    result_df_path = os.path.join(output_path, "results", "results_df.pickle")
    if not os.path.exists(result_df_path):
        return None
    with open(result_df_path, "rb") as f:
        return pickle.load(f)


def compare_dataframes(df1, df2, tolerance=1e-9):
    """
    Compare two dataframes for equality.
    Returns (is_equal, differences_report)
    """
    if df1 is None or df2 is None:
        return False, "One or both DataFrames are None"

    if df1.shape != df2.shape:
        return False, f"Shape mismatch: {df1.shape} vs {df2.shape}"

    if not df1.columns.equals(df2.columns):
        return False, f"Column mismatch: {df1.columns.tolist()} vs {df2.columns.tolist()}"

    # Check for differences
    differences = []

    for col in df1.columns:
        if df1[col].dtype == 'object':
            # String or object columns - exact match
            if not df1[col].equals(df2[col]):
                diff_rows = (df1[col] != df2[col]).sum()
                differences.append(f"  - Column '{col}': {diff_rows} rows differ (object type)")
        else:
            # Numeric columns - check with tolerance
            try:
                if not df1[col].equals(df2[col]):
                    # Try numeric comparison with tolerance
                    max_diff = (df1[col] - df2[col]).abs().max()
                    if max_diff > tolerance:
                        differences.append(f"  - Column '{col}': max difference = {max_diff}")
            except (TypeError, ValueError):
                # Fall back to exact match for problematic columns
                if not df1[col].equals(df2[col]):
                    diff_rows = (df1[col] != df2[col]).sum()
                    differences.append(f"  - Column '{col}': {diff_rows} rows differ")

    if differences:
        report = "Differences found:\n" + "\n".join(differences)
        return False, report

    return True, "DataFrames are identical"


def cleanup_output(config, exec_name):
    """Clean up output directory before running."""
    output_dir = config["DIRECTORIES"]["output_dir"] + f"_{exec_name}"
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)


def main():
    """Main verification script."""
    print("=" * 80)
    print("DETERMINISM VERIFICATION SCRIPT")
    print("=" * 80)
    print()

    config_file_path = './config.ini'

    # Read configuration
    original_config = read_config(config_file_path)
    config = read_config(config_file_path)

    # Get current random_seed setting
    random_seed = config.get("EXPERIMENTS", "random_seed", fallback="0")
    print(f"Current random_seed setting: {random_seed}")
    if random_seed == "0":
        print("⚠️  WARNING: random_seed is set to 0 (non-deterministic mode)")
        print("   For determinism testing, please set random_seed to a positive value (e.g., 42)")
    else:
        print(f"✓ Running in deterministic mode with seed={random_seed}")
    print()

    # Set dataset
    dataset_name = "DGov_Typo_subsets/DGov_Typo_high_TP_ratio"
    execution_name_base = "Integration_Option_0"
    execution_name_1 = f"{execution_name_base}_Run1"
    execution_name_2 = f"{execution_name_base}_Run2"
    labeling_budget = 50  # Use a reasonable budget for testing

    print(f"Dataset: {dataset_name}")
    print(f"Execution (Run 1): {execution_name_1}")
    print(f"Execution (Run 2): {execution_name_2}")
    print(f"Labeling Budget: {labeling_budget}")
    print()

    # Verify dataset exists
    sandbox_path = config["DIRECTORIES"]["sandbox_dir"]
    dataset_path = os.path.join(sandbox_path, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return False
    print(f"✓ Dataset found at {dataset_path}")
    print()

    # Configure for this dataset
    update_config(config, 'DIRECTORIES', 'tables_dir', dataset_name)
    update_config(config, 'DIRECTORIES', 'output_dir', 'output/verify_determinism')
    update_config(config, 'EXPERIMENTS', 'labeling_budget', str(labeling_budget))
    update_config(config, 'AUTO-TEST', 'integration_pipeline_option', '0')

    print("RUN 1: Running pipeline first time...")
    print("-" * 80)
    cleanup_output(config, execution_name_1)
    save_config(config, config_file_path)
    try:
        pipeline.main(execution_name_1)
    except Exception as e:
        print(f"ERROR in Run 1: {e}")
        save_config(original_config, config_file_path)
        return False

    # Get output path and load results
    output_path_1 = get_output_path(execution_name_1, config)
    results_df_1 = load_results_df(output_path_1)

    if results_df_1 is None:
        print(f"ERROR: Could not load results_df from Run 1")
        save_config(original_config, config_file_path)
        return False

    print(f"✓ Run 1 completed. Loaded {len(results_df_1)} rows")
    print()

    print("RUN 2: Running pipeline second time...")
    print("-" * 80)
    cleanup_output(config, execution_name_2)
    save_config(config, config_file_path)
    try:
        pipeline.main(execution_name_2)
    except Exception as e:
        print(f"ERROR in Run 2: {e}")
        save_config(original_config, config_file_path)
        return False

    # Get output path and load results
    output_path_2 = get_output_path(execution_name_2, config)
    results_df_2 = load_results_df(output_path_2)

    if results_df_2 is None:
        print(f"ERROR: Could not load results_df from Run 2")
        save_config(original_config, config_file_path)
        return False

    print(f"✓ Run 2 completed. Loaded {len(results_df_2)} rows")
    print()

    # Compare results
    print("COMPARISON RESULTS")
    print("-" * 80)
    is_identical, report = compare_dataframes(results_df_1, results_df_2)

    print(report)
    print()

    # Restore original config
    save_config(original_config, config_file_path)

    if is_identical:
        print("=" * 80)
        print("✅ SUCCESS: Pipeline is DETERMINISTIC!")
        print("   Both runs produced identical results_df")
        print(f"   Seed configuration (random_seed={random_seed}) is working correctly")
        print("=" * 80)
        return True
    else:
        print("=" * 80)
        print("❌ FAILURE: Pipeline is NOT DETERMINISTIC")
        print("   Results differ between runs")
        if random_seed != "0":
            print(f"   Even with random_seed={random_seed} set, results are not deterministic")
            print("   Please check seed_manager configuration in pipeline.py")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

