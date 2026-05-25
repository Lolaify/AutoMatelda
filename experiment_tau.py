import os
import shutil
import pipeline
from gather_tau_results import read_config, get_output_path, get_columns_num

def write_config(config, file_path):
    with open(file_path, 'w') as configfile:
        config.write(configfile)

def copy_reusable_features(src_dir, dst_dir, include_cell_clustering=False):
    if src_dir and dst_dir and os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)

    # Files to copy for tg/cg/features (independent of budget/tau)
    reusable_files = [
        "tables_dict.pickle",
        "table_group_dict.pickle",
        "table_size_dict.pickle",
        "features.pickle",
        "tables_tuples.pickle",
    ]

    for f in reusable_files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

    # Mediate files (col groups etc)
    src_mediate = os.path.join(src_dir, "mediate_files")
    dst_mediate = os.path.join(dst_dir, "mediate_files")
    if os.path.exists(src_mediate) and not os.path.exists(dst_mediate):
        shutil.copytree(src_mediate, dst_mediate)

    if include_cell_clustering:
        src_cc = os.path.join(src_dir, "cell_clustering")
        dst_cc = os.path.join(dst_dir, "cell_clustering")
        if os.path.exists(src_cc) and not os.path.exists(dst_cc):
            shutil.copytree(src_cc, dst_cc)

import sys

def run_tau_experiment(tables_dir_override=None):
    config_path = './config.ini'
    config = read_config(config_path)

    if tables_dir_override:
        config.set('DIRECTORIES', 'tables_dir', tables_dir_override)
        dataset_name = os.path.basename(tables_dir_override)
        config.set('DIRECTORIES', 'output_dir', f"tau_exp_output/{dataset_name}")
        print(f"[INFO] Override dataset: {dataset_name}")
    else:
        # Default fallback isolated folder if no CLI override
        dataset_name = os.path.basename(config.get('DIRECTORIES', 'tables_dir', fallback='DGov_Typo_10'))
        config.set('DIRECTORIES', 'output_dir', f"tau_exp_output/{dataset_name}")
        print(f"[INFO] Standard dataset: {dataset_name}")

    # We want Variant 2 for all runs
    config.set('AUTO-TEST', 'integration_pipeline_option', '2')
    config.set('EXPERIMENTS', 'save_mediate_res_on_disk', '1')

    taus = [0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.33,0.50]
    print(f"[INFO] Taus scheduled: {taus}")

    # Calculate budgets dynamically like autotest_exp.py
    dataset_path = os.path.join(config["DIRECTORIES"]["sandbox_dir"], config["DIRECTORIES"]["tables_dir"])
    cols_num = get_columns_num(dataset_path)
    if cols_num == 0:
        cols_num = 100 # safe fallback
    multipliers = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

    budgets = []
    for m in multipliers:
        b = max(10, round(cols_num * m))
        if b not in budgets:
            budgets.append(b)

    print(f"[INFO] Total columns detected: {cols_num}")
    print(f"[INFO] Computed labeling budgets: {budgets}")

    first_run_ever = True
    base_src_dir = None

    for budget in budgets:
        print(f"\n>>> Starting Phase for Budget: {budget} <<<")
        config.set('EXPERIMENTS', 'labeling_budget', str(budget))
        budget_src_dir = None

        for first_run_tau, tau in enumerate(taus):
            config.set('AUTO-TEST', 'override_threshold', str(tau))
            exec_name = f"tau_{tau}_budget_{budget}"

            out_dir = get_output_path(exec_name, config, budget)

            if first_run_ever:
                print(f"    [Setup] Initializing fresh state for first run...")
                config.set('TABLE_GROUPING', 'tg_res_available', '0')
                config.set('COLUMN_GROUPING', 'cg_res_available', '0')
                config.set('CELL_GROUPING', 'cell_clustering_res_available', '0')
                config.set('CELL_GROUPING', 'cell_feature_generator_enabled', '1')
            else:
                config.set('TABLE_GROUPING', 'tg_res_available', '1')
                config.set('COLUMN_GROUPING', 'cg_res_available', '1')
                config.set('CELL_GROUPING', 'cell_feature_generator_enabled', '0')

                if first_run_tau == 0:
                    # New budget -> new cell clustering, but copy general features
                    print(f"    [Setup] Re-computing Cell Clustering for new budget: {budget}")
                    config.set('CELL_GROUPING', 'cell_clustering_res_available', '0')
                    copy_reusable_features(base_src_dir, out_dir, include_cell_clustering=False)
                else:
                    # Same budget -> reuse cell clustering
                    print(f"    [Setup] Reusing Cell Clustering for identical budget (tau={tau})")
                    config.set('CELL_GROUPING', 'cell_clustering_res_available', '1')
                    copy_reusable_features(budget_src_dir, out_dir, include_cell_clustering=True)

            write_config(config, config_path)

            print(f"\n==========================================")
            print(f"Executing -> Tau: {tau} | Budget: {budget}")
            print(f"Output Directory: {out_dir}")
            print(f"==========================================")

            # Optimization: Skip if results already exist
            results_pickle = os.path.join(out_dir, "results", "scores_all.pickle")
            if os.path.exists(results_pickle):
                print(f"[Skip] Results already exist for Tau: {tau} | Budget: {budget}. Skipping pipeline execution.")
                if first_run_ever:
                    base_src_dir = out_dir
                    first_run_ever = False
                if first_run_tau == 0:
                    budget_src_dir = out_dir
                continue

            try:
                pipeline.main(exec_name)
                print(f"[Success] Completed Tau: {tau} | Budget: {budget}")
            except Exception as e:
                print(f"[Error] Failed execution for Tau: {tau} | Budget: {budget}")
                print(f"Details: {e}")

            if first_run_ever:
                base_src_dir = out_dir
                first_run_ever = False

            if first_run_tau == 0:
                budget_src_dir = out_dir

    print("\n[INFO] All Experiment executions finished successfully.")

if __name__ == "__main__":
    tables_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_tau_experiment(tables_dir_arg)
