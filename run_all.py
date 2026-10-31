import os
import sys
import time

from src import config


def check_data_present():
    problems = []
    if not os.path.exists(config.TABULAR_DIAGNOSIS_CSV):
        problems.append(f"Missing: {config.TABULAR_DIAGNOSIS_CSV}")
    if not os.path.exists(config.TABULAR_PROGRESSION_CSV):
        problems.append(f"Missing: {config.TABULAR_PROGRESSION_CSV}")
    if not os.path.isdir(config.IMAGING_DIR):
        problems.append(f"Missing folder: {config.IMAGING_DIR}")
    return problems


def main():
    print("=" * 60)
    print("Quantum AI Parkinson's Disease Diagnosis Optimizer")
    print("=" * 60)

    problems = check_data_present()
    if problems:
        print("\nCan't start - missing data:")
        for p in problems:
            print(f"  - {p}")
        print("\nSee README.md step 'Place your 4 raw data files' and fix these, then re-run.")
        sys.exit(1)

    summary = []
    t_start = time.time()

    steps = [
        ("Diagnosis (clinical CSV, 2105 rows)", "train_diagnosis", False),
        ("Progression (severity CSV, 500 rows)", "train_progression", False),
        ("Voice (healthy vs Parkinson's)", "train_voice", True),
        ("Imaging (MRI radiomics + anomaly detection)", "explore_imaging", False),
        ("Imaging diagnosis (labeled MRI classifier)", "train_imaging", True),
        ("Export consolidated backend_summary.json", "export_backend_summary", False),
    ]

    for label, module_name, optional in steps:
        print(f"\n--- {label} ---")
        t0 = time.time()
        try:
            module = __import__(module_name)
            module.main()
            summary.append((label, "OK", time.time() - t0))
        except FileNotFoundError as e:
            if optional:
                summary.append((label, "SKIPPED (data not present - see README)", time.time() - t0))
                print(f"[skip] {label}: {e}")
            else:
                summary.append((label, f"FAILED: {e}", time.time() - t0))
                print(f"[!] {label} failed: {e}")
        except Exception as e:
            summary.append((label, f"FAILED: {e}", time.time() - t0))
            print(f"[!] {label} failed: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, status, dt in summary:
        print(f"{status:10s} {label}  ({dt:.1f}s)")
    print(f"\nTotal time: {(time.time() - t_start)/60:.1f} min")
    print(f"Results written to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
