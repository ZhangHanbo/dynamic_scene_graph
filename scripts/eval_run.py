#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

# Bootstrap: scripts/ for sibling-import of data_demo, repo-root for top-level packages.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_demo import load_config


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DEMO_PATH = os.path.join(SCRIPTS_DIR, "data_demo.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on all datasets in a folder")
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--tracker', choices=['heuristic', 'ekf'], default='heuristic',
                        help="Tracker variant passed through to data_demo.py.")
    parser.add_argument('--ekf-backend', choices=['gaussian', 'rbpf'], default='gaussian',
                        help="EKF backend (only when --tracker=ekf).")
    return parser.parse_args()

def is_dataset_folder(path):
    required = ["rgb", "depth", "pose_txt"]
    return all(os.path.isdir(os.path.join(path, x)) for x in required)

def run_dataset_in_subprocess(dataset_path, config_path, tracker, ekf_backend):
    """
    每个 dataset 开一个子进程执行 data_demo.py，相当于手动 Ctrl-C 之后重新运行。
    """
    print(f"\n🚀 启动子进程处理数据集: {dataset_path}  (tracker={tracker})")

    cmd = [
        "python3",
        DATA_DEMO_PATH,
        "-c", config_path,
        "--dataset", dataset_path,
        "--tracker", tracker,
    ]
    if tracker == "ekf":
        cmd += ["--ekf-backend", ekf_backend]

    # subprocess.run 会等待子进程结束
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"⚠️ 子进程处理失败（return code={result.returncode}）: {dataset_path}")
    else:
        print(f"✅ 子进程处理完成: {dataset_path}")

def main():
    args = parse_args()
    config = load_config(args.config)
    root_dir = config['dataset']['path']

    print("============================================")
    print(f" Evaluating all datasets in: {root_dir}")
    print("============================================")

    subdirs = sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    valid_datasets = []
    for d in subdirs:
        if is_dataset_folder(d):
            valid_datasets.append(d)
        else:
            print(f"[Skip] {d} 不是合法数据集，跳过")

    print("--------------------------------------------")
    print(f"共找到 {len(valid_datasets)} 个合法数据集：")
    for ds in valid_datasets:
        print(" -", ds)
    print("--------------------------------------------")

    for ds in valid_datasets:
        run_dataset_in_subprocess(ds, args.config, args.tracker, args.ekf_backend)

    print("============================================")
    print("🎉 全部数据集处理完成!")
    print("============================================")

if __name__ == "__main__":
    main()
