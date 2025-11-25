from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Please install torch.")
    sys.exit(1)

from fusion import DataService, FusionWrapper, SceneSpec


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser()

    # ========== 基本参数 ==========
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="场景名称 (必需)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data",
        help="数据集根目录 (默认: data)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data = DataService(Path(args.dataset_root) / args.scene_name)
    data.materialize_gaussian_scene(Path("work/gaussian_sources") / args.scene_name)
    data.materialize_neus_scene(Path("work/neus_sources") / args.scene_name)


if __name__ == "__main__":
    main()
