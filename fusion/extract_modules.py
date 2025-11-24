# 脚本用于提取和创建模块文件
import os

# 读取原始文件
with open(r"e:\11\NN_Project\3DGSDF\fusion\wrapper.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# ===== 创建 gaussian_adapter.py =====
header_gaussian = """from __future__ import annotations

\"\"\"
Gaussian Splatting adapter for fusion wrapper.

3DGS 包装器模块，提供训练、渲染和属性访问接口。
\"\"\"

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from .common import (
    AdapterBase,
    APIRegistry,
    ExchangeBus,
    SceneSpec,
    GaussianIterationState,
    MutableHandle,
)
from .data_service import DataService


"""

# 提取 GaussianSplattingAdapter (634-1511, 0-indexed 633-1510)
gaussian_content = "".join(lines[633:1511])

with open(
    r"e:\11\NN_Project\3DGSDF\fusion\gaussian_adapter.py", "w", encoding="utf-8"
) as f:
    f.write(header_gaussian + gaussian_content)

print("Created gaussian_adapter.py")

# ===== 创建 neus_adapter.py =====
header_neus = """from __future__ import annotations

\"\"\"
NeuS adapter for fusion wrapper.

NeuS 包装器模块，提供 SDF 训练和评估接口。
\"\"\"

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .common import (
    AdapterBase,
    APIRegistry,
    ExchangeBus,
    SceneSpec,
    NeuSIterationState,
    MutableHandle,
)
from .data_service import DataService


"""

# 提取 NeuSAdapter (1512-1954, 0-indexed 1511-1953)
neus_content = "".join(lines[1511:1954])

with open(
    r"e:\11\NN_Project\3DGSDF\fusion\neus_adapter.py", "w", encoding="utf-8"
) as f:
    f.write(header_neus + neus_content)

print("Created neus_adapter.py")

# ===== 创建 fusion_wrapper.py =====
header_fusion = """from __future__ import annotations

\"\"\"
Fusion wrapper combining Gaussian Splatting and NeuS.

联合融合包装器，整合 3DGS 和 NeuS 的训练与交互。
\"\"\"

from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .common import APIRegistry, ExchangeBus, SceneSpec
from .data_service import DataService
from .gaussian_adapter import GaussianSplattingAdapter
from .neus_adapter import NeuSAdapter


"""

# 提取 FusionWrapper (1955-end, 0-indexed 1954-end)
fusion_content = "".join(lines[1954:])

with open(
    r"e:\11\NN_Project\3DGSDF\fusion\fusion_wrapper.py", "w", encoding="utf-8"
) as f:
    f.write(header_fusion + fusion_content)

print("Created fusion_wrapper.py")
print("All adapter files created successfully!")
