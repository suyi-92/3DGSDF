from __future__ import annotations

"""
Prewarm utilities for Gaussian Splatting and NeuS.

在进入联合优化前，为 3DGS 与 NeuS 提供预热训练与检查点读写能力。
所有逻辑都位于 fusion 包装层，避免侵入原始子仓。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .fusion_wrapper import FusionWrapper
from .gaussian_adapter import GaussianSplattingAdapter
from .neus_adapter import NeuSAdapter


@dataclass
class PrewarmResult3DGS:
    adapter: GaussianSplattingAdapter
    last_state: Any
    checkpoint_dir: Path
    meta: Dict[str, Any]


@dataclass
class PrewarmResultNeuS:
    adapter: NeuSAdapter
    last_state: Any
    checkpoint_path: Path
    meta: Dict[str, Any]


def _prepare_output_dir(base: Path, scene_name: str, model: str, overwrite: bool) -> Path:
    target = base / scene_name / model
    target.mkdir(parents=True, exist_ok=True)
    meta_path = target / "meta.json"
    if meta_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prewarm metadata already exists at {meta_path}. "
            "Pass overwrite=True to replace it."
        )
    return target


def _dump_meta(meta_dir: Path, meta: Dict[str, Any]):
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta_path


def prewarm_gaussian(
    wrapper: FusionWrapper,
    iterations: int,
    log_every: int,
    output_root: Path,
    overwrite: bool = False,
    save_checkpoint: bool = True,
) -> PrewarmResult3DGS:
    """
    Run Gaussian Splatting prewarm using the existing adapter training loop.

    Args:
        wrapper: 已经 bootstrap 完成的 FusionWrapper。
        iterations: 预热步数。
        log_every: 指标打印间隔。
        output_root: 预热元数据的根目录。
        overwrite: 是否覆盖已有 meta.json。
        save_checkpoint: 是否在结束时导出 checkpoint。
    """

    last_state = None
    for step in range(1, iterations + 1):
        last_state = wrapper.gaussian.train_step()
        if step % log_every == 0 or step == iterations:
            print(
                f"[Prewarm][3DGS] iter={last_state.iteration} "
                f"loss={last_state.loss:.4f} l1={last_state.l1:.4f} "
                f"ssim={last_state.ssim:.4f} gaussians={last_state.num_gaussians}"
            )

    checkpoint_path: Optional[Path] = None
    if save_checkpoint:
        checkpoint_path = wrapper.gaussian.export_surface(last_state.iteration)

    meta_dir = _prepare_output_dir(output_root, wrapper.spec.scene_name, "3dgs", overwrite)
    meta = {
        "scene": wrapper.spec.scene_name,
        "iterations": iterations,
        "checkpoint": checkpoint_path.as_posix() if checkpoint_path else None,
        "gaussian_model_path": wrapper.spec.gaussian_model_path,
    }
    _dump_meta(meta_dir, meta)

    return PrewarmResult3DGS(
        adapter=wrapper.gaussian,
        last_state=last_state,
        checkpoint_dir=checkpoint_path.parent if checkpoint_path else Path(wrapper.spec.gaussian_model_path),
        meta=meta,
    )


def prewarm_neus(
    wrapper: FusionWrapper,
    iterations: int,
    log_every: int,
    output_root: Path,
    overwrite: bool = False,
    save_checkpoint: bool = True,
) -> PrewarmResultNeuS:
    """
    Run NeuS prewarm via the adapter's train_step loop.

    Args:
        wrapper: 已经 bootstrap 完成的 FusionWrapper。
        iterations: 预热步数。
        log_every: 指标打印间隔。
        output_root: 预热元数据的根目录。
        overwrite: 是否覆盖已有 meta.json。
        save_checkpoint: 是否保存 NeuS checkpoint。
    """

    last_state = None
    for step in range(1, iterations + 1):
        last_state = wrapper.neus.train_step()
        if step % log_every == 0 or step == iterations:
            print(
                f"[Prewarm][NeuS] iter={last_state.iteration} "
                f"loss={last_state.loss:.4f} color={last_state.color_loss:.4f} "
                f"eikonal={last_state.eikonal_loss:.4f}"
            )

    checkpoint_path: Optional[Path] = None
    if save_checkpoint:
        runner = wrapper.neus.runner
        if runner is None:
            raise RuntimeError("NeuS runner is not initialized; cannot save checkpoint.")
        runner.save_checkpoint()
        checkpoint_path = Path(runner.base_exp_dir) / "checkpoints" / f"ckpt_{last_state.iteration:0>6d}.pth"

    meta_dir = _prepare_output_dir(output_root, wrapper.spec.scene_name, "neus", overwrite)
    meta = {
        "scene": wrapper.spec.scene_name,
        "iterations": iterations,
        "checkpoint": checkpoint_path.as_posix() if checkpoint_path else None,
        "base_exp_dir": getattr(wrapper.neus.runner, "base_exp_dir", None),
    }
    _dump_meta(meta_dir, meta)

    return PrewarmResultNeuS(
        adapter=wrapper.neus,
        last_state=last_state,
        checkpoint_path=checkpoint_path if checkpoint_path else Path(),
        meta=meta,
    )


def load_prewarm_gaussian(
    adapter: GaussianSplattingAdapter,
    meta_path: Path,
) -> Path:
    """
    Load 3DGS checkpoint according to prewarm metadata.

    Returns loaded checkpoint path.
    """

    meta = json.loads(Path(meta_path).read_text())
    ckpt = meta.get("checkpoint")
    if not ckpt:
        raise FileNotFoundError(f"No Gaussian checkpoint recorded in {meta_path}")
    ckpt_path = Path(ckpt)
    adapter.load_checkpoint(ckpt_path)
    return ckpt_path


def load_prewarm_neus(adapter: NeuSAdapter, meta_path: Path) -> Path:
    """
    Load NeuS checkpoint according to prewarm metadata.

    Returns loaded checkpoint path.
    """

    meta = json.loads(Path(meta_path).read_text())
    ckpt = meta.get("checkpoint")
    if not ckpt:
        raise FileNotFoundError(f"No NeuS checkpoint recorded in {meta_path}")
    ckpt_path = Path(ckpt)
    adapter.load_checkpoint(ckpt_path)
    return ckpt_path
