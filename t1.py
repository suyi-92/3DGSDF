import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tempfile
import os

try:
    import torch
except ImportError:  # pragma: no cover - tests skipped when torch missing
    torch = None

from fusion.wrapper import (
    DataService,
    FusionWrapper,
    GaussianIterationState,
    NeuSIterationState,
    RayBatch,
    SceneSpec,
)


@unittest.skipIf(torch is None, "PyTorch not available for DataService tests.")
class DataServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / "images").mkdir(parents=True)
        (self.root / "sparse" / "0").mkdir(parents=True)
        for i in range(3):
            (self.root / "images" / f"{i:03d}.png").write_bytes(b"fake")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (self.root / "sparse" / "0" / name).write_bytes(b"\x00")
        np.save(self.root / "poses_bounds.npy", np.zeros((1, 17), dtype=np.float32))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_materialize_and_sample(self):
        ds = DataService(self.root)
        self.assertEqual(len(ds.list_images()), 3)
        bundle = ds.get_sparse_bundle()
        self.assertTrue(bundle.cameras.exists())

        dest = self.root / "gs_materialized"
        manifest = ds.materialize_gaussian_scene(dest)
        self.assertTrue((Path(manifest["source"]) / "images").exists())

        def dummy_sampler(service: DataService, batch_size: int, **_):
            rays = torch.zeros((batch_size, 3))
            return RayBatch(rays, rays + 1, rays + 2)

        ds.register_ray_sampler(dummy_sampler)
        batch = ds.sample_rays(4)
        self.assertEqual(batch.origins.shape[0], 4)


class FusionWrapperTests(unittest.TestCase):
    def setUp(self):
        self.spec = SceneSpec(
            scene_name="dummy",
            dataset_root="data/raw",
            gaussian_source_path="data/gs_scene",
            gaussian_model_path="data/dummy_model",
            neus_conf_path="dummy.conf",
            neus_case="dummy",
            shared_workspace="output/fusion",
        )
        self.gaussian_repo = Path("gaussian_splatting")
        self.neus_repo = Path("NeuS")
        self.gaussian_cfg = {"model": {"sh_degree": 3}}
        self.neus_cfg = {"mode": "train"}
        self.data_service = MagicMock()
        self.tmp_scene = tempfile.TemporaryDirectory()
        self.data_service.materialize_neus_scene.return_value = {
            "scene_dir": Path(self.tmp_scene.name)
        }
        self.tmp_conf = tempfile.NamedTemporaryFile(delete=False, suffix=".conf")
        self.tmp_conf.write(b"dataset { data_dir = ./public_data/CASE_NAME/ }\n")
        self.tmp_conf.write(b"general { base_exp_dir = ./exp/CASE_NAME }\n")
        self.tmp_conf.flush()
        self.tmp_conf.close()
        self.spec.neus_conf_path = self.tmp_conf.name

    def tearDown(self):
        if hasattr(self, "tmp_conf") and os.path.exists(self.tmp_conf.name):
            os.unlink(self.tmp_conf.name)
        if hasattr(self, "tmp_scene"):
            self.tmp_scene.cleanup()

    @patch("fusion.wrapper.NeuSAdapter")
    @patch("fusion.wrapper.GaussianSplattingAdapter")
    def test_bootstrap_invokes_underlying_adapters(
        self, mock_gaussian_cls, mock_neus_cls
    ):
        mock_gaussian = MagicMock()
        mock_neus = MagicMock()
        mock_gaussian_cls.return_value = mock_gaussian
        mock_neus_cls.return_value = mock_neus

        wrapper = FusionWrapper(
            self.spec,
            self.gaussian_repo,
            self.neus_repo,
            self.data_service,
            self.gaussian_cfg,
            self.neus_cfg,
        )
        wrapper.bootstrap()

        mock_gaussian.bootstrap.assert_called_once_with(self.spec)
        mock_neus.bootstrap.assert_called_once_with(self.spec)

    @patch("fusion.wrapper.NeuSAdapter")
    @patch("fusion.wrapper.GaussianSplattingAdapter")
    def test_joint_step_returns_combined_state(
        self, mock_gaussian_cls, mock_neus_cls
    ):
        gaussian_state = GaussianIterationState(
            iteration=10,
            loss=1.0,
            l1=0.5,
            ssim=0.8,
            lr_position=1e-4,
            num_gaussians=42,
        )
        neus_state = NeuSIterationState(
            iteration=20, loss=0.7, color_loss=0.3, eikonal_loss=0.1, lr=5e-4
        )

        mock_gaussian = MagicMock()
        mock_gaussian.train_step.return_value = gaussian_state
        mock_neus = MagicMock()
        mock_neus.train_step.return_value = neus_state

        mock_gaussian_cls.return_value = mock_gaussian
        mock_neus_cls.return_value = mock_neus

        wrapper = FusionWrapper(
            self.spec,
            self.gaussian_repo,
            self.neus_repo,
            self.data_service,
            self.gaussian_cfg,
            self.neus_cfg,
        )

        payload = wrapper.joint_step(mesh_every=5)

        self.assertEqual(payload["gaussian"], gaussian_state)
        self.assertEqual(payload["neus"], neus_state)
        self.assertEqual(payload["fusion_step"], 1)
        mock_neus.train_step.assert_called_once()
        mock_gaussian.train_step.assert_called_once()

    @patch("fusion.wrapper.NeuSAdapter")
    @patch("fusion.wrapper.GaussianSplattingAdapter")
    def test_mutable_delegates_to_selected_adapter(
        self, mock_gaussian_cls, mock_neus_cls
    ):
        mock_gaussian = MagicMock()
        mock_neus = MagicMock()
        mock_gaussian_cls.return_value = mock_gaussian
        mock_neus_cls.return_value = mock_neus

        wrapper = FusionWrapper(
            self.spec,
            self.gaussian_repo,
            self.neus_repo,
            self.data_service,
            self.gaussian_cfg,
            self.neus_cfg,
        )

        wrapper.mutable("gaussian", "weights")
        mock_gaussian.mutable.assert_called_once_with("weights")

        wrapper.mutable("neus", "renderer")
        mock_neus.mutable.assert_called_once_with("renderer")

    @patch("fusion.wrapper.NeuSAdapter")
    @patch("fusion.wrapper.GaussianSplattingAdapter")
    def test_depth_cache_updates_from_bus(
        self, mock_gaussian_cls, mock_neus_cls
    ):
        mock_gaussian_cls.return_value = MagicMock()
        mock_neus_cls.return_value = MagicMock()

        wrapper = FusionWrapper(
            self.spec,
            self.gaussian_repo,
            self.neus_repo,
            self.data_service,
            self.gaussian_cfg,
            self.neus_cfg,
        )

        payload = {"camera_id": 3, "depth": "depth_map", "normal": "n", "iteration": 7}
        wrapper._on_gaussian_render(payload)
        self.assertIn(3, wrapper.depth_cache)
        self.assertEqual(wrapper.depth_cache[3]["iteration"], 7)


if __name__ == "__main__":
    unittest.main()
