"""
YourMT3 模型路径解析单元测试

测试 checkpoint 名称到实际文件系统路径的映射功能
"""
import unittest
from pathlib import Path
from src.utils.yourmt3_downloader import (
    get_model_path,
    resolve_model_checkpoint_path,
    CHECKPOINT_FILENAME_MAP,
    DEFAULT_MODEL,
    YOURMT3_MODELS
)


class TestYourMT3PathResolution(unittest.TestCase):
    """测试 YourMT3 模型路径解析"""

    def test_checkpoint_filename_map_exists(self):
        """测试映射表存在且不为空"""
        self.assertIsNotNone(CHECKPOINT_FILENAME_MAP)
        self.assertGreater(len(CHECKPOINT_FILENAME_MAP), 0)

        # 验证 MoE 模型映射
        self.assertIn("YPTF.MoE+Multi (PS)", CHECKPOINT_FILENAME_MAP)
        self.assertEqual(
            CHECKPOINT_FILENAME_MAP["YPTF.MoE+Multi (PS)"],
            "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2"
        )

    def test_default_model_is_moe(self):
        """测试默认模型已切换为 MoE"""
        self.assertEqual(DEFAULT_MODEL, "yptf_moe_multi_ps")

    def test_resolve_moe_checkpoint_path(self):
        """测试解析 MoE checkpoint 路径"""
        path = resolve_model_checkpoint_path("YPTF.MoE+Multi (PS)")

        if path is None:
            self.skipTest("MoE 模型未下载，跳过路径验证")

        # 验证路径存在
        self.assertIsNotNone(path)
        self.assertTrue(path.exists(), f"路径不存在: {path}")

        # 验证文件名
        self.assertEqual(path.name, "model.ckpt")

        # 验证目录结构包含 MoE 标识
        self.assertIn("moe", str(path).lower())

        # 验证文件大小（MoE PS 版本约 724MB）
        size_mb = path.stat().st_size / (1024 * 1024)
        self.assertGreater(size_mb, 700)
        self.assertLess(size_mb, 750)

    def test_get_model_path_short_name(self):
        """测试通过短名称获取模型路径"""
        path = get_model_path("yptf_moe_multi_ps")

        if path is None:
            self.skipTest("MoE 模型未下载")

        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    def test_get_model_path_fallback(self):
        """测试回退到旧模型"""
        # 如果新模型不存在，应该能找到旧模型
        path = get_model_path("mc13_256_all_cross_v6")

        if path is None:
            self.skipTest("旧模型未下载")

        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    def test_all_registered_models_have_checkpoints(self):
        """测试所有注册的模型都有 checkpoint 配置"""
        for model_name, model_info in YOURMT3_MODELS.items():
            self.assertIn("checkpoint", model_info,
                         f"模型 '{model_name}' 缺少 checkpoint 配置")
            self.assertIsNotNone(model_info["checkpoint"])

    def test_resolve_old_format_checkpoint(self):
        """测试解析旧格式 checkpoint 名称（带 @model.ckpt）"""
        old_format = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
        path = resolve_model_checkpoint_path(old_format)

        if path is None:
            self.skipTest("旧模型未下载")

        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    def test_resolve_nonexistent_model(self):
        """测试解析不存在的模型返回 None"""
        path = resolve_model_checkpoint_path("nonexistent_model_xyz")
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
