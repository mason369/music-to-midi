import os
import ssl
import urllib.request
from pathlib import Path
from huggingface_hub import list_repo_files, snapshot_download

# YPTF.MoE+Multi (PS) 的实际 checkpoint 目录名（来自官方训练命令）
# 参见：https://github.com/mimbres/YourMT3/issues/2#issuecomment-2255643217
MOE_EXP_ID = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2"
REPO_ID = "mimbres/YourMT3"


def _fix_ssl_if_needed():
    """
    检测并修复 SSL 证书验证问题。
    在企业网络/校园网/代理环境下，HTTPS 流量可能被拦截导致证书验证失败。
    尝试安装 certifi 证书；若仍失败则设置环境变量跳过验证。
    """
    test_url = "https://huggingface.co"
    # 先测试默认 SSL 是否正常
    try:
        req = urllib.request.Request(test_url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        return  # SSL 正常，无需修复
    except (ssl.SSLCertVerificationError, urllib.error.URLError):
        pass
    except Exception:
        pass

    # 尝试使用 certifi 证书
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        req = urllib.request.Request(test_url, method="HEAD")
        ctx = ssl.create_default_context(cafile=certifi.where())
        urllib.request.urlopen(req, timeout=10, context=ctx)
        print("已使用 certifi 证书修复 SSL 验证")
        return
    except Exception:
        pass

    # 最终回退：跳过 SSL 验证（不安全，但在受限网络下是唯一选择）
    print("警告：SSL 证书验证失败，将跳过验证（可能处于代理/企业网络环境）")
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    # 同时为 requests 库禁用警告
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


def find_ckpt_path_in_repo(exp_id: str) -> tuple[str | None, str | None]:
    """
    动态查询仓库中 checkpoint 文件的实际路径，不依赖硬编码目录结构。
    返回 (model.ckpt路径, last.ckpt路径)，未找到则为 None。
    """
    print(f"查询仓库文件列表以确定 checkpoint 路径...")
    files = list(list_repo_files(REPO_ID, repo_type="space"))
    model_ckpt = next((f for f in files if exp_id in f and f.endswith("model.ckpt")), None)
    last_ckpt = next((f for f in files if exp_id in f and f.endswith("last.ckpt")), None)
    return model_ckpt, last_ckpt


def download_ultimate_moe():
    base_dir = os.path.expanduser("~/.cache/music_ai_models/yourmt3_all")
    os.makedirs(base_dir, exist_ok=True)

    print("正在下载 YPTF.MoE+Multi (PS) - YourMT3+ 最高性能模型...")
    print(f"架构：8 专家 MoE, Top-2 路由, 13 通道 Perceiver Transformer")
    print(f"目标路径：{base_dir}")

    # 修复 SSL 证书问题（企业网络/校园网/代理环境）
    _fix_ssl_if_needed()

    try:
        # 动态发现仓库中的实际路径，避免因目录结构变化导致下载失败
        model_ckpt_path, last_ckpt_path = find_ckpt_path_in_repo(MOE_EXP_ID)

        if not model_ckpt_path and not last_ckpt_path:
            raise RuntimeError(
                f"在仓库 {REPO_ID} 中未找到包含 '{MOE_EXP_ID}' 的 checkpoint 文件，"
                f"请检查 MOE_EXP_ID 是否正确。"
            )

        allow_patterns = [p for p in [model_ckpt_path, last_ckpt_path] if p]
        print(f"将下载：{allow_patterns}")

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="space",
            local_dir=base_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=["*.msgpack", "*.h5", "*.safetensors", "__pycache__/**"],
            local_dir_use_symlinks=False,
        )

        # 验证：递归查找已下载的 .ckpt 文件（不依赖固定路径）
        downloaded = list(Path(base_dir).rglob("*.ckpt"))
        moe_ckpts = [p for p in downloaded if MOE_EXP_ID in str(p)]

        if moe_ckpts:
            for p in moe_ckpts:
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"\n下载成功！")
                print(f"模型文件：{p} ({size_mb:.1f} MB)")
        else:
            print(f"\n警告：未找到 checkpoint 文件，请检查网络或手动放置模型文件")

    except Exception as e:
        print(f"\n[错误] 模型下载失败：{e}")
        print("\n" + "=" * 50)
        print("手动下载方法：")
        print(f"1. 访问 https://huggingface.co/spaces/{REPO_ID}")
        print(f"2. 找到包含 '{MOE_EXP_ID}' 的 model.ckpt 文件并下载")
        print(f"3. 放置到 {base_dir}/<仓库中的相对路径>")
        print("=" * 50)
        import sys
        sys.exit(1)


if __name__ == "__main__":
    download_ultimate_moe()
