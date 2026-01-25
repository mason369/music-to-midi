import os
from huggingface_hub import snapshot_download

# 强制使用镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_ultimate_moe():
    # 设定专门存放最强专家的路径
    base_dir = os.path.expanduser("~/.cache/music_ai_models/yourmt3_ultimate")
    os.makedirs(base_dir, exist_ok=True)
    
    print("🎯 正在精准定位 YPTF.MoE + Multi (128专家模型)...")

    try:
        # 只拉取包含 'moe' 和 'multi' 关键字的 v7 版本权重
        # 这是目前 YourMT3+ 论文中性能最强的生产级别权重
        snapshot_download(
            repo_id="mimbres/YourMT3",
            local_dir=base_dir,
            allow_patterns=[
                "*moe*multi*.ckpt", 
                "*v7*moe*.ckpt",
                "config.yaml" 
            ],
            ignore_patterns=["*.msgpack", "*.h5", "notask_*"], # 排除非专家模型
            local_dir_use_symlinks=False
        )
        
        print(f"\n✅ 128专家‘完全体’下载成功！")
        print(f"📍 模型存放位置: {base_dir}")
        
        # 列出下载到的 .ckpt 文件，确保它是那个 ~760MB 的大家伙
        print("\n当前已就绪的专家权重：")
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    size = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
                    print(f"  - {file} ({size:.2f} MB)")

    except Exception as e:
        print(f"❌ 下载失败，请检查镜像站连接: {e}")

if __name__ == "__main__":
    download_ultimate_moe()