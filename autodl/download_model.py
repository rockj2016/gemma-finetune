
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
print(HF_TOKEN)

# https://huggingface.co/google/gemma-3-4b-it
HF_MODEL_PATH = "google/gemma-3-4b-it"

# # 本地模型目录
LOCAL_MODEL_DIR = "../models/gemma-3-4b-it"

if __name__ == "__main__":
    print(HF_TOKEN)
    # 下载模型
    print("开始下载模型")
    snapshot_download(
        repo_id=HF_MODEL_PATH,
        local_dir=LOCAL_MODEL_DIR,
        token=HF_TOKEN,
        resume_download=True,
        max_workers=4
    )


