import os
from huggingface_hub import snapshot_download


def download_local_model():
    # Swapped to the 4B model
    model_repo = "Qwen/Qwen3.5-4B"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    # Updated the local folder name
    local_dir = os.path.join(project_root, "models", "Qwen3.5-4B")

    print(f"Starting download of {model_repo}...")
    print(f"Saving explicitly to: {local_dir}")
    print("This may take a while depending on your internet connection...\n")

    snapshot_download(
        repo_id=model_repo,
        local_dir=local_dir,
        ignore_patterns=["*.bin", "*.pt", "*.h5", "*.msgpack"]
    )

    print("\n✅ Model successfully saved locally!")


if __name__ == "__main__":
    download_local_model()