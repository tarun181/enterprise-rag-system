import os
import zipfile
import subprocess
import requests


def download_python_docs(target_dir: str = "data/raw_docs/python_docs"):
    """Downloads and extracts the official Python HTML documentation."""
    os.makedirs(target_dir, exist_ok=True)

    # Updated URL to the working Python 3.14 docs archive
    url = "https://docs.python.org/3/archives/python-3.14-docs-html.zip"
    zip_path = os.path.join(target_dir, "python-docs.zip")

    print(f"Downloading Python documentation from {url}...")

    # Using 'requests' is more robust for handling redirects and headers
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Will immediately tell us if there's a 404

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(zip_path)  # Cleanup zip file to save space
    print(f"Python docs successfully extracted to {target_dir}")


def download_langchain_docs(target_dir: str = "data/raw_docs/langchain_docs"):
    """Clones the LangChain repo specifically for its docs folder."""
    os.makedirs(target_dir, exist_ok=True)
    print("Fetching LangChain documentation via git sparse-checkout...")

    repo_url = "https://github.com/langchain-ai/langchain.git"

    try:
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", repo_url, target_dir], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "docs"], cwd=target_dir, check=True)
        print(f"LangChain docs ready in {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch LangChain docs. Ensure git is installed. Error: {e}")


if __name__ == "__main__":
    download_python_docs()