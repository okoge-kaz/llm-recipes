import os
import argparse
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo


def upload_directory(api, local_dir, repo_name, repo_type, branch_name):
    for root, _, files in os.walk(local_dir):
        for file in tqdm(files, desc=f"Uploading files in {root}"):
            local_path = os.path.join(root, file)
            repo_path = os.path.relpath(local_path, local_dir)

            print(f"Uploading {repo_path} to branch {branch_name}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_name,
                repo_type=repo_type,
                commit_message=f"Upload {repo_path}",
                revision=branch_name,
            )
            print(f"Successfully uploaded {repo_path}!")


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str, help="Path to the checkpoint directory")
parser.add_argument("--repo-name", type=str, help="Name of the Hugging Face repository")
parser.add_argument("--branch-name", type=str, default="main", help="Branch name in the repository")
args = parser.parse_args()

converted_ckpt: str = args.ckpt_path
repo_name: str = args.repo_name
branch_name: str = args.branch_name

try:
    create_repo(repo_name, repo_type="model", private=True)
except Exception as e:
    print(f"Repository {repo_name} already exists! Error: {e}")

api = HfApi()
if branch_name != "main":
    try:
        api.create_branch(
            repo_id=repo_name,
            repo_type="model",
            branch=branch_name,
        )
    except Exception as e:
        print(f"Branch {branch_name} already exists. Error: {e}")

print(f"Starting upload of directory: {converted_ckpt}")
upload_directory(api, converted_ckpt, repo_name, "model", branch_name)
print("Upload completed successfully!")
