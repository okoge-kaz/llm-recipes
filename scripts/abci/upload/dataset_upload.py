import os
import argparse

from huggingface_hub import HfApi, create_repo


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str)
parser.add_argument("--repo-name", type=str)
parser.add_argument("--branch-name", type=str, default="main")
args = parser.parse_args()

dataset_path: str = args.dataset_path
repo_name: str = args.repo_name
branch_name: str = args.branch_name
try:
    create_repo(repo_name, repo_type="dataset", private=True)
except Exception as e:
    print(f"repo {repo_name} already exists! error: {e}")
    pass

files = os.listdir(dataset_path)

api = HfApi()
if branch_name != "main":
    try:
        api.create_branch(
            repo_id=repo_name,
            repo_type="dataset",
            branch=branch_name,
        )
    except Exception:
        print(f"branch {branch_name} already exists, try again...")
print(f"to upload: {files}")
for file in files:
    if os.path.isdir(os.path.join(dataset_path, file)):
        continue

    print(f"Uploading {file} to branch {branch_name}...")
    api.upload_file(
        path_or_fileobj=os.path.join(dataset_path, file),
        path_in_repo=file,
        repo_id=repo_name,
        repo_type="dataset",
        commit_message=f"Upload {file}",
        revision=branch_name,
    )
    print(f"Successfully uploaded {file} !")
