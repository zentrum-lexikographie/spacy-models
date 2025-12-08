from os import environ
from pathlib import Path

version = environ.get("CI_COMMIT_TAG", "v0.0.0").strip("v")

max_steps = int(environ.get("MAX_STEPS", "20000"))
gpu_id = int(environ.get("GPU_ID", "0"))
repo_url = environ.get("TWINE_REPO", "")

project_dir = Path(__file__).parent.parent
dataset_dir = project_dir / "dataset"
