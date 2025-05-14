import os
import sys

from huggingface_hub import HfApi


def main():
    """Main deployment function"""
    username = os.environ.get("HF_USERNAME")
    space_name = os.environ.get("HF_SPACE_NAME")
    docker_image = os.environ.get("DOCKER_IMAGE").lower()

    if not all([username, space_name, docker_image]):
        print("Error: Missing required environment variables")
        print("Make sure HF_USERNAME, HF_SPACE_NAME, and DOCKER_IMAGE are set")
        sys.exit(1)

    api = HfApi()

    try:
        space_info = api.get_space_info(repo_id=f"{username}/{space_name}")
        print(f"Space exists: {space_info.name}")
    except Exception:
        print(f"Creating new space: {username}/{space_name}")
        api.create_repo(
            repo_id=f"{username}/{space_name}",
            repo_type="space",
            space_sdk="docker",
            private=False,
        )

    with open("Dockerfile", "w") as f:
        f.write(f"FROM {docker_image}\n")

    api.upload_file(
        path_or_fileobj="Dockerfile",
        path_in_repo="Dockerfile",
        repo_id=f"{username}/{space_name}",
        repo_type="space",
    )

    print(
        f"Successfully deployed to https://huggingface.co/spaces/{username}/{space_name}"
    )


if __name__ == "__main__":
    main()
