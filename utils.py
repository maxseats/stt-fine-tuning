import os

def find_git_repo():
    """
    git 저장소의 루트 디렉토리를 찾습니다.
    Returns:
        str: git 저장소의 루트 디렉토리 경로. 저장소를 찾지 못한 경우 None을 반환합니다.
    """
    start_path = os.getcwd()

    current_path = start_path

    while current_path != os.path.dirname(current_path):
        if os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)
    
    return None

if __name__ == "__main__":
    repo_path = find_git_repo()
    if repo_path:
        print(f"Git repository found at: {repo_path}")
    else:
        print("No Git repository found.")