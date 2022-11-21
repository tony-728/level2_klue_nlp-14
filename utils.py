import os


def create_directory(directory: str) -> bool:
    """
    디렉토리 경로를 받아서 없으면 생성한다.

    Parameters
    ----------
    directory : str
       디렉토리 경로
    Returns
    -------
    bool

    """
    try:
        if os.path.exists(directory):
            return True
        else:
            os.makedirs(directory)
            return True
    except OSError:
        print("Error: Failed to create the directory.")
        return False
