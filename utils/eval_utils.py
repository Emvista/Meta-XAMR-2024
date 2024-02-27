from pathlib import Path
import os
from settings import AMR_SCRIPT
import subprocess

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def save_predictions(pred, save_to: Path) -> Path:

    p_folder = save_to.parent
    p_folder.mkdir(parents=True, exist_ok=True)
    with open(save_to, 'w') as f:
        print(f'saving temp predictions to {save_to}')
        f.writelines("\n".join(pred))

    return save_to

def amr_postprocessing(pred_path: Path, sent_file: Path, silent=False):

    with cd(AMR_SCRIPT):
        postprocessed = pred_path.as_posix() + ".restore.final"
        if silent:
            subprocess.check_call(["python", "postprocess_AMRs.py", "-f", pred_path.as_posix(), "-s", sent_file.as_posix(), "-fo", "--no_wiki", "--mute"])
        else:
            subprocess.check_call(["python", "postprocess_AMRs.py", "-f", pred_path.as_posix(), "-s", sent_file.as_posix(), "-fo", "--no_wiki"])
        subprocess.check_call(["python", "reformat_single_amrs.py", "-f", postprocessed, "-e", ".form"])
