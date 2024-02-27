from typing import List
from pathlib import Path
import os
import re

"""
code largely borrowed from transformers trainer 
"""

class CheckpointHandler:
    def __init__(self, save_total_limit, checkpoint_dir):
        self.save_total_limit = save_total_limit
        self.checkpoint_dir = checkpoint_dir

    def sorted_checkpoints(
        self, use_mtime=True, best_model_checkpoint=None
    ) -> List[str]:

        checkpoint_prefix = "checkpoint"
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.checkpoint_dir).glob(f"{checkpoint_prefix}-*") if os.path.exists(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

        # Make sure we don't delete the best model.
        if best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]

        return checkpoints_sorted

    def rotate_checkpoints(self, use_mtime=True, best_model_checkpoint=None) -> None:
        if self.save_total_limit is None or self.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self.sorted_checkpoints(use_mtime=use_mtime, best_model_checkpoint=best_model_checkpoint)
        if len(checkpoints_sorted) <= self.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.

        save_total_limit = self.save_total_limit
        if (
            best_model_checkpoint is not None
            and self.save_total_limit == 1
            and checkpoints_sorted[-1] != best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            print(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            os.remove(checkpoint)
