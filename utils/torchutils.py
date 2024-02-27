import torch
from pathlib import Path
import operator

class EarlyStopper:
    def __init__(self, patience=1):

        self.patience = patience
        self.counter = 0

    def __call__(self, increase_counter):

        if increase_counter:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                return True

        else:
            self.counter = 0

        return False

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, smaller_is_better):
        self.smaller_is_better = smaller_is_better
        self.best_step = 0

        if smaller_is_better:
            self.best_val = float('inf')
            self.better_than = operator.lt # less or equal than

        else:
            self.best_val = float('-inf')
            self.better_than = operator.gt # greater or equal than


    def __call__(self,
                 current_val,
                 current_step,
                 model,
                 optimizer,
                 train_dataloader=None,
                 save_to='outputs/best_model.pt'):

        model_save_to = Path(save_to)
        better_than_str = '<' if self.smaller_is_better else '>'

        if not model_save_to.parent.exists():
            model_save_to.parent.mkdir(parents=True)

        if self.better_than(current_val, self.best_val):

            print(f"\n current val {better_than_str} best val : {current_val} {better_than_str} {self.best_val}")
            self.best_val = current_val
            self.best_step = current_step
            print(f"\nSaving best model for step: {current_step}\n")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': current_step,
                'best_smatch': current_val,
                'early_stop_counter': 0,
                'best_step' : self.best_step
            }, model_save_to)

        else:
            print(f"\n best val {better_than_str} current val : {self.best_val} {better_than_str} {current_val} \n")
            print("Skipping model saving...")