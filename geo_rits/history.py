class History:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.train_losses = []
        self.train_entire_trajectory_errors = []
        self.train_masked_trajectory_errors = []
        self.val_losses = []
        self.val_entire_trajectory_errors = []
        self.val_masked_trajectory_errors = []

    def update(self, train_loss, train_entire_trajectory_error, train_masked_trajectory_error, val_loss, val_entire_trajectory_error, val_masked_trajectory_error):
        self.train_losses.append(train_loss)
        self.train_entire_trajectory_errors.append(train_entire_trajectory_error)
        self.train_masked_trajectory_errors.append(train_masked_trajectory_error)
        self.val_losses.append(val_loss)
        self.val_entire_trajectory_errors.append(val_entire_trajectory_error)
        self. val_masked_trajectory_errors.append(val_masked_trajectory_error)