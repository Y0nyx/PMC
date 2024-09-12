from clearml import Task

def on_train_epoch_end(trainer):
    """Logs debug samples for the first epoch of YOLO training and report current training progress."""
    if task := Task.current_task():
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            if k == "train/box_loss":
                task.get_logger().report_scalar("box loss", k, v, iteration=trainer.epoch)
            if k == "train/cls_loss":
                task.get_logger().report_scalar("cls loss", k, v, iteration=trainer.epoch)


def on_fit_epoch_end(trainer):
    """Reports model information to logger at the end of an epoch."""
    if task := Task.current_task():
        for k, v in trainer.metrics.items():
            if k == "val/box_loss":
                task.get_logger().report_scalar("box loss", k, v, iteration=trainer.epoch)
            if k == "val/cls_loss":
                task.get_logger().report_scalar("cls loss", k, v, iteration=trainer.epoch)
