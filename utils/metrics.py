from time import time


class AverageRecord(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class LossRecord:
    """
    A class for keeping track of loss values during training.

    Attributes:
        start_time (float): The time when the LossRecord was created.
        loss_list (list): A list of loss names to track.
        loss_dict (dict): A dictionary mapping each loss name to an AverageRecord object.
    """

    def __init__(self, loss_list):
        self.start_time = time()
        self.loss_list = loss_list
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}
    
    def update(self, update_dict, n):
        for key, value in update_dict.items():
            self.loss_dict[key].update(value, n)
    
    def format_metrics(self):
        result = ""
        for loss in self.loss_list:
            result += "{}: {:.8f} | ".format(loss, self.loss_dict[loss].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result
    
    def to_dict(self):
        return {
            loss: self.loss_dict[loss].avg for loss in self.loss_list
        }
    
    def __str__(self):
        return self.format_metrics()
    
    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg
