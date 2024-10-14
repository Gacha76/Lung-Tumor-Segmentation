import torch


class DiceLoss(torch.nn.Module):
    """
    class to compute the Dice Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denominator = pred.sum() + mask.sum() + 1e-8 # denominator
        dice_score = (2 * counter) / denominator

        return 1 - dice_score
    

class Precision(torch.nn.Module):
    """
    class to compute the precision
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denominator = pred.sum() + 1e-8 # denominator
        precision = torch.mean(counter / denominator)

        return precision
    

class Recall(torch.nn.Module):
    """
    class to compute the recall
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denominator = mask.sum() + 1e-8 # denominator
        recall = torch.mean(counter / denominator)

        return recall
    

class IoU(torch.nn.Module):
    """
    class to compute the Intersection over Union
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denominator = pred.sum() + mask.sum() - counter + 1e-8 # denominator
        iou = torch.mean(counter / denominator)

        return iou
