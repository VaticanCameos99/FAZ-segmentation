import torch

def dice_loss(output, gt, smooth = 1):
    #output = torch.nn.Softmax()(output)

    intersection = torch.sum(output * gt, axis = [1, 2, 3])
    union = torch.sum(gt, axis = [1, 2, 3]) + torch.sum(output, axis = [1, 2, 3])
    dice = torch.mean((2 * intersection + smooth) / (union + smooth), axis = 0)
    return dice

metrics = {
    'dice_loss': dice_loss,
    #IoU -> measures something similar to worst case performance
    #Dice -> measures something similar to average case performance
} 

