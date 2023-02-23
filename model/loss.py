import torch.nn.functional as F

num_classes = [74874, 134415, 25459, 14090, 19754]

w_list = [max(num_classes) / num_class for num_class in num_classes]


def robust_loss(output, target):
    index = int(target.tolist())
    w = w_list[index]
    return w * F.cross_entropy(output, target)
