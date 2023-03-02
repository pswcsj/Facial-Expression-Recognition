import torch.nn.functional as F
import torch

num_classes = [74874, 134415, 25459, 14090, 19754]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
w_list = torch.FloatTensor([max(num_classes) / num_class for num_class in num_classes]).to(device)

def robust_loss(output, target):
    target = target.type(torch.LongTensor).to(device)
    return F.cross_entropy(output, target, weight=w_list)
