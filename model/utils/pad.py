import torch
from torch.autograd import Variable
from .convert import to_var

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def pad(tensor, length):
    if isinstance(tensor, Variable):
        var = tensor
        a=length-var.size(0)
        if length > var.size(0):
            return torch.cat([var,
                              torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])#原来有.cuda()方法
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat([tensor,
                              torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])#原来有.cuda()方法
        else:
            return tensor


def pad_and_pack(tensor_list):
    length_list = ([t.size(0) for t in tensor_list])
    max_len = max(length_list)
    padded = [pad(t, max_len) for t in tensor_list]
    packed = torch.stack(padded, 0)
    return packed, length_list
