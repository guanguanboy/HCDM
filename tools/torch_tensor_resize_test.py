import torch

from torchvision.transforms import Resize

input_tensor = output_tensor = torch.randn(1, 3, 256, 256)

torch_resize = Resize([128, 128])

#output_tensor = torch_resize(input_tensor)
#print(output_tensor.shape)

width=input_tensor.shape[2]
height = input_tensor.shape[3]

output_tensor_list = []
output_tensor_list.append(input_tensor)
for i in range(3):
    width = width//2
    height = height//2
    tensor = output_tensor
    torch_resize_fun = Resize([width,height])
    output_tensor = torch_resize_fun(tensor)
    output_tensor_list.insert(0, output_tensor)

for i in range(len(output_tensor_list)):
    print(output_tensor_list[i].shape)


def resize_tensor(input_tensor):
    width=input_tensor.shape[2]
    height = input_tensor.shape[3]
    output_tensor = input_tensor
    output_tensor_list = []
    output_tensor_list.append(output_tensor)
    for i in range(3):
        width = width//2
        height = height//2
        tensor = output_tensor
        torch_resize_fun = Resize([width,height])
        output_tensor = torch_resize_fun(tensor)
        output_tensor_list.insert(0, output_tensor)

    return output_tensor_list

output_list = resize_tensor(input_tensor)
for i in range(len(output_list)):
    print(output_list[i].shape)