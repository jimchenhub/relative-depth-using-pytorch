import torch
from torch import nn

class relative_depth_crit(nn.Module):

    def __loss_func_arr(self, z_A, z_B, ground_truth):
        mask = torch.abs(ground_truth)
        z_A = z_A[0]
        z_B = z_B[0]
        return mask*torch.log(1+torch.exp(-ground_truth*(z_A-z_B)))+(1-mask)*(z_A-z_B)**2

    def __init__(self, ranking=False):
        super(relative_depth_crit, self).__init__()
        self.ranking = ranking  # proposed by CVPR-2018 Ke Xian

    def forward(self, input, target):
        self.input = input
        self.target = target
        self.output = torch.Tensor([0]).cuda()
        n_point_total = 0
        cpu_input = input

        all_z_A_arr = torch.Tensor([]).to(torch.device("cuda:0"))
        all_z_B_arr = torch.Tensor([]).to(torch.device("cuda:0"))
        all_ground_truth_arr = torch.Tensor([]).to(torch.device("cuda:0"))
        for batch_idx in range(0,cpu_input.size()[0]):
            n_point_total += target[batch_idx]['n_point']

            x_A_arr = target[batch_idx]['x_A']
            y_A_arr = target[batch_idx]['y_A']
            x_B_arr = target[batch_idx]['x_B']
            y_B_arr = target[batch_idx]['y_B']

            batch_input = cpu_input[batch_idx, 0]
            z_A_arr = batch_input.index_select(1, x_A_arr.long()).gather(0, y_A_arr.view(1,-1).long())
            z_B_arr = batch_input.index_select(1, x_B_arr.long()).gather(0, y_B_arr.view(1,-1).long())
            ground_truth_arr = target[batch_idx]['ordianl_relation']

            all_z_A_arr = torch.cat((all_z_A_arr, z_A_arr), 1)
            all_z_B_arr = torch.cat((all_z_B_arr, z_B_arr), 1)
            all_ground_truth_arr = torch.cat((all_ground_truth_arr, ground_truth_arr), 0)

        # all_x_A_arr.
        if self.ranking is False:
            self.output = torch.sum(self.__loss_func_arr(all_z_A_arr, all_z_B_arr, all_ground_truth_arr))
        else:
            # ignore last 25% data
            pass

        return self.output/n_point_total


if __name__ == '__main__':
    crit = relative_depth_crit()
    print(crit)
    x = torch.zeros(2,1,6,6).cuda()
    target = {}
    target[0] = {}
    target[0]['x_A'] = torch.Tensor([0,1,2,3,4,5]).cuda()
    target[0]['y_A'] = torch.Tensor([0,1,2,3,4,5]).cuda()
    target[0]['x_B'] = torch.Tensor([0,0,0,0,0,0]).cuda()
    target[0]['y_B'] = torch.Tensor([5,4,3,2,1,0]).cuda()
    target[0]['ordianl_relation'] = torch.Tensor([-1,0,1,1,-1,-1]).cuda()
    target[0]['n_point'] = 6
    target[1] = {}
    target[1]['x_A'] = torch.Tensor([0, 1, 2, 3, 4, 5]).cuda()
    target[1]['y_A'] = torch.Tensor([0, 1, 1, 3, 4, 5]).cuda()
    target[1]['x_B'] = torch.Tensor([0, 0, 0, 0, 0, 0]).cuda()
    target[1]['y_B'] = torch.Tensor([5, 4, 3, 2, 1, 0]).cuda()
    target[1]['ordianl_relation'] = torch.Tensor([0, 0, 1, 1, -1, -1]).cuda()
    target[1]['n_point'] = 6
    loss = crit.forward(x,target)
    print(loss)
    loss.backward()
    print(x.grad)
