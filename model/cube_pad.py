"""
Random mask generator

This code is based on https://github.com/hsientzucheng/CP-360-Weakly-Supervised-Saliency.git
"""
import os
import sys
import numpy as np

import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.dataset import PanoramaDataset, ToTensor, Normalize

def input_cp_setting(cube_map_image_set):
    # Process batch
    init_batch = np.array([])
    cube_map_image_set = cube_map_image_set.permute(0,2,3,1).contiguous()
    cube_map_image_set = cube_map_image_set.numpy()

    for idx in range(len(cube_map_image_set)):
        cube_img = cube_map_image_set[idx]
        print("cube_img_type:",type(cube_img))
        if idx in [1, 2]:
            cube_img = np.flip(cube_img, axis=1)
        if idx == 4:
            cube_img = np.flip(cube_img, axis=0)
        if idx == 0:
            init_batch = np.expand_dims(cube_img, axis=0)
        else:
            init_batch = np.concatenate((init_batch, np.expand_dims(cube_img, axis=0)), axis=0)
    init_batch = init_batch.astype(np.float32)
    cube_pad_class_input=init_batch

    return cube_pad_class_input

def get_pad_size(lrtd_pad):
    if type(lrtd_pad) == np.int:
        p_l = lrtd_pad
        p_r = lrtd_pad
        p_t = lrtd_pad
        p_d = lrtd_pad
    else:
        [p_l, p_r, p_t, p_d] = lrtd_pad
    return p_l, p_r, p_t, p_d


class CubePad(nn.Module):
    def __init__(self, lrtd_pad, use_gpu=True):
        super(CubePad, self).__init__()
        self.CP = CubePadding(lrtd_pad, use_gpu)

    def forward(self, x):
        """
            Input shape:  [6N, C, H, W]
            Output shape: [6N, C, H + (top down padding), W + (left right padding)]
        """
        if x.size()[0] % 6 != 0:
            print('CubePad size mismatch!')
            exit()
        batch_size = int(x.size()[0]/6)
        tmp = []
        for i in range(batch_size):
            patch = x[i*6:i*6+6, :, :, :]
            tmp.append(self.CP(patch))
        result = torch.cat(tmp, dim=0)
        return result



class CubePadding(nn.Module):
    """
        Cube padding support astmetric padding and rectangle input

        Order of cube faces: 123456 => bdflrt (back, bottom, front, left, right, top)
        The surrounding volume of cube padding includes 4 concatenated plates

                                  //＝＝＝//|
        4 plates (t, d, l, r):   //  t  // |
                                ||＝＝＝|| r|
                               l||  f  || /
                                ||＝＝＝||/
                                   d
    """

    def __init__(self, lrtd_pad, use_gpu=True):
        super(CubePadding, self).__init__()
        self.use_gpu = use_gpu
        #self.pad = pad
        if type(lrtd_pad) == np.int:
            self.p_l = lrtd_pad
            self.p_r = lrtd_pad
            self.p_t = lrtd_pad
            self.p_d = lrtd_pad
        else:
            [self.p_l, self.p_r, self.p_t, self.p_d] = lrtd_pad

    def flip(self, tensor, dim):

        idx = [i for i in range(tensor.size(dim)-1, -1, -1)]

        if self.use_gpu:
            idx = Variable(torch.cuda.LongTensor(idx))
        else:
            idx = Variable(torch.LongTensor(idx))

        inverted_tensor = tensor.index_select(dim, idx)
        return inverted_tensor

    def make_cubepad_edge(self, feat_td, feat_lr):
        td_pad = feat_td.size(2)
        lr_pad = feat_lr.size(3)

        if td_pad > lr_pad:
            return feat_lr.repeat(1, 1, td_pad, 1)
        else:
            return feat_td.repeat(1, 1, 1, lr_pad)

        #avg_feat = (tile_lr+tile_td)*0.5
        # return avg_feat

    def forward(self, x):
        #print("cube padding x shape",x.shape)
        """
            Input shape:  [6, C, H, W]
            Output shape: [6, C, H + p_t + p_d, W + p_l + p_r]
            Method: Create 4 plates -> Create corners -> Concatenate
        """
        p_l = self.p_l
        p_r = self.p_r
        p_t = self.p_t
        p_d = self.p_d

        # F R B L T D
        f_front = x[0]
        f_right = x[1]
        f_back = x[2]
        f_left = x[3]
        f_top = x[4]
        f_down = x[5]

        # f_back = x[0]
        # f_down = x[1]
        # f_front = x[2]
        # f_left = x[3]
        # f_right = x[4]
        # f_top = x[5]

        # Construct top, down, left, right padding volume if needed
        if p_t != 0:
            _t12 = torch.cat(
                [torch.unsqueeze(f_top[:, -p_t:, :], 0),
                 torch.unsqueeze(self.flip(f_top[:, :, -p_t:].permute(0, 2, 1),2), 0)], 0)# turn right 90 degrees
            _t123 = torch.cat(
                [_t12, torch.unsqueeze(self.flip(f_top[:, :p_t, :],2), 0)],0) # turn right 180 degrees
            _t1234 = torch.cat(
                [_t123, torch.unsqueeze(self.flip(f_top[:, :, :p_t].permute(0, 2, 1),1), 0)], 0) # turn left 90 degrees
            _t12345 = torch.cat(
                [_t1234, torch.unsqueeze(self.flip(self.flip(f_back[:, :p_t, :],2),1), 0)], 0)
            _t123456 = torch.cat(
                [_t12345, torch.unsqueeze(f_front[:, -p_t:, :], 0)], 0)

        if p_d != 0:
            _d12 = torch.cat(
                [torch.unsqueeze(f_down[:, :p_d, :], 0),
                 torch.unsqueeze(self.flip(f_down[:, :, -p_d:].permute(0, 2, 1),1), 0)], 0) # turn left 90 degrees
            _d123 = torch.cat(
                [_d12, torch.unsqueeze(self.flip(self.flip(f_down[:, -p_d:, :], 1),2),0)], 0) # turn left 180 degrees
            _d1234 = torch.cat([_d123, torch.unsqueeze(self.flip(f_down[:, :, :p_d].permute(0, 2, 1),2), 0)], 0) # turn right 90 degrees
            _d12345 = torch.cat([_d1234, torch.unsqueeze(f_front[:, :p_d, :], 0)], 0)
            _d123456 = torch.cat(
                [_d12345, torch.unsqueeze(self.flip(self.flip(f_back[:, -p_d:, :],2),1), 0)], 0)


        if p_l != 0:
            _l12 = torch.cat(
                [torch.unsqueeze(f_left[:, :, -p_l:], 0),
                 torch.unsqueeze(f_front[:, :, -p_l:], 0)], 0)
            _l123 = torch.cat(
                [_l12, torch.unsqueeze(f_right[:, :, -p_l:], 0)], 0)
            _l1234 = torch.cat(
                [_l123, torch.unsqueeze(f_back[:, :, -p_l:], 0)], 0)
            _l12345 = torch.cat(
                [_l1234, torch.unsqueeze(f_left[:, :p_l, :].permute(0, 2, 1), 0)], 0)
            _l123456 = torch.cat(
                [_l12345, torch.unsqueeze(self.flip(f_left[:, -p_l:, :].permute(0, 2, 1),1), 0)], 0)

        if p_r != 0:
            _r12 = torch.cat(
                [torch.unsqueeze(f_right[:, :, :p_r], 0),
                 torch.unsqueeze(f_back[:, :, :p_r], 0)], 0)
            _r123 = torch.cat(
                [_r12, torch.unsqueeze(f_left[:, :, :p_r], 0)], 0)
            _r1234 = torch.cat(
                [_r123, torch.unsqueeze(f_front[:, :, :p_r], 0)], 0)
            _r12345 = torch.cat(
                [_r1234, torch.unsqueeze(self.flip(f_right[:, :p_r, :].permute(0, 2, 1), 1), 0)], 0)
            _r123456 = torch.cat(
                [_r12345, torch.unsqueeze(self.flip(f_right[:, -p_r:, :],1).permute(0, 2, 1), 0)], 0)

        # For edge corner
        if p_r != 0 and p_t != 0:
            p_tr = self.make_cubepad_edge(
                _t123456[:, :, -p_t:, -1:], _r123456[:, :, :1, :p_r])
        if p_t != 0 and p_l != 0:
            p_tl = self.make_cubepad_edge(
                _t123456[:, :, :p_t, :1], _l123456[:, :, :1, :p_l])
        if p_d != 0 and p_r != 0:
            p_dr = self.make_cubepad_edge(
                _d123456[:, :, -p_d:, -1:], _r123456[:, :, -1:, -p_r:])
        if p_d != 0 and p_l != 0:
            p_dl = self.make_cubepad_edge(
                _d123456[:, :, :p_d, :1], _l123456[:, :, -1:, -p_l:])

        # Concatenate each padding volume
        if p_r != 0:
            _rp123456p = _r123456
            if 'p_tr' in locals():
                _rp123456 = torch.cat([p_tr, _r123456], 2)
            else:
                _rp123456 = _r123456

            if 'p_dr' in locals():
                _rp123456p = torch.cat([_rp123456, p_dr], 2)
            else:
                _rp123456p = _rp123456
        if p_l != 0:
            _lp123456p = _l123456
            if 'p_tl' in locals():
                _lp123456 = torch.cat([p_tl, _l123456], 2)
            else:
                _lp123456 = _l123456
            if 'p_dl' in locals():
                _lp123456p = torch.cat([_lp123456, p_dl], 2)
            else:
                _lp123456p = _lp123456
        if p_t != 0:
            t_out = torch.cat([_t123456, x], 2)
        else:
            t_out = x
        if p_d != 0:
            td_out = torch.cat([t_out, _d123456], 2)
        else:
            td_out = t_out
        if p_l != 0:
            tdl_out = torch.cat([_lp123456p, td_out], 3)
        else:
            tdl_out = td_out
        if p_r != 0:
            tdlr_out = torch.cat([tdl_out, _rp123456p], 3)
        else:
            tdlr_out = tdl_out
        return tdlr_out

class ZeroPad(nn.Module):
    """ This ZeroPad is for compuational efficiency experiment only"""
    def __init__(self, lrtd_pad, use_gpu=True):
        super(ZeroPad, self).__init__()
        self.use_gpu = use_gpu
        self.lrtd_pad = lrtd_pad

    def forward(self, x, lrtd_pad=None):
        if lrtd_pad is None:
            self.p_l, self.p_r, self.p_t, self.p_d = get_pad_size(self.lrtd_pad)
        else:
            self.p_l, self.p_r, self.p_t, self.p_d = get_pad_size(lrtd_pad)

        if self.use_gpu:
            pad_row_t = Variable(torch.FloatTensor(x.size(0), x.size(1), self.p_t, x.size(3)).zero_()).cuda()
            pad_row_d = Variable(torch.FloatTensor(x.size(0), x.size(1), self.p_d, x.size(3)).zero_()).cuda()
            pad_col_l = Variable(torch.FloatTensor(x.size(0), x.size(1), x.size(2) + self.p_t + self.p_d,
                                                   self.p_l).zero_()).cuda()
            pad_col_r = Variable(torch.FloatTensor(x.size(0), x.size(1), x.size(2) + self.p_t + self.p_d,
                                                   self.p_r).zero_()).cuda()
        else:
            pad_row_t = Variable(torch.FloatTensor(x.size(0), x.size(1), self.p_t, x.size(3)).zero_())
            pad_row_d = Variable(torch.FloatTensor(x.size(0), x.size(1), self.p_d, x.size(3)).zero_())
            pad_col_l = Variable(torch.FloatTensor(x.size(0), x.size(1), x.size(2) + self.p_t + self.p_d,
                                                   self.p_l).zero_())
            pad_col_r = Variable(torch.FloatTensor(x.size(0), x.size(1), x.size(2) + self.p_t + self.p_d,
                                                   self.p_r).zero_())

        if self.p_t != 0: x = torch.cat((pad_row_t, x), 2)
        if self.p_d != 0: x = torch.cat((x, pad_row_d), 2)
        if self.p_l != 0: x = torch.cat((pad_col_l, x), 3)
        if self.p_r != 0: x = torch.cat((x, pad_col_r), 3)

        return x

def im_norm(in_img, mean, std):
    out_img = in_img
    out_img[:, :, 0] = (in_img[:, :, 0] - mean[0]) / std[0]
    out_img[:, :, 1] = (in_img[:, :, 1] - mean[1]) / std[1]
    out_img[:, :, 2] = (in_img[:, :, 2] - mean[2]) / std[2]
    return out_img

import cv2
import os

if __name__ == '__main__':
    transforms = transforms.Compose([Normalize(), ToTensor()])
    transformed_dataset = PanoramaDataset(in_dir='/***', transform=transforms)
    train_loader = DataLoader(transformed_dataset, batch_size=4, shuffle=False)
    
    for x in train_loader:
        for bn in range(x['image'].shape[0]):
            img = input_cp_setting(x['image'][bn])
            img = torch.Tensor(img)
            img = img.permute(0, 3, 1, 2).contiguous()
            img = Variable(img)
            cp = CubePad(2,use_gpu=True)
            afterimg = cp(img)
            afterimg = afterimg.permute(0,2,3,1).contiguous()
            for kk in range(6):
                save_pad_img = np.asarray(afterimg[kk]*255).astype('uint16')
                cv2.imshow("afterimage",save_pad_img)
                cv2.waitKey(0)
