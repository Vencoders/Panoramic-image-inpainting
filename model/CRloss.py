import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_conv2d, batch_transposeconv2d


def hardmax(similar):
    val_max, id_max = torch.max(similar, 1)
    num = similar.size(1)
    sb = torch.Tensor(range(num)).long().to(similar.device)
    id_max = id_max[:, None, :, :]
    sb = sb[None, ..., None, None]
    similar = (sb==id_max).float().detach()
    return similar

class ReduceContextAttentionP1(nn.Module):
    def __init__(self, bkg_patch_size=4, stride=1, ufstride=1,
            softmax_scale=10., nn_hard=False, pd=1,
                 fuse_k=3, is_fuse=False,
                 th=0.5, norm_type=1, is_th=False):
        super(ReduceContextAttentionP1, self).__init__()
        self.bkg_patch_size = bkg_patch_size
        self.nn_hard = nn_hard
        self.stride = stride
        self.ufstride = ufstride
        self.softmax_scale = softmax_scale
        self.forward = self.forward_batch
        self.pd = pd
        self.fuse_k = fuse_k
        self.is_fuse = is_fuse
        self.th = th
        self.is_th = is_th
        self.norm_type = norm_type
        self.conv_1 = torch.nn.Conv3d(in_channels=961,out_channels=961*6,kernel_size=1,stride=1)
        #self.forward = self.forward_test

    def get_conv_kernel(self, x, mask=None): #x : 8 384 64 64
        batch, c, h_small, w_small = x.shape
        if self.norm_type == 1:
            x = x / torch.sqrt((x**2).sum(3, keepdim=True).sum(2, keepdim=True) + 1e-8)
        _x = F.pad(x, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        kernel = F.unfold(input=_x, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        kernel = kernel.transpose(1, 2) \
            .view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        kernel = self.conv_1(kernel)
        if self.norm_type == 2:
            kernel = kernel/ torch.sqrt(
                    (kernel**2).sum(3, keepdim=True).sum(4, keepdim=True)+1e-8)
        # b*hw*c*k*c
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        #m = m.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        m = m.transpose(1, 2).reshape(batch, 6,-1 , self.bkg_patch_size, self.bkg_patch_size)
        # m = m.squeeze(2)
        mm = (m.mean(4, keepdim=True).mean(3, keepdim=True)).float()
        #mm = (m.mean(3, keepdim=True).mean(2, keepdim=True)==1).float()
        return kernel, mm

    def forward_batch(self, f, b, mask=None):
        batch, c, h, w = b.shape
        batch, c, h_small, w_small = f.shape
        if mask is None:
            mask = torch.ones(batch, 1, h_small, w_small).to(f.device)
        else:
            mask = 1-mask
        # mask valid region
        softmax_scale = self.softmax_scale
        kernel, mmk = self.get_conv_kernel(b, mask)  #mmk:8 6 961 1 1  kernel:8 961 384 4 4 需要变成-> 8 961*6 384 4 4
        # mmk: valid ratio of each bkg patch 每个bkg有效补丁的比率
        _f = F.pad(f, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        cos_similar = batch_conv2d(_f, weight=kernel, stride=self.stride)
        _, cs, hs, ws = cos_similar.shape
        hb, wb = h//2, w//2

        if self.is_fuse:
            fuse_weight = torch.eye(self.fuse_k).to(f.device)
            fuse_weight = fuse_weight[None, None, ...]
            cos_similar = cos_similar.view(-1, cs, hs*ws)[:, None, ...]
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.reshape(batch, 1, cs, hs*ws)
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.squeeze(1)
            cos_similar = cos_similar.reshape(batch, cs, hs, ws)

        if self.is_th:
            mm = (mmk>self.th).float()
        else:
            _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
            m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), \
                         stride=self.stride)
            m = m.transpose(1, 2).reshape(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
            m = m.squeeze(2)
            mmp = (m.mean(3).mean(2)).float()
            mmp = mmp.view(batch, 6,-1, hs, ws) # mmp: valid ratio of fg patch 8 6 1 31 31   mmk:8 6 961 1 1
           # print(mmp,mmk)
            mm = (mmk>mmp).float()  # replace with more valid
            ppp = (mmp>self.th).float() # ppp: mask of partial valid
            mm = mm*ppp # partial valid being replaced with more valid
            mm = mm + (mmk==1).float().expand_as(mm)  # and full valid
            mm = (mm>0).float()
            mm = mm.view(mm.shape[0],-1,mm.shape[3],mm.shape[4])
        cos_similar = cos_similar * mm
        cos_similar = F.softmax(cos_similar*softmax_scale, dim=1)
        if self.nn_hard:
            cos_similar = hardmax(cos_similar)
        return cos_similar

class ReduceContextAttentionP2(nn.Module):
    def __init__(self, bkg_patch_size=16, stride=8, ufstride=8, pd=4, mk=True):
        super(ReduceContextAttentionP2, self).__init__()
        self.stride = stride
        self.bkg_patch_size = bkg_patch_size
        self.forward = self.forward_batch
        self.ufstride = ufstride
        self.pd = pd
        self.mk = mk
        #self.forward = self.forward_test
        self.stride_aux = stride
        self.aux_patch_size = bkg_patch_size
        self.ufstride_aux = ufstride
        self.conv3d_1 = torch.nn.Conv3d(in_channels=3969,out_channels=5766,kernel_size=1,stride=1)
        self.conv3d_2 = torch.nn.Conv3d(in_channels=23814,out_channels=5766,kernel_size=1,stride=1)
        self.conv3d_aux_kernel = torch.nn.Conv3d(in_channels=3969,out_channels=5766,kernel_size=1,stride=1)
        self.upscale_2 = torch.nn.ConvTranspose2d(1, 6, 2, 2, dilation=1)
        self.upscale_3 = torch.nn.ConvTranspose2d(1,1,2,2,dilation=1)
    def get_aux_kernel(self, b):
        batch,f, c, h, w = b.shape
        b = b.view(batch,-1,h,w)
        _b = F.pad(b, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        bkg_kernel = F.unfold(input=_b, kernel_size=(self.aux_patch_size, self.aux_patch_size),
                              stride=self.ufstride_aux)
        bkg_kernel = bkg_kernel.transpose(1, 2).reshape(batch, -1, c*6, self.aux_patch_size, self.aux_patch_size)
        bkg_kernel = self.conv3d_aux_kernel(bkg_kernel)
        return bkg_kernel

    def get_deconv_kernel(self, b, mask):
        batch, c, h, w = b.shape
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        msk_kernel = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        msk_kernel = msk_kernel.transpose(1, 2).reshape(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        msk_kernel = self.conv3d_2(msk_kernel)
        _b = F.pad(b, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        bkg_kernel = F.unfold(input=_b, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        bkg_kernel = bkg_kernel.transpose(1, 2).view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        bkg_kernel = self.conv3d_1(bkg_kernel)
        if self.mk:
            bkg_kernel = bkg_kernel*(1-msk_kernel)

        return bkg_kernel, msk_kernel

    def forward_batch(self, cos_similar, b, mask, dict_aux):
        # use original background for reconstruction
        _, _, hs, ws = cos_similar.shape
        bkg_kernel, msk_kernel = self.get_deconv_kernel(b, mask)
        #hard_similar = hardmax(cos_similar.detach())
        output = batch_transposeconv2d(cos_similar,
                                       weight=bkg_kernel,stride=self.stride)

        norm_kernel = torch.ones(1, 1, self.bkg_patch_size, self.bkg_patch_size).to(mask.device)
        weight_map = torch.ones(1, 1, hs, ws).to(mask.device)
        weight_map = F.conv_transpose2d(weight_map, norm_kernel, stride=self.stride)

        mask_recon = batch_transposeconv2d(cos_similar,
                                           weight=msk_kernel,stride=self.stride)
        weight_map_1 = self.upscale_3(weight_map)
        mask_recon = self.upscale_2(mask_recon)
        mask_recon = mask_recon / weight_map_1
        if self.pd > 0:
            output = output[:,:,self.pd:-self.pd,self.pd:-self.pd]
            mask_recon = mask_recon[:,:,self.pd:-self.pd,self.pd:-self.pd]
        recon_aux = {"hole":mask_recon}
        for k,v in dict_aux.items():
            hard_similar = hardmax(cos_similar)
            kernel = self.get_aux_kernel(v)
            recon = batch_transposeconv2d(hard_similar,
                                          weight=kernel,stride=self.stride_aux)
            recon = recon / weight_map
            if self.pd > 0:
                recon = recon[:,:,self.pd:-self.pd,self.pd:-self.pd]
            recon_aux[k] = recon
        recon_aux['raw'] = recon_aux['raw'].view(recon_aux['raw'].shape[0],-1,3,recon_aux['raw'].shape[-2],recon_aux['raw'].shape[-1])
        recon_aux['hole'] = recon_aux['hole'].view(recon_aux['hole'].shape[0], -1, 1, recon_aux['hole'].shape[-2],recon_aux['hole'].shape[-1])
        return output,recon_aux
