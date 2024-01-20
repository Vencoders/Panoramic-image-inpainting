import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
from .CRloss import ReduceContextAttentionP1, ReduceContextAttentionP2
from .layers import ContextualAttention
from .utils import gen_conv, gen_deconv


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class FaceGenerator(BaseNetwork):
    def __init__(self, in_channels, out_channels, nker, norm="bnorm", relu=True, init_weights=True):
        super(FaceGenerator, self).__init__()
        self.relu = relu
        self.enc_gate_1 = layers.GatedConv2d(in_channels, 1 * nker, kernel_size=4, stride=2, padding=1,
                                             norm='in', activation='lrelu', pad_type='zero')
        # self.enc1 = layers.CBR2d(in_channels, 1 * nker, kernel_size=4, padding=1,
        #                          norm=None, relu=self.relu,  stride=2)
        self.enc_gate_2 = layers.GatedConv2d(1 * nker, 2 * nker, kernel_size=4, stride=2, padding=1,
                                             norm='in', activation='lrelu', pad_type='zero')
        #self.enc2 = layers.CBR2d(1 * nker, 2 * nker, kernel_size=4, padding=1,
        #                        norm=norm, relu=self.relu, stride=2)
        self.enc_gate_3 = layers.GatedConv2d(2 * nker, 4 * nker, kernel_size=4, stride=2, padding=1,
                                             norm='in', activation='lrelu', pad_type='zero')
        #self.enc3 = layers.CBR2d(2 * nker, 4 * nker, kernel_size=4, padding=1,
        #                         norm=norm, relu=self.relu,  stride=2)
        self.enc_gate_4 = layers.GatedConv2d(4 * nker, 4 * nker, kernel_size=4, stride=2, padding=1,
                                             norm='in', activation='lrelu', pad_type='zero')
        #self.enc4 = layers.CBR2d(4 * nker, 4 * nker, kernel_size=4, padding=1,
        #                         norm=norm, relu=self.relu,  stride=2)
        self.enc_gate_5 = layers.GatedConv2d(4 * nker, 8 * nker, kernel_size=4, stride=2, padding=1,
                                             norm='in', activation='lrelu', pad_type='zero')
        #self.enc5 = layers.CBR2d(4 * nker, 8 * nker, kernel_size=4, padding=1,
        #                         norm=norm, relu=self.relu,  stride=2)
        self.enc_gate_6 = layers.GatedConv2d(8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1,
                                             norm='in', activation='lrelu', pad_type='zero')
        #self.enc6 = layers.CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1,
        #                         norm=norm, relu=self.relu, stride=2)

        self.dec1_gate = layers.TransposeGatedConv2d(8 * nker,  8 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)
        #self.dec1 = layers.DECBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1,
        #                           norm=norm, relu=self.relu, stride=2)
        self.drop1 = nn.Dropout2d(0.2)

        self.dec2_gate = layers.TransposeGatedConv2d(2  * 8 * nker,  4 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)

        #self.dec2 = layers.DECBR2d(2 * 8 * nker, 4 * nker, kernel_size=4, padding=1,
        #                           norm=norm, relu=self.relu,  stride=2)
        self.drop2 = nn.Dropout2d(0.2)

        self.dec3_gate = layers.TransposeGatedConv2d(2 * 4 * nker, 4 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)

        #self.dec3 = layers.DECBR2d(2 * 4 * nker, 4 * nker, kernel_size=4, padding=1,
        #                           norm=norm, relu=self.relu,  stride=2)

        self.drop3 = nn.Dropout2d(0.5)
        self.dec4_gate = layers.TransposeGatedConv2d(2 * 4 * nker, 2 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)
        #self.dec4 = layers.DECBR2d(2 * 4 * nker, 2 * nker, kernel_size=4, padding=1,
        #                        norm=norm, relu=self.relu,  stride=2)
        self.drop4 = nn.Dropout2d(0.5)
        self.dec5_gate = layers.TransposeGatedConv2d(2 * 2 * nker, 1 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)

        #self.dec5 = layers.DECBR2d(2 * 2 * nker, 1 * nker, kernel_size=4, padding=1,
        #                        norm=norm, relu=self.relu,  stride=2)

        self.dec6_gate = layers.TransposeGatedConv2d(2 * 1 * nker, out_channels, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)

        #self.dec6 = layers.DECBR2d(2 * 1 * nker, out_channels, kernel_size=4, padding=1,
        #                        norm=None, relu=self.relu, stride=2)
        if init_weights:
                self.init_weights()
                print("FaceGenerator init weights.")

    def forward(self, g4f, m4f, cm4f):
        # x_cube_mask + cube_mask -> B,4,H,W*4
        #cm4f ： B c H W  带有mask的4面
        st1_input = torch.cat((cm4f, m4f), dim=1)

        enc1 = self.enc_gate_1(st1_input)
        enc2 = self.enc_gate_2(enc1)
        enc3 = self.enc_gate_3(enc2)
        enc4 = self.enc_gate_4(enc3)
        enc5 = self.enc_gate_5(enc4)
        enc6 = self.enc_gate_6(enc5)
        # enc1 = self.enc1(st1_input)
        # enc2 = self.enc2(enc1)
        # enc3 = self.enc3(enc2)
        # enc4 = self.enc4(enc3)
        # enc5 = self.enc5(enc4)
        # enc6 = self.enc6(enc5)

        # dec1 = self.dec1(enc6)
        dec1 = self.dec1_gate(enc6)
        drop1 = self.drop1(dec1)

        cat2 = torch.cat((drop1, enc5), dim=1)
        dec2 = self.dec2_gate(cat2)
        # dec2 = self.dec2(cat2)
        drop2 = self.drop2(dec2)

        cat3 = torch.cat((drop2, enc4), dim=1)
        # dec3 = self.dec3(cat3)
        dec3 = self.dec3_gate(cat3)
        cat4 = torch.cat((dec3, enc3), dim=1)
        # dec4 = self.dec4(cat4)
        dec4 = self.dec4_gate(cat4)

        cat5 = torch.cat((dec4, enc2), dim=1)
        dec5 = self.dec5_gate(cat5)
        # dec5 = self.dec5(cat5)
        cat6 = torch.cat((dec5, enc1), dim=1)
        # dec6 = self.dec6(cat6)
        dec6 = self.dec6_gate(cat6)
        face_stage = torch.tanh(dec6)

        return face_stage

class FaceDis(BaseNetwork):
    def __init__(self, in_channels, out_channels, nker, norm="bnorm", relu=True, init_weights=True):
        super(FaceDis, self).__init__()
        self.relu = relu
        self.enc1 = layers.CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=0.2, bias=False)

        self.enc2 = layers.CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = layers.CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = layers.CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = layers.CBR2d(8 * nker, out_channels, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=None, bias=False)
        if init_weights:
            self.init_weights()
            print("FaceDis init weights.")

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = torch.sigmoid(x)

        return x



class CubeGenerator(BaseNetwork):
    def __init__(self, in_channels, out_channels, nker, norm="bnorm", init_weights=True, activation='lrelu',relu= True):
        super(CubeGenerator, self).__init__()
        # self.relu = relu
        # self.enc1 = layers.CBR2d(6 * in_channels, 6 * 1 * nker, kernel_size=4, padding=1,
        #                          norm=None, relu=self.relu, stride=2)
        self.enc_gate_1 = layers.GatedConv2d(6 * in_channels, 6 * 1 * nker, kernel_size=4, stride=2, padding=1,
                                 norm='in', activation='lrelu', pad_type='zero')
        # self.enc2 = layers.CBR2d(6 * 1 * nker, 6 * 2 * nker, kernel_size=4, padding=1,
        #                          norm=norm, relu=self.relu,  stride=2)
        self.enc2_gate_2 = layers.GatedConv2d(6*1* nker, 6 * 2 * nker, kernel_size=4, stride=2, padding=1,
                                 norm='in', activation='lrelu',pad_type='zero')
        # self.enc3 = layers.CBR2d(6 * 2 * nker, 6 * 4 * nker, kernel_size=4, padding=1,
        #                          norm=norm, relu=self.relu,  stride=2)
        self.enc3_gate_3 = layers.GatedConv2d(6 * 2 * nker, 6 * 4 * nker, kernel_size=4, stride=2, padding=1,
                                              norm='in', activation='lrelu',pad_type='zero')
        # self.enc4 = layers.CBR2d(6 * 4 * nker, 6 * 4 * nker, kernel_size=4, padding=1,
        #                          norm=norm, relu=self.relu,  stride=2)
        self.enc4_gate_4 = layers.GatedConv2d(6 * 4 * nker, 6 * 4 * nker, kernel_size=4, stride=2, padding=1,
                                              norm='in', activation='lrelu',pad_type='zero')
        # self.enc5 = layers.CBR2d(6 * 4 * nker, 6 * 8 * nker, kernel_size=4, padding=1,
        #                          norm=norm, relu=self.relu,  stride=2)
        self.enc5_gate_5 = layers.GatedConv2d(6 * 4 * nker, 6 * 8 * nker, kernel_size=4, stride=2, padding=1,
                                              norm='in', activation='lrelu',pad_type='zero')
        # self.enc6 = layers.CBR2d(6 * 8 * nker, 6 * 8 * nker, kernel_size=4, padding=1,
        #                          norm=norm, relu=self.relu,  stride=2)
        self.enc6_gate_6 = layers.GatedConv2d(6 * 8 * nker, 6 * 8 * nker, kernel_size=4, stride=2, padding=1,
                                              norm='in', activation='lrelu',pad_type='zero')

        #self.dec1 = layers.DECBR2d(6 * 8 * nker, 6 * 8 * nker, kernel_size=4, padding=1,
        #                            norm=norm, relu=relu,  stride=2)
        self.dec1_gate_1 = layers.TransposeGatedConv2d(6 * 8 * nker, 6 * 8 * nker, kernel_size=3, padding=1,
                                   norm='in', activation='lrelu',  stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        #self.dec2 = layers.DECBR2d(2 * 6 * 8 * nker, 6 * 4 * nker, kernel_size=4, padding=1,
         #                           norm=norm, relu=relu,  stride=2)
        self.dec2_gate_2 = layers.TransposeGatedConv2d(2* 6 * 8 * nker, 6 * 4* nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)
        self.drop2 = nn.Dropout2d(0.2)
        #self.dec3 = layers.DECBR2d(2 * 6 * 4 * nker, 6 * 4 * nker, kernel_size=4, padding=1,
        #                            norm=norm, relu=relu,  stride=2)
        self.dec3_gate_3 = layers.TransposeGatedConv2d(2 * 6 * 4 * nker, 6 * 4 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)
        self.drop3 = nn.Dropout2d(0.5)
        #self.dec4 = layers.DECBR2d(2 * 6 * 4 * nker, 6 * 2 * nker, kernel_size=4, padding=1,
        #                         norm=norm, relu=relu,  stride=2)
        self.dec4_gate_4 = layers.TransposeGatedConv2d(2 * 6 * 4 * nker, 6 * 2 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)
        self.drop4 = nn.Dropout2d(0.5)
       # self.dec5 = layers.DECBR2d(2 * 6 * 2 * nker, 6 * 1 * nker, kernel_size=3, padding=1,
        #                         norm=norm, relu=relu,  stride=2)
        self.dec5_gate_5 = layers.TransposeGatedConv2d(2 * 6 * 2 * nker, 6 * 1 * nker, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)

        #self.dec6 = layers.DECBR2d(2 * 6 * 1 * nker, 6 * out_channels, kernel_size=4, padding=1,
        #                         norm=None, relu=relu,  stride=2)
        self.dec6_gate_6 = layers.TransposeGatedConv2d(2 * 6 * 1 * nker, 6 * out_channels, kernel_size=3, padding=1,
                                                       norm='in', activation='lrelu', stride=1)
        self.crloss =Crloss()

        if init_weights:
                self.init_weights()
                print("CubeGenerator init weights.")

    def forward(self, x,cube,masks,isTrain = False):
        # enc1 = self.enc1(x)
        enc1 = self.enc_gate_1(x)
        enc2 = self.enc2_gate_2(enc1)
        # enc2 = self.enc2(enc1)
        # enc3 = self.enc3(enc2)
        enc3 = self.enc3_gate_3(enc2)
        # enc4 = self.enc4(enc3)
        enc4 = self.enc4_gate_4(enc3)
        # enc5 = self.enc5(enc4)
        enc5 = self.enc5_gate_5(enc4)
        # enc6 = self.enc6(enc5)
        enc6 = self.enc6_gate_6(enc5)

        # dec1 = self.dec1(enc6)
        dec1 = self.dec1_gate_1(enc6)
        drop1 = self.drop1(dec1)

        cat2 = torch.cat((drop1, enc5), dim=1)
        # dec2 = self.dec2(cat2)
        dec2 = self.dec2_gate_2(cat2)
        drop2 = self.drop2(dec2)

        cat3 = torch.cat((drop2, enc4), dim=1)
        # dec3 = self.dec3(cat3)
        dec3 = self.dec3_gate_3(cat3)

        cat4 = torch.cat((dec3, enc3), dim=1)
        # dec4 = self.dec4(cat4)
        dec4 = self.dec4_gate_4(cat4)

        cat5 = torch.cat((dec4, enc2), dim=1)
        # dec5 = self.dec5(cat5)
        dec5 = self.dec5_gate_5(cat5)
        cat6 = torch.cat((dec5, enc1), dim=1)
        # dec6 = self.dec6(cat6)
        dec6 = self.dec6_gate_6(cat6)

        cube_stage = torch.tanh(dec6)


        if isTrain :
            aux_image, recon_aux = self.crloss(cube, masks, cube_stage, enc2)
            return cube_stage,aux_image,recon_aux

        return cube_stage


class WholeDis(BaseNetwork):
    def __init__(self, in_channels, nker, norm="bnorm", relu=True, init_weights=True):
        super(WholeDis, self).__init__()
        self.relu = relu
        self.enc1 = layers.CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=None, relu=self.relu, bias=False)

        self.enc2 = layers.CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=norm, relu=self.relu,  bias=False)

        self.enc3 = layers.CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=norm, relu=self.relu,  bias=False)

        self.enc4 = layers.CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=norm, relu=self.relu,  bias=False)

        self.linear = nn.Linear(8 * nker * 16*16, 1)  # 256 = 16*16, 128 = 8*8

        if init_weights:
            self.init_weights()
            print("WholeDis init weights.")

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class SliceDis(BaseNetwork):
    def __init__(self, in_channels, out_channels, nker, norm="bnorm", relu=True, init_weights=True):
        super(SliceDis, self).__init__()
        self.relu = relu
        self.enc1 = layers.CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=None, relu=self.relu,  bias=False)

        self.enc2 = layers.CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=norm, relu=self.relu,  bias=False)

        self.enc3 = layers.CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=norm, relu=self.relu,  bias=False)

        self.enc4 = layers.CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                                 padding=1, norm=norm, relu=self.relu,  bias=False)

        self.enc5 = layers.CBR2d(8 * nker, out_channels, kernel_size=4, stride=2,
                                 padding=1, norm=norm, relu=self.relu,  bias=False)
        self.linear = nn.Linear(8 * nker * 16*16, 1)  # 256 = 16*16, 128 = 8*8

        if init_weights:
            self.init_weights()
            print("SliceDis init weights.")

    def forward(self, x):

        for k in range(6):
            z = x[:, 6*k:6*k+6, :, :]
            z = self.enc1(z)
            z = self.enc2(z)
            z = self.enc3(z)
            z = self.enc4(z)
            z = z.view(z.size()[0], -1)
            z = self.linear(z)

            if k == 0:
                z_ = z
            else:
                z_ = torch.cat((z_, z), dim=1)

        return z_


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3,
                                    padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3,
                                    padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class Crloss(BaseNetwork):
    def __init__(self):
        super(Crloss, self).__init__()
        self.cnum = 48
        cnum = self.cnum
        rate = 1
        # similarity encoder
        self.sconv1 = gen_conv(2 * cnum, 4 * cnum, 5, 1)  # skip cnn out
        self.sconv2 = gen_conv(2 * cnum, 4 * cnum, 3, 1, activation=nn.ReLU())  # skip cnn out

        # feature encoder
        self.bconv1 = gen_conv(18, cnum, 5, 1)  # skip cnn out
        self.bconv2_downsample = gen_conv(int(cnum / 2), 2 * cnum, 3, 2)
        self.bconv3 = gen_conv(cnum, 2 * cnum, 3, 1)  # skip cnn out
        self.bconv4_downsample = gen_conv(cnum, 4 * cnum, 3, 2)

        self.conv13_upsample_conv = gen_deconv(2 * cnum, 2 * cnum)
        self.conv14 = gen_conv(cnum * 2, 2 * cnum, 3, 1)  # skip cnn in
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum, cnum, 3, 1)  # skip cnn in

        self.conv16_2 = gen_conv(cnum // 2, cnum, 3, 1)  # skip cnn in
        ##cnum//2
        self.conv17 = gen_conv(cnum // 2, cnum, 3, 1, activation=None)
        self.conv18_upsample_conv = gen_deconv(cnum, 36)
        self.cam_1 = ReduceContextAttentionP1(nn_hard=False, ufstride=2 * rate,
                                              stride=2 * rate, pd=0,
                                              norm_type=1, is_th=False, th=0.5)
        self.cam_2 = ReduceContextAttentionP2(ufstride=4 * rate, bkg_patch_size=8 * rate,
                                              stride=4 * rate, pd=0, mk=False)

    def forward(self, x, mask,x_stage2,pm):
        # mask_obj: mask of avoid region
        b,f,c,hin,win = x.shape
        if not self.training:
            #print("network not forwarding refine")
            return  x_stage2, x_stage2, {}
        x_stage2 =  x_stage2.view(x_stage2.shape[0],-1,3,x_stage2.shape[2],x_stage2.shape[3])
        # similarify
        xnow = x_stage2*mask + x*(1-mask)
        xnow = xnow.view(xnow.shape[0],-1,xnow.shape[3],xnow.shape[4])

        xs = self.sconv1(pm)
        x_similar = self.sconv2(xs)
        #x_similar = pm

        bsize, _, h, w = pm.size()
        mask = mask.view(mask.shape[0],-1,mask.shape[3],mask.shape[4])
        mask_s = F.avg_pool2d(mask, kernel_size=4, stride=4) #mask_s  8 6 64 64
        similar = self.cam_1(x_similar, x_similar, mask_s)
        # feature
        xb = self.bconv1(xnow)
        x_skip1 = xb
        xb = self.bconv2_downsample(xb)
        xb = self.bconv3(xb)
        x_skip2 = xb
        xb = self.bconv4_downsample(xb)
        xb = self.conv13_upsample_conv(xb)
        xb = self.conv14(torch.cat((xb, x_skip2), 1))
        xb = self.conv15_upsample_conv(xb)
        xb = self.conv16(torch.cat((xb, x_skip1), 1))

        xb, recon_aux = self.cam_2(similar, xb, mask, {'raw':x})
        xb = self.conv16_2(xb)
        xb = self.conv17(xb)
        xb = self.conv18_upsample_conv(xb)
        xb = torch.tanh(xb)
        xb = xb.view(xb.shape[0],-1,3,xb.shape[-2],xb.shape[-1])
        return xb, recon_aux

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

