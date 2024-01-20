import pdb
import random

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

import model.loss
from model.PIUnet2 import *
from data.dataset import *
from pre_proc.create_data import *
from utils.util import *
from utils.cube_to_equi import c2e

import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from torch.utils.data import DataLoader
from model.loss import *


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_psnr(generated, gt):
    generated = (generated + 1) / 2 * 255
    h, w, c = gt.shape
    gt = (gt + 1) / 2 * 255
    mse = ((generated - gt) ** 2).sum(2).sum(1).sum(0)
    mse /= (c * h * w)
    psnr = 10 * torch.log10(255.0 * 255.0 / (mse + 1e-8))
    return psnr.item()


def discriminate(self, fake_image, real_image, mask):
    fake_concat = fake_image
    real_concat = real_image

    # In Batch Normalization, the fake and real images are
    # recommended to be in the same batch to avoid disparate
    # statistics in fake and real images.
    # So both fake and real images are fed to D all at once.
    fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
    if self.opt.d_mask_in:
        mask_all = torch.cat([mask, mask], dim=0)
    else:
        mask_all = None

    discriminator_out = self.netD(fake_and_real, mask_all)

    pred_fake, pred_real = self.divide_pred(discriminator_out)

    return pred_fake, pred_real


def g_image_loss(coarse_image, fake_image,  real_image, pred_fake):
    G_losses = {}
    no_fine_loss= False
    no_ganFeat_loss = True
    no_gan_loss = False
    no_vgg_loss= True
    no_l1_loss = False
    lambda_l1 = 1.0
    FloatTensor = torch.cuda.FloatTensor
    criterionGAN = GANLoss(gan_mode='hinge', tensor=FloatTensor)
    if not no_gan_loss and not no_fine_loss:

        G_losses['GAN'] = criterionGAN(pred_fake, True,
                                            for_discriminator=False)

    if not no_ganFeat_loss:
        raise NotImplementedError
        num_D = len(pred_fake)
        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                    pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        G_losses['GAN_Feat'] = GAN_Feat_loss

    if not no_vgg_loss and not no_fine_loss:
        G_losses['VGG'] = criterionVGG(fake_image, real_image) \
                          * self.opt.lambda_vgg
    if not no_l1_loss:
        if coarse_image is not None:
            G_losses['L1c'] = torch.nn.functional.l1_loss(coarse_image, real_image) * lambda_l1
        if not no_fine_loss:
            G_losses['L1f'] = torch.nn.functional.l1_loss(fake_image, real_image) * lambda_l1
    return G_losses


def train(args):
    # torch.manual_seed(3407)
    setup_seed(3407)
    mode = args.mode
    train_continue = args.train_continue
    data_parallel = args.data_parallel

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    nker = args.nker
    norm = args.norm

    network = args.network

    if torch.cuda.is_available() == False:
        raise Exception('At least one gpu must be available.')
    else:
        gpu = torch.device('cuda:0')

    print("mode: %s" % mode)
    print("data_parallel: %s" % data_parallel)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("network: %s" % network)
    print("norm: %s" % norm)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % gpu)


    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')
    result_dir_test = os.path.join(result_dir, 'test')
    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))
    if not os.path.exists(result_dir_val):
        os.makedirs(os.path.join(result_dir_val, 'png'))
    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))


    if mode == 'train':
        transform_train = transforms.Compose([Normalize(), ToTensor()])
        transform_val = transforms.Compose([Normalize(), ToTensor()])

        # dataset_train = PanoramaDataset(in_dir=os.path.join(
        #     data_dir, 'train'), transform=transform_train)
        dataset_train = PanoramaDataset(in_dir=data_dir, transform=transform_train)
        # dataset_val = PanoramaDataset(in_dir=os.path.join(
        #     data_dir, 'val'), transform=transform_val)
        dataset_val = PanoramaDataset(in_dir=os.path.join(
            data_dir, ), transform=transform_val)

        loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
        loader_val = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(loader_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

        num_data_val = len(loader_val)
        num_batch_val = np.ceil(num_data_val / batch_size)


    if network == "PIUnet":
        netFaceG = FaceGenerator(
            in_channels=4, out_channels=3, nker=nker, norm=norm, relu=True)
        netFaceD = FaceDis(in_channels=6, out_channels=1,
                           nker=nker, norm=norm, relu=True)
        netCubeG = CubeGenerator(
            in_channels=4, out_channels=3, nker=nker, norm=norm, activation='lrelu',relu=True)
        netWholeD = WholeDis(in_channels=6 * 6, nker=nker,
                             norm=norm, relu=True)
        netSliceD = SliceDis(in_channels=6, out_channels=1,
                             nker=nker, norm=norm, relu=True)

        if data_parallel:
            netFaceG = DataParallel(netFaceG)
            netFaceD = DataParallel(netFaceD)
            netCubeG = DataParallel(netCubeG)
            netWholeD = DataParallel(netWholeD)
            netSliceD = DataParallel(netSliceD)

        netFaceG = netFaceG.to(gpu)
        netFaceD = netFaceD.to(gpu)
        netCubeG = netCubeG.to(gpu)
        netWholeD = netWholeD.to(gpu)
        netSliceD = netSliceD.to(gpu)


    fn_l1 = nn.L1Loss().to(gpu)
    fn_gan = nn.BCELoss().to(gpu)


    optimFG = torch.optim.Adam(netFaceG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimCG = torch.optim.Adam(netCubeG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimFD = torch.optim.Adam(netFaceD.parameters(), lr=lr, betas=(0.5, 0.999))
    d_params = list(netWholeD.parameters()) + list(netSliceD.parameters())
    optimCD = torch.optim.Adam(d_params, lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to(
        'cpu').detach().numpy().transpose(0, 1, 3, 4, 2)
    fn_tonumpy_4 = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = None


    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    st_epoch = 0
    FG = sum(np.prod(list(p.size())) for p in netFaceG.parameters())
    CG = sum(np.prod(list(p.size())) for p in netCubeG.parameters())
    FD = sum(np.prod(list(p.size())) for p in netFaceD.parameters())
    WD = sum(np.prod(list(p.size())) for p in netWholeD.parameters())
    SD = sum(np.prod(list(p.size())) for p in netSliceD.parameters())

    print('Number of params in netFaceG: %d' % FG)
    print('Number of params in netFD: %d' % FD)
    print('Number of params in netCubeG: %d' % CG)
    print('Number of params in netWD: %d' % WD)
    print('Number of params in netSD: %d' % SD)
    print('Total Number of params in Network: %d' % (FG + FD + CG + WD + SD))

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netFaceG, netFaceD, optimFG, optimFD, st_epoch = face_load(ckpt_dir=ckpt_dir,
                                                                       netFaceG=netFaceG, netFaceD=netFaceD,
                                                                       optimFG=optimFG, optimFD=optimFD)
            netCubeG, netWholeD, netSliceD, optimCG, optimCD, st_epoch = cube_load(ckpt_dir=ckpt_dir,
                                                                                   netCubeG=netCubeG,
                                                                                   netWholeD=netWholeD,
                                                                                   netSliceD=netSliceD, optimCG=optimCG,
                                                                                   optimCD=optimCD)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netFaceG.train()
            netFaceD.train()
            netCubeG.train()
            netWholeD.train()
            netSliceD.train()

            loss_FG_train = []
            loss_CG_train = []

            loss_FD_train = []
            loss_CD_train = []
            D_Fake_aux_train = []
            D_real_aux_train = []
            GAN_aux = []
            L1f_aux = []
            for batch, sample in enumerate(loader_train, 1):
                # forward pass
                cube = sample['cube'].to(
                    gpu, dtype=torch.float32)  # B, F, 3, H, W
                cube_mask = sample['cube_mask'].to(
                    gpu, dtype=torch.float32)  # B, F, 1, H, W
                cube_mask_4 = cube_mask.view(
                    cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3], cube_mask.shape[4])

                # face order is ['f', 'r', 'b', 'l', 't', 'd']
                # we need 4 faces like this order ['f', 'r', 'b', 'l']
                # concate 4 faces
                # ground truth 4 faces -> B, 3, H, W*4
                g4f = torch.cat(
                    (cube[:, 0], torch.flip(cube[:, 1], [3]), torch.flip(cube[:, 2], [3]), cube[:, 3]), dim=3)
                # mask 4 faces -> B, 1, H, W*4
                m4f = torch.cat(
                    (cube_mask[:, 0], torch.flip(cube_mask[:, 1], [3]), torch.flip(cube_mask[:, 2], [3]),
                     cube_mask[:, 3]),
                    dim=3)
                # with mask 4 faces -> B, 3, H, W*4
                # mask 1是损坏 0是完好的像素
                cm4f = g4f - g4f * m4f

                # st1_output -> BN, C, H, W*4
                # st2_output -> BN, 6*C, H, W
                st1_output = netFaceG(g4f, m4f, cm4f)
                st1_output_inp = st1_output * m4f + g4f * (1 - m4f)

                # st1_output -> BN, 4, C, H, W
                st1_output_5 = st1_output.view(st1_output.shape[0], 4, st1_output.shape[1], st1_output.shape[2],
                                               int(st1_output.shape[3] / 4))

                st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3 = torch.split(
                    st1_output_5, 1, dim=1)

                st1_output_split1 = torch.flip(st1_output_split1[:], [2])
                st1_output_split2 = torch.flip(st1_output_split2[:], [2])
                st1_output_5 = torch.cat(
                    (st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3), dim=1)

                # st1_cube -> BN, 6, C, H, W
                # st1_cube_4 -> BN, 6*C, H, W
                st1_cube = torch.cat(
                    (st1_output_5, cube[:, 4:6] * (1 - cube_mask[:, 4:6])), dim=1)
                st1_cube_inp = st1_cube * cube_mask + cube * (1 - cube_mask)

                st1_cube_inp_4 = st1_cube_inp.view(
                    (st1_cube_inp.shape[0], -1, st1_cube_inp.shape[3], st1_cube_inp.shape[4]))

                cube_mask_4 = cube_mask.view(
                    cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3], cube_mask.shape[4])
                st2_input = torch.cat((st1_cube_inp_4, cube_mask_4), dim=1)

                st2_output, aux_image, recon_aux = netCubeG(st2_input, cube, cube_mask,isTrain = True)
                st2_output_5 = st2_output.view(
                    st2_output.shape[0], 6, 3, st2_output.shape[2], st2_output.shape[3])

                # cube + cube_mask
                # x_cube_mask_5 -> BN, 6, 3, H, W
                x_cube_mask_5 = cube - cube * cube_mask

                # inpainted_cube -> BN, 6, 3, H, W
                inpainted_cube = st2_output_5 * cube_mask + (1 - cube_mask) * cube
                composed_image_aux = aux_image * cube_mask + cube * (1 - cube_mask)

                # input cube + cube mask / (net output + cube) + cube mask
                # x_real_cube_mask -> BN, 6*6, H, W
                # x_fake_cube_mask -> BN, 6*6, H, W
                x_real_cube_mask = torch.cat((st1_cube_inp, cube), dim=2)
                x_fake_cube_mask = torch.cat((st1_cube_inp, st2_output_5), dim=2)
                x_fake_cube_mask_aux = torch.cat((st1_cube_inp, composed_image_aux), dim=2)
                x_real_cube_mask = x_real_cube_mask.view(
                    x_real_cube_mask.shape[0], -1, x_real_cube_mask.shape[3], x_real_cube_mask.shape[4])
                x_fake_cube_mask = x_fake_cube_mask.view(
                    x_fake_cube_mask.shape[0], -1, x_fake_cube_mask.shape[3], x_fake_cube_mask.shape[4])
                x_fake_cube_mask_aux = x_fake_cube_mask_aux.view(x_fake_cube_mask_aux.shape[0], -1,
                                                                 x_fake_cube_mask_aux.shape[-2],
                                                                 x_fake_cube_mask_aux.shape[-1])
                # backward netWholeD, netSliceD
                set_requires_grad(netWholeD, True)
                set_requires_grad(netSliceD, True)
                optimCD.zero_grad()

                pred_whole_real = netWholeD(x_real_cube_mask)
                pred_whole_fake = netWholeD(x_fake_cube_mask.detach())

                pred_slice_real = netSliceD(x_real_cube_mask)
                pred_slice_fake = netSliceD(x_fake_cube_mask.detach())
                pred_slice_fake_aux = netSliceD(x_fake_cube_mask_aux.detach())

                FloatTensor = torch.cuda.FloatTensor
                criterionGAN = GANLoss(gan_mode='hinge', tensor=FloatTensor)

                D_Fake_aux = criterionGAN(pred_slice_fake_aux, False,
                                          for_discriminator=True)
                D_real_aux = criterionGAN(pred_slice_real, True,
                                          for_discriminator=True)

                whole_penalty = calc_gradient_penalty(netWholeD, x_real_cube_mask, x_fake_cube_mask.detach(), gpu)
                slice_penalty = calc_gradient_penalty(netSliceD, x_real_cube_mask, x_fake_cube_mask.detach(), gpu)
                loss_wgan_gp = whole_penalty + slice_penalty
                loss_wgan_d = torch.mean(pred_whole_fake - pred_whole_real) + torch.mean(
                    pred_slice_fake - pred_slice_real)
                loss_CD = loss_wgan_gp * 10 + loss_wgan_d

                loss_CD.backward(retain_graph=True)
                D_real_aux.backward(retain_graph=True)
                D_Fake_aux.backward(retain_graph=True)

                # backward netCubeG
                set_requires_grad(netWholeD, False)
                set_requires_grad(netSliceD, False)
                optimCG.zero_grad()

                pred_whole_fake = netWholeD(x_fake_cube_mask)
                pred_slice_fake = netSliceD(x_fake_cube_mask)

                loss_l1 = fn_l1(st2_output_5 * cube_mask, cube * cube_mask)
                loss_ae = fn_l1(st2_output_5 * (1 - cube_mask), cube * (1 - cube_mask))
                loss_G_wgan = - torch.mean(pred_whole_fake) - torch.mean(pred_slice_fake)
                loss_CG = loss_G_wgan * 0.001 + loss_l1 * 10 + loss_ae

                G_losses_aux = g_image_loss(None, aux_image,  cube, pred_slice_fake_aux)
                G_losses = { }
                for k, v in G_losses_aux.items():
                    G_losses[k + "_aux"] = v * 0.5

                loss_CG.backward(retain_graph=True)
                G_losses['GAN_aux'].backward(retain_graph = True)
                G_losses['L1f_aux'].backward(retain_graph=True)


                # backward netFaceD
                set_requires_grad(netFaceD, True)
                optimFD.zero_grad()

                # input face + face gt / net output + face gt
                # x_real_face_mask -> BN, 6, H, W*4
                # x_fake_face_mask -> BN, 6, H, W*4
                x_real_face_mask = torch.cat((cm4f, g4f), dim=1)
                x_fake_face_mask = torch.cat((cm4f, st1_output), dim=1)

                pred_face_real = netFaceD(x_real_face_mask)
                pred_face_fake = netFaceD(x_fake_face_mask.detach())
                dis_real_loss = fn_gan(
                    pred_face_real, torch.ones_like(pred_face_real))
                dis_fake_loss = fn_gan(
                    pred_face_fake, torch.zeros_like(pred_face_fake))
                loss_FD = (dis_real_loss + dis_fake_loss) / 2 * 100

                loss_FD.backward(retain_graph=True)

                # backward netFaceG
                set_requires_grad(netFaceD, False)
                optimFG.zero_grad()

                pred_face_fake = netFaceD(x_fake_face_mask)

                loss_l1 = fn_l1(st1_output * m4f, g4f * m4f)
                loss_ae = fn_l1(st1_output * (1 - m4f), g4f * (1 - m4f))
                loss_FG_gan = fn_gan(pred_face_fake, torch.ones_like(pred_face_fake))
                loss_FG = loss_FG_gan * 0.001 + loss_l1 * 10 + loss_ae

                loss_FG.backward()

                optimCD.step()
                optimCG.step()
                optimFD.step()
                optimFG.step()

                # lr_scheduler_CD.step()
                # lr_scheduler_CG.step()
                # lr_scheduler_FD.step()
                # lr_scheduler_FG.step()

                # 손실함수 계산
                loss_FG_train += [loss_FG.item()]
                loss_CG_train += [loss_CG.item()]
                loss_FD_train += [loss_FD.item()]
                loss_CD_train += [loss_CD.item()]
                D_Fake_aux_train += [ D_Fake_aux.item()]
                D_real_aux_train += [ D_real_aux.item()]
                GAN_aux += [ G_losses['GAN_aux'].item()]
                L1f_aux += [G_losses['L1f_aux'].item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "FG %.4f | CG %.4f | FD %.4f | CD %.4f | GAN_aux %.4f | L1f_aux %.4f" %
                      (epoch, num_epoch, batch, num_batch_train * batch_size,
                       np.mean(loss_FG_train), np.mean(loss_CG_train),
                       np.mean(loss_FD_train), np.mean(loss_CD_train),
                       G_losses['GAN_aux'].item(), G_losses['L1f_aux'].item()))

                if batch % 1 == 0:
                    # Tensorboard
                    id = num_batch_train * (epoch - 1) + batch

                    # 4 face ori
                    g4f = fn_tonumpy_4(fn_denorm(g4f, mean=0.5, std=0.5))
                    # 4 face with mask
                    mask4f = fn_tonumpy_4(fn_denorm(cm4f, mean=0.5, std=0.5))
                    # 4 face inpaint result
                    result4f = fn_tonumpy_4(
                        fn_denorm(st1_output_inp, mean=0.5, std=0.5))

                    inpainted_cube = fn_tonumpy(
                        fn_denorm(inpainted_cube, mean=0.5, std=0.5))
                    cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))
                    x_cube_mask_5 = fn_tonumpy(
                        fn_denorm(x_cube_mask_5, mean=0.5, std=0.5))

                    equirec_ori = c2e(
                        cube[0], h=512, w=1024, cube_format='list')
                    equirec_ori_mask = c2e(
                        x_cube_mask_5[0], h=512, w=1024, cube_format='list')
                    equirec = c2e(
                        inpainted_cube[0], h=512, w=1024, cube_format='list')

                    # plt.imsave(os.path.join(result_dir_train, 'png', '%07d_4face_mask.png' % (id)),
                    #            mask4f[0], cmap=cmap)
                    # plt.imsave(os.path.join(result_dir_train, 'png', '%07d_4face.png' % (id)),
                    #            result4f[0], cmap=cmap)
                    # plt.imsave(os.path.join(result_dir_train, 'png', '%07d_4face_ori.png' % (id)),
                    #            g4f[0], cmap=cmap)
                    #plt.imsave(os.path.join(result_dir_train, 'png', '%07d_pano_mask.png' % (id)),
                    #           equirec_ori_mask, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%07d_pano.png' % (id)),
                               equirec, cmap=cmap)
                    #plt.imsave(os.path.join(result_dir_train, 'png', '%07d_pano_ori.png' % (id)),
                     #          equirec_ori, cmap=cmap)

                    # writer_train.add_image(
                    #     '4face_mask.png', mask4f[0], id, dataformats='HWC')
                    # writer_train.add_image(
                    #     '4face.png', result4f[0], id, dataformats='HWC')
                    writer_train.add_image(
                        'pano_ori.png', equirec_ori, id, dataformats='HWC')
                    writer_train.add_image(
                        'pano_mask.png', equirec_ori_mask, id, dataformats='HWC')
                    writer_train.add_image(
                        'pano.png', equirec, id, dataformats='HWC')

            writer_train.add_scalar('loss_FG', np.mean(loss_FG_train), epoch)
            writer_train.add_scalar('loss_CG', np.mean(loss_CG_train), epoch)
            writer_train.add_scalar('loss_FD', np.mean(loss_FD_train), epoch)
            writer_train.add_scalar('loss_CD', np.mean(loss_CD_train), epoch)

            with torch.no_grad():
                netFaceG.eval()
                netFaceD.eval()
                netCubeG.eval()
                netWholeD.eval()
                netSliceD.eval()

                loss_FG_val = []
                loss_CG_val = []

                psnr_total = 0

                if epoch % 100 ==0:
                    for batch, sample in enumerate(loader_val, 1):
                        # forward pass
                        cube = sample['cube'].to(gpu, dtype=torch.float32)  # B, F, 3, H, W
                        cube_mask = sample['cube_mask'].to(gpu, dtype=torch.float32)  # B, F, 1, H, W
                        cube_mask_4 = cube_mask.view(cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3],
                                                     cube_mask.shape[4])

                        # face order is ['f', 'r', 'b', 'l', 't', 'd']
                        # we need 4 faces like this order ['f', 'r', 'b', 'l']
                        # concate 4 faces
                        # ground truth 4 faces -> B, 3, H, W*4
                        g4f = torch.cat(
                            (cube[:, 0], torch.flip(cube[:, 1], [3]), torch.flip(cube[:, 2], [3]), cube[:, 3]), dim=3)
                        # mask 4 faces -> B, 1, H, W*4
                        m4f = torch.cat(
                            (cube_mask[:, 0], torch.flip(cube_mask[:, 1], [3]), torch.flip(cube_mask[:, 2], [3]),
                             cube_mask[:, 3]),
                            dim=3)
                        # with mask 4 faces -> B, 3, H, W*4
                        cm4f = g4f - g4f * m4f

                        # st1_output -> BN, C, H, W*4
                        # st2_output -> BN, 6*C, H, W
                        st1_output = netFaceG(g4f, m4f, cm4f)
                        st1_output_inp = st1_output * m4f + g4f * (1 - m4f)

                        # st1_output -> BN, 4, C, H, W
                        st1_output_5 = st1_output.view(st1_output.shape[0], 4, st1_output.shape[1], st1_output.shape[2],
                                                       int(st1_output.shape[3] / 4))
                        # f:0 r:1 b:2 l:3
                        st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3 = torch.split(
                            st1_output_5, 1, dim=1)

                        st1_output_split1 = torch.flip(st1_output_split1[:], [2])
                        st1_output_split2 = torch.flip(st1_output_split2[:], [2])
                        st1_output_5 = torch.cat(
                            (st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3), dim=1)

                        # st1_cube -> BN, 6, C, H, W
                        # st1_cube_4 -> BN, 6*C, H, W
                        # 6面 cat 前面在 FaceG中修复好的四面 cat 后面的两面
                        st1_cube = torch.cat(
                            (st1_output_5, cube[:, 4:6] * (1 - cube_mask[:, 4:6])), dim=1)
                        st1_cube_inp = st1_cube * cube_mask + cube * (1 - cube_mask)

                        st1_cube_inp_4 = st1_cube_inp.view(
                            (st1_cube_inp.shape[0], -1, st1_cube_inp.shape[3], st1_cube_inp.shape[4]))

                        cube_mask_4 = cube_mask.view(
                            cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3], cube_mask.shape[4])
                        st2_input = torch.cat((st1_cube_inp_4, cube_mask_4), dim=1)

                        st2_output = netCubeG(st2_input, cube, cube_mask, isTrain=False)
                        st2_output_5 = st2_output.view(
                            st2_output.shape[0], 6, 3, st2_output.shape[2], st2_output.shape[3])

                        # cube + cube_mask
                        # x_cube_mask_5 -> BN, 6, 3, H, W
                        x_cube_mask_5 = cube - cube * cube_mask

                        # inpainted_cube -> BN, 6, 3, H, W
                        inpainted_cube = st2_output_5 * cube_mask + (1 - cube_mask) * cube

                        # input cube + cube mask / (net output + cube) + cube mask
                        # x_real_cube_mask -> BN, 6*6, H, W
                        # x_fake_cube_mask -> BN, 6*6, H, W
                        x_fake_cube_mask = torch.cat((st1_cube_inp, st2_output_5), dim=2)
                        x_fake_cube_mask = x_fake_cube_mask.view(
                            x_fake_cube_mask.shape[0], -1, x_fake_cube_mask.shape[3], x_fake_cube_mask.shape[4])

                        # backward netCubeG
                        pred_whole_fake = netWholeD(x_fake_cube_mask)
                        pred_slice_fake = netSliceD(x_fake_cube_mask)

                        loss_l1 = fn_l1(st2_output_5 * cube_mask, cube * cube_mask)
                        loss_ae = fn_l1(st2_output_5 * (1 - cube_mask), cube * (1 - cube_mask))
                        loss_G_wgan = - torch.mean(pred_whole_fake) - torch.mean(pred_slice_fake)
                        loss_CG = loss_G_wgan * 0.001 + loss_l1 * 10 + loss_ae

                        # backward netFaceG
                        x_fake_face_mask = torch.cat((cm4f, st1_output), dim=1)
                        pred_face_fake = netFaceD(x_fake_face_mask)

                        loss_l1 = fn_l1(st1_output * m4f, g4f * m4f)
                        loss_ae = fn_l1(st1_output * (1 - m4f), g4f * (1 - m4f))
                        loss_FG_gan = fn_gan(pred_face_fake, torch.ones_like(pred_face_fake))
                        loss_FG = loss_FG_gan * 0.001 + loss_l1 * 10 + loss_ae


                        loss_FG_val += [loss_FG.item()]
                        loss_CG_val += [loss_CG.item()]

                        print("VAL: EPOCH %04d / %04d | BATCH %04d / %04d | "
                              "FG %.4f | CG %.4f" %
                              (epoch, num_epoch, batch, num_batch_val * batch_size,
                               np.mean(loss_FG_val), np.mean(loss_CG_val)))

                        # 4 face with mask
                        mask4f = fn_tonumpy_4(fn_denorm(cm4f, mean=0.5, std=0.5))
                        # 4 face inpaint result
                        result4f = fn_tonumpy_4(fn_denorm(st1_output_inp, mean=0.5, std=0.5))

                        inpainted_cube = fn_tonumpy(fn_denorm(inpainted_cube, mean=0.5, std=0.5))
                        cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))
                        x_cube_mask_5 = fn_tonumpy(fn_denorm(x_cube_mask_5, mean=0.5, std=0.5))

                        equirec_ori = c2e(cube[0], h=512, w=1024, cube_format='list')
                        equirec = c2e(inpainted_cube[0], h=512, w=1024, cube_format='list')
                        equirec_tensor = torch.from_numpy(equirec)
                        equirec_ori_tensor = torch.from_numpy(equirec_ori)

                        if batch % 40 == 0:
                            # Tensorboard
                            id = num_batch_val * (epoch - 1) + batch

                            # 4 face with mask
                            mask4f = fn_tonumpy_4(fn_denorm(cm4f, mean=0.5, std=0.5))
                            # 4 face inpaint result
                            result4f = fn_tonumpy_4(fn_denorm(st1_output_inp, mean=0.5, std=0.5))

                            # inpainted_cube = fn_tonumpy(fn_denorm(inpainted_cube, mean=0.5, std=0.5))
                            # cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))
                            # x_cube_mask_5 = fn_tonumpy(fn_denorm(x_cube_mask_5, mean=0.5, std=0.5))

                            # equirec_ori = c2e(cube[0], h=512, w=1024, cube_format='list')
                            # equirec_ori_mask = c2e(x_cube_mask_5[0], h=512, w=1024, cube_format='list')
                            # equirec = c2e(inpainted_cube[0], h=512, w=1024, cube_format='list')

                            # plt.imsave(os.path.join(result_dir_val, 'png', '%07d_4face_mask.png' % (id)),
                            #            mask4f[0], cmap=cmap)
                            # plt.imsave(os.path.join(result_dir_val, 'png', '%07d_4face.png' % (id)),
                            #            result4f[0], cmap=cmap)
                            plt.imsave(os.path.join(result_dir_val, 'png', '%07d_pano_mask.png' % (id)),
                                       equirec_ori_mask, cmap=cmap)
                            plt.imsave(os.path.join(result_dir_val, 'png', '%07d_pano.png' % (id)),
                                       equirec, cmap=cmap)
                            plt.imsave(os.path.join(result_dir_val, 'png', '%07d_pano_ori.png' % (id)),
                                       equirec_ori, cmap=cmap)

                            # writer_val.add_image('4face_mask.png', mask4f[0], id, dataformats='HWC')
                            # writer_val.add_image('4face.png', result4f[0], id, dataformats='HWC')
                            writer_val.add_image('pano_ori.png', equirec_ori, id, dataformats='HWC')
                            writer_val.add_image('pano_mask.png', equirec_ori_mask, id, dataformats='HWC')
                            writer_val.add_image('pano.png', equirec, id, dataformats='HWC')

                        psnr = get_psnr(equirec_tensor, equirec_ori_tensor)
                        psnr_total += psnr
                    psnr_total /= (num_batch_val * batch_size)
                    f = open("./0.1psnr_test.txt", "a")
                    a = 'epoch' + str(epoch) + ':' + str(psnr_total) + '\n'
                    f.writelines(a)
                    f.flush()
                    f.close  # 关闭文件
                    writer_val.add_scalar('loss_FG', np.mean(loss_FG_val), epoch)
                    writer_val.add_scalar('loss_CG', np.mean(loss_CG_val), epoch)

                if epoch % 100 == 0:
                  face_save(ckpt_dir=ckpt_dir, netFaceG=netFaceG, netFaceD=netFaceD, optimFG=optimFG, optimFD=optimFD,
                            epoch=epoch)
                  cube_save(ckpt_dir=ckpt_dir, netCubeG=netCubeG, netWholeD=netWholeD,
                            netSliceD=netSliceD, optimCG=optimCG, optimCD=optimCD, epoch=epoch)

        writer_val.close()
        writer_train.close()
        print("training was finished")


def test(args):
    setup_seed(3407)
    ## test
    mode = args.mode
    data_parallel = args.data_parallel

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    nker = args.nker

    norm = args.norm

    network = args.network
    result_dir_test = os.path.join(result_dir, 'test')

    if torch.cuda.is_available() == False:
        raise Exception('At least one gpu must be available.')
    else:
        gpu = torch.device('cuda:0')

    print("mode: %s" % mode)
    print("data_parallel: %s" % data_parallel)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("network: %s" % network)

    print("norm: %s" % norm)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % gpu)


    result_dir_ori = os.path.join(result_dir, 'ori')
    result_dir_gen = os.path.join(result_dir, 'gen')

    if not os.path.exists(result_dir_ori):
        os.makedirs(result_dir_ori)
    if not os.path.exists(result_dir_gen):
        os.makedirs(result_dir_gen)


    if mode == "test":
        transform_test = transforms.Compose([Normalize(), ToTensor()])

        dataset_test = PanoramaDataset(in_dir=os.path.join(data_dir, ), transform=transform_test)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)


        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / 1)


    if network == "PIUnet":
        netFaceG = FaceGenerator(
            in_channels=4, out_channels=3, nker=nker, norm=norm, relu=True)
        netFaceD = FaceDis(in_channels=6, out_channels=1, nker=nker, norm=norm, relu=True)
        netCubeG = CubeGenerator(
            in_channels=4, out_channels=3, nker=nker, norm=norm, relu=True)
        netWholeD = WholeDis(in_channels=6 * 6, nker=nker, norm=norm, relu=True)
        netSliceD = SliceDis(in_channels=6, out_channels=1, nker=nker, norm=norm, relu=True)

        if data_parallel:
            netFaceG = DataParallel(netFaceG)
            netFaceD = DataParallel(netFaceD)
            netCubeG = DataParallel(netCubeG)
            netWholeD = DataParallel(netWholeD)
            netSliceD = DataParallel(netSliceD)

        netFaceG = netFaceG.to(gpu)
        netFaceD = netFaceD.to(gpu)
        netCubeG = netCubeG.to(gpu)
        netWholeD = netWholeD.to(gpu)
        netSliceD = netSliceD.to(gpu)


    fn_l1 = nn.L1Loss().to(gpu)
    fn_gan = nn.BCELoss().to(gpu)


    optimFG = torch.optim.Adam(netFaceG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimCG = torch.optim.Adam(netCubeG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimFD = torch.optim.Adam(netFaceD.parameters(), lr=lr, betas=(0.5, 0.999))
    d_params = list(netWholeD.parameters()) + list(netSliceD.parameters())
    optimCD = torch.optim.Adam(d_params, lr=lr, betas=(0.5, 0.999))


    fn_tonumpy = lambda x: x.to(
        'cpu').detach().numpy().transpose(0, 1, 3, 4, 2)
    fn_tonumpy_4 = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = None
    id = 0
    # TEST MODE
    if mode == "test":
        netFaceG, netFaceD, optimFG, optimFD, st_epoch = face_load(ckpt_dir=ckpt_dir, netFaceG=netFaceG,
                                                                   netFaceD=netFaceD, optimFG=optimFG, optimFD=optimFD)
        netCubeG, netWholeD, netSliceD, optimCG, optimCD, st_epoch = cube_load(ckpt_dir=ckpt_dir,
                                                                               netCubeG=netCubeG, netWholeD=netWholeD,
                                                                               netSliceD=netSliceD, optimCG=optimCG,
                                                                               optimCD=optimCD)

        with torch.no_grad():
            netFaceG.eval()
            netFaceD.eval()
            netCubeG.eval()
            netWholeD.eval()
            netSliceD.eval()

            loss_FG_test = []
            loss_CG_test = []

            for batch, sample in enumerate(loader_test, 1):
                # forward pass
                cube = sample['cube'].to(gpu, dtype=torch.float32)  # B, F, 3, H, W
                cube_mask = sample['cube_mask'].to(gpu, dtype=torch.float32)  # B, F, 1, H, W
                cube_mask_4 = cube_mask.view(cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3],
                                             cube_mask.shape[4])

                # face order is ['f', 'r', 'b', 'l', 't', 'd']
                # we need 4 faces like this order ['f', 'r', 'b', 'l']
                # concate 4 faces
                # ground truth 4 faces -> B, 3, H, W*4
                g4f = torch.cat(
                    (cube[:, 0], torch.flip(cube[:, 1], [3]), torch.flip(cube[:, 2], [3]), cube[:, 3]), dim=3)
                # mask 4 faces -> B, 1, H, W*4
                m4f = torch.cat(
                    (cube_mask[:, 0], torch.flip(cube_mask[:, 1], [3]), torch.flip(cube_mask[:, 2], [3]),
                     cube_mask[:, 3]),
                    dim=3)
                # with mask 4 faces -> B, 3, H, W*4
                cm4f = g4f - g4f * m4f

                # st1_output -> BN, C, H, W*4
                # st2_output -> BN, 6*C, H, W
                st1_output = netFaceG(g4f, m4f, cm4f)
                st1_output_inp = st1_output * m4f + g4f * (1 - m4f)

                # st1_output -> BN, 4, C, H, W
                st1_output_5 = st1_output.view(st1_output.shape[0], 4, st1_output.shape[1], st1_output.shape[2],
                                               int(st1_output.shape[3] / 4))

                st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3 = torch.split(
                    st1_output_5, 1, dim=1)

                st1_output_split1 = torch.flip(st1_output_split1[:], [2])
                st1_output_split2 = torch.flip(st1_output_split2[:], [2])
                st1_output_5 = torch.cat(
                    (st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3), dim=1)

                # st1_cube -> BN, 6, C, H, W
                # st1_cube_4 -> BN, 6*C, H, W
                st1_cube = torch.cat(
                    (st1_output_5, cube[:, 4:6] * (1 - cube_mask[:, 4:6])), dim=1)
                st1_cube_inp = st1_cube * cube_mask + cube * (1 - cube_mask)

                st1_cube_inp_4 = st1_cube_inp.view(
                    (st1_cube_inp.shape[0], -1, st1_cube_inp.shape[3], st1_cube_inp.shape[4]))

                cube_mask_4 = cube_mask.view(
                    cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3], cube_mask.shape[4])
                st2_input = torch.cat((st1_cube_inp_4, cube_mask_4), dim=1)

                st2_output = netCubeG(st2_input,cube,cube_mask)
                st2_output_5 = st2_output.view(
                    st2_output.shape[0], 6, 3, st2_output.shape[2], st2_output.shape[3])

                # cube + cube_mask
                # x_cube_mask_5 -> BN, 6, 3, H, W
                x_cube_mask_5 = cube - cube * cube_mask

                # inpainted_cube -> BN, 6, 3, H, W
                inpainted_cube = st2_output_5 * cube_mask + (1 - cube_mask) * cube

                # input cube + cube mask / (net output + cube) + cube mask
                # x_real_cube_mask -> BN, 6*6, H, W
                # x_fake_cube_mask -> BN, 6*6, H, W
                x_fake_cube_mask = torch.cat((st1_cube_inp, st2_output_5), dim=2)
                x_fake_cube_mask = x_fake_cube_mask.view(
                    x_fake_cube_mask.shape[0], -1, x_fake_cube_mask.shape[3], x_fake_cube_mask.shape[4])

                # backward netCubeG
                pred_whole_fake = netWholeD(x_fake_cube_mask)
                pred_slice_fake = netSliceD(x_fake_cube_mask)

                loss_l1 = fn_l1(st2_output_5 * cube_mask, cube * cube_mask)
                loss_ae = fn_l1(st2_output_5 * (1 - cube_mask), cube * (1 - cube_mask))
                loss_G_wgan = - torch.mean(pred_whole_fake) - torch.mean(pred_slice_fake)
                loss_CG = loss_G_wgan * 0.001 + loss_l1 * 10 + loss_ae

                # backward netFaceG
                x_fake_face_mask = torch.cat((cm4f, st1_output), dim=1)
                pred_face_fake = netFaceD(x_fake_face_mask)

                loss_l1 = fn_l1(st1_output * m4f, g4f * m4f)
                loss_ae = fn_l1(st1_output * (1 - m4f), g4f * (1 - m4f))
                loss_FG_gan = fn_gan(pred_face_fake, torch.ones_like(pred_face_fake))
                loss_FG = loss_FG_gan * 0.001 + loss_l1 * 10 + loss_ae


                loss_FG_test += [loss_FG.item()]
                loss_CG_test += [loss_CG.item()]

                print("TEST: BATCH %04d / %04d | FG %.4f | CG %.4f" %
                      (batch, num_batch_test * batch_size, np.mean(loss_FG_test), np.mean(loss_CG_test)))

                # 4 face ori
                g4f = fn_tonumpy_4(fn_denorm(g4f, mean=0.5, std=0.5))
                # 4 face with mask
                mask4f = fn_tonumpy_4(fn_denorm(cm4f, mean=0.5, std=0.5))
                # 4 face inpaint result
                result4f = fn_tonumpy_4(
                    fn_denorm(st1_output_inp, mean=0.5, std=0.5))

                inpainted_cube = fn_tonumpy(
                    fn_denorm(inpainted_cube, mean=0.5, std=0.5))
                cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))
                x_cube_mask_5 = fn_tonumpy(
                    fn_denorm(x_cube_mask_5, mean=0.5, std=0.5))

                equirec_ori = c2e(
                    cube[0], h=512, w=1024, cube_format='list')
                equirec_ori_mask = c2e(
                    x_cube_mask_5[0], h=512, w=1024, cube_format='list')
                equirec = c2e(
                    inpainted_cube[0], h=512, w=1024, cube_format='list')

                plt.imsave(os.path.join(result_dir_test, 'png', '%07d_pano_mask.png' % (id)),
                           equirec_ori_mask, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%07d_pano.png' % (id)),
                           equirec, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%07d_pano_ori.png' % (id)),
                           equirec_ori, cmap=cmap)

                id += 1


def calc_gradient_penalty(netD, real_data, fake_data, gpu):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)

    alpha = alpha.to(gpu)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_().clone()

    disc_interpolates = netD(interpolates)
    grad_outputs = torch.ones(disc_interpolates.size())

    grad_outputs = grad_outputs.to(gpu)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
