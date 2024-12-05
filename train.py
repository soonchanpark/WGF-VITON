#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from FTB_dataset import FTB_dataset, FTBDataLoader
from networks import VGGLoss, load_checkpoint, save_checkpoint, MultiscaleDiscriminator, WGFVITON,GANLoss
from networks import make_grid as mkgrid

from tensorboardX import SummaryWriter
from visualization import board_add_images, vis_densepose

from torch.autograd import Variable
from utils import create_network

import cv2
import numpy as np

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "WGFVITON")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--fine_width", type=int, default = 384)
    parser.add_argument("--fine_height", type=int, default = 512)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    
    #--DISCIRMINATOR    
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')        
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')    
    parser.add_argument('--gen_semantic_nc', type=int, default=64, help='# of input label classes without unknown class')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')    
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--n_layers_D', type=int, default=3, help='# layers in each discriminator')
    parser.add_argument('--num_D', type=int, default=1, help='number of discriminators to be used in multiscale')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')    
    parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for adam of discriminator')    

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument('--checkpointD', type=str, default='', help='model checkpoint for discriminator')    
    parser.add_argument("--display_count", type=int, default = 100)
    parser.add_argument("--save_count", type=int, default = 20000)
    parser.add_argument("--keep_step", type=int, default = 50000)
    parser.add_argument("--decay_step", type=int, default = 50000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def train_WGFVITON(opt, train_loader, Gmodel, Dmodel, board):
    gpus = [int(i) for i in opt.gpu_ids.split(',')]
    Gmodel = torch.nn.DataParallel(Gmodel, device_ids=gpus).cuda()
    Dmodel = torch.nn.DataParallel(Dmodel, device_ids=gpus).cuda()    

    Gmodel.train()
    Dmodel.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionFeat = nn.L1Loss()
    criterionGAN = GANLoss('hinge', tensor=torch.cuda.HalfTensor)    
        
    # optimizer
    optimizerG = torch.optim.Adam(Gmodel.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    optimizerD = torch.optim.Adam(Dmodel.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))    
    
    #<< [ SC : prepare GIC ] 
    nepoch = 0    
    ndata = train_loader.GetLength()
    print('***** DATA: ', ndata)
    criterionVGG = VGGLoss()
    
    lambda_l1, lambda_VGG, lambda_tv, lambda_iVGG, lambda_FM = 10.0, 1.0, 0.75, 8.0, 4.0
    lambda_wg = 60.0  
    lambda_wg2 = 0.2

    for step in range(opt.keep_step + opt.decay_step):
        nepoch = (step*opt.batch_size) / ndata

        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()

        im_pose = inputs['pose_image'].cuda()
        m_names = inputs['m_name']
        
        ##<< [ SC : LOADING FOR MODEL ] 
        agnostic = inputs['agnostic'].cuda() # [dp, pose, mod_seg] = [25, 17, 18]
        top_m_cloth =  inputs['top_m_cloth'].cuda()        
        bottom_m_cloth =  inputs['bottom_m_cloth'].cuda()
        top_m_seg = inputs['top_m_seg'].cuda()
        bottom_m_seg = inputs['bottom_m_seg'].cuda()        
        m_seg = inputs['model_seg'].cuda()
        
        ##<< [ SC : LOADING TOP ITEM ] 
        top_c_img = inputs['top_c_cloth'].cuda()
        top_c_seg = inputs['top_c_seg'].cuda()

        ##<< [ SC : LOADING BOTTOM ITEM ]         
        bottom_c_img = inputs['bottom_c_cloth'].cuda()
        bottom_c_seg = inputs['bottom_c_seg'].cuda()     
                
        ##<< [ SC : VISUALIZATION ] 
        im_g = inputs['grid_image'].cuda()
        im_occ = inputs['occ_parse'].cuda()
        im_in = inputs['mod_m_img']        

        input_model = agnostic
        input_top = torch.cat((top_c_img, top_c_seg),dim=1)
        input_bt = torch.cat((bottom_c_img, bottom_c_seg),dim=1)        
        flow_list, fake_image, warped_top, warped_top_m, warped_bt, warped_bt_m = Gmodel(input_top, input_bt, input_model)

        wg_mask = ((input_model[:,63:64,:,:])+1)/2.0 

        ##<< [ SC : TRAINING  DISCRIMINATOR ] 
        real_images = Variable(im)

        Dmodel.zero_grad()

        fake_concat = torch.cat((input_model,fake_image.detach()),1)
        real_concat = torch.cat((input_model,real_images),1)
        pred = Dmodel(torch.cat((fake_concat, real_concat),dim=0))        
        
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]        
        
        fake_loss = criterionGAN(pred_fake, False, for_discriminator=True)
        real_loss = criterionGAN(pred_real, True, for_discriminator=True)   
        GANloss_D = (fake_loss+real_loss)*0.5
        
        optimizerD.zero_grad()
        GANloss_D.backward()
        optimizerD.step()
        ##<< [ SC : TRAINING DISCRIMINATOR ] 
        
        ##<< [ SC : TRAINING GENERATOR ] 
        Gmodel.zero_grad()

        fake_concat = torch.cat((input_model,fake_image),1)
        real_concat = torch.cat((input_model,real_images),1)
        pred = Dmodel(torch.cat((fake_concat, real_concat),dim=0))
        
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]        

        GANloss_G = criterionGAN(pred_fake, True, for_discriminator=False)
        

        num_D = len(pred_fake)
        GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * lambda_FM / num_D
        loss_FM_full = GAN_Feat_loss          
        
        ##-- loss warping
        w_l1loss = (criterionL1(warped_top_m*im_occ, top_m_seg*im_occ) + criterionL1(warped_bt_m*im_occ, bottom_m_seg*im_occ)) * lambda_l1
        w_vggloss = (criterionVGG(warped_top*im_occ, top_m_cloth*im_occ) + criterionVGG(warped_bt*im_occ, bottom_m_cloth*im_occ))*lambda_VGG
        
        #221014_3
        w_wgloss_top = criterionL1(((warped_top_m[:,0:1,:,:]+1)/2.0)*(1-wg_mask),torch.zeros_like(1-wg_mask))*lambda_wg  
        w_wgloss_top2 = criterionL1(((warped_top_m[:,0:1,:,:]+1)/2.0)*(wg_mask),wg_mask)*lambda_wg2

        flow_top = flow_list[-1][0]
        top_y_tv = torch.abs(flow_top[:, 1:, :, :] - flow_top[:, :-1, :, :]).mean()
        top_x_tv = torch.abs(flow_top[:, :, 1:, :] - flow_top[:, :, :-1, :]).mean()
           
        flow_bt = flow_list[-1][1]
        bt_y_tv = torch.abs(flow_bt[:, 1:, :, :] - flow_bt[:, :-1, :, :]).mean()
        bt_x_tv = torch.abs(flow_bt[:, :, 1:, :] - flow_bt[:, :, :-1, :]).mean()           
        
        loss_tv = lambda_tv*(top_y_tv + top_x_tv+ bt_y_tv + bt_x_tv)
        
        N, _, iH, iW = top_c_img.size()
        for i in range(len(flow_list)-1):
            flow_top = flow_list[i][0]
            N, fH, fW, _ = flow_top.size()
            grid = mkgrid(N, iH, iW)
            flow_top = F.interpolate(flow_top.permute(0, 3, 1, 2), size = top_c_img.shape[2:], mode='bilinear').permute(0, 2, 3, 1)
            flow_top_norm = torch.cat([flow_top[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow_top[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
            warped_top_c = F.grid_sample(top_c_img, flow_top_norm + grid, padding_mode='border')
            warped_top_cm = F.grid_sample(top_c_seg, flow_top_norm + grid, padding_mode='border')
            w_l1loss += lambda_l1*criterionL1(warped_top_cm*im_occ, top_m_seg*im_occ) / (2 ** (4-i))
            w_vggloss += lambda_VGG*criterionVGG(warped_top_c*im_occ, top_m_cloth*im_occ) / (2 ** (4-i))        
            w_wgloss_top += lambda_wg*criterionL1(((warped_top_cm[:,0:1,:,:]+1)/2.0)*(1-wg_mask),torch.zeros_like(1-wg_mask)) / (2**(4-i))
            w_wgloss_top2 += lambda_wg2*criterionL1(((warped_top_cm[:,0:1,:,:]+1)/2.0)*(wg_mask),wg_mask) /(2**(4-i))
            
            flow_bt = flow_list[i][1]
            N, fH, fW, _ = flow_bt.size()
            grid = mkgrid(N, iH, iW)
            flow_bt = F.interpolate(flow_bt.permute(0, 3, 1, 2), size = bottom_c_img.shape[2:], mode='bilinear').permute(0, 2, 3, 1)
            flow_bt_norm = torch.cat([flow_bt[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow_bt[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
            warped_bt_c = F.grid_sample(bottom_c_img, flow_bt_norm + grid, padding_mode='border')
            warped_bt_cm = F.grid_sample(bottom_c_seg, flow_bt_norm + grid, padding_mode='border')
            w_l1loss += lambda_l1*criterionL1(warped_bt_cm*im_occ, bottom_m_seg*im_occ) / (2 ** (4-i))
            w_vggloss += lambda_VGG*criterionVGG(warped_bt_c*im_occ, bottom_m_cloth*im_occ) / (2 ** (4-i))         
        
        ##-- loss seg
        i_vggLoss = criterionVGG(fake_image,im)*lambda_iVGG
        
        w_l1loss = w_l1loss.sum()
        w_vggloss = w_vggloss.sum()
        i_vggLoss = i_vggLoss.sum()
        loss_tv = loss_tv.sum()
        GANloss_G = GANloss_G.sum()
        loss_FM_full = loss_FM_full.sum()
        w_wgloss_top = w_wgloss_top.sum()
        w_wgloss_top2 = w_wgloss_top2.sum()

        ## -- loss sum
        total_loss = w_l1loss + w_vggloss + i_vggLoss + loss_tv + GANloss_G + loss_FM_full + w_wgloss_top + w_wgloss_top2
        
        optimizerG.zero_grad()
        total_loss.backward()
        optimizerG.step()
        
        ##-- Step for lr scheduler
        schedulerG.step()
        schedulerD.step()
            
        if (step+1) % opt.display_count == 0:

            grid = mkgrid(N, iH, iW)        
            flow_top = F.interpolate(flow_list[-1][0].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)
            flow_top_norm = torch.cat([flow_top[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_top[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
            grid_warped_top = F.grid_sample(im_g, flow_top_norm + grid, padding_mode='border')
                        
            grid = mkgrid(N, iH, iW)
            flow_bt = F.interpolate(flow_list[-1][1].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)        
            flow_bt_norm = torch.cat([flow_bt[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_bt[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)        
            grid_warped_bt = F.grid_sample(im_g, flow_bt_norm + grid, padding_mode='border')                       
                  
            text_canvas_list = []       
            for b in range(agnostic.size(0)):
                tmp = np.zeros([opt.fine_height, opt.fine_width,3]).astype(np.uint8)
                tmp = cv2.putText(tmp,m_names[b], (20,20), 1, 1, (0,0,255), 2, cv2.LINE_AA)
                tmp = tmp.transpose(2,0,1)
                text_canvas_list.append(tmp)
                
            text_canvas_np = np.array(text_canvas_list)
            text_canvas_tensor = torch.from_numpy(text_canvas_np)            

            vis_dp = vis_densepose(agnostic[:,0:25,:,:])     
                      
            visuals = [[im_in, vis_dp,text_canvas_tensor], 
                    [top_c_img, warped_top*im_occ, top_m_cloth*im_occ],
                    [bottom_c_img, warped_bt*im_occ, bottom_m_cloth*im_occ],
                    [grid_warped_top, grid_warped_bt, (warped_top_m+warped_bt_m)*0.5],   
                    [agnostic[:,63:64,:,:], fake_image, im]]               

            t = time.time() - iter_start_time
            
            print('epoch: %4d, step: %8d, time:%.3f, loss: %4f, w_l1: %4f, w_VGG: %4f, wtv: %4f, i_VGG: %4f,  G_gan: %4f, FM: %4f, wgori: %4f, wginv: %4f, D_gan: %4f'% (nepoch, step+1, t, total_loss.item(), w_l1loss.item(), w_vggloss.item(),loss_tv.item(), i_vggLoss.item(),GANloss_G.item(), loss_FM_full.item(),(w_wgloss_top).item(), w_wgloss_top2.item(), GANloss_D.item()))            
            board_add_images(board, m_names, visuals, step+1)
            board.add_scalar('0_total loss', total_loss.item(), step+1)
            board.add_scalar('image_vgg loss', i_vggLoss.item(), step+1)
            board.add_scalar('wl1 loss', w_l1loss.item(), step+1)
            board.add_scalar('wagg loss', w_vggloss.item(), step+1)
            board.add_scalar('wtv loss', loss_tv.item(), step+1)
            board.add_scalar('FM loss', loss_FM_full.item(), step+1)            
            board.add_scalar('G_GAN loss', GANloss_G.item(), step+1)
            board.add_scalar('D_GAN loss', GANloss_D.item(), step+1)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(Gmodel, os.path.join(opt.checkpoint_dir, opt.name, 'step_G_%06d.pth' % (step+1)))
            save_checkpoint(Dmodel, os.path.join(opt.checkpoint_dir, opt.name, 'step_D_%06d.pth' % (step+1)))   

def main():
    opt = get_opt()
    print(opt)
    print("***** WGF-VITON TRAINING *****")    
    print("***** Start to train stage: WGF-VITON, checkpoint named: %s!" % (opt.name))
     
    # create dataset 
    train_dataset = FTB_dataset(opt)
    # create dataloader
    train_loader = FTBDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

    Gmodel = WGFVITON(opt, 6, opt.gen_semantic_nc,3 ,target_height=opt.fine_height)
    Gmodel.print_network()
        
    if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
        load_checkpoint(Gmodel, opt.checkpoint)        

    Dmodel1 = create_network(MultiscaleDiscriminator,opt)
    if not opt.checkpointD =='' and os.path.exists(opt.checkpointD):
        load_checkpoint(Dmodel1, opt.checkpointD)                            
            
    #221015: Use VITON-HR
    train_WGFVITON(opt, train_loader,Gmodel, Dmodel1,board)
        
    print('forward done')
    print('Finished training WGF-VITON, checkpoint named: %s!' % (opt.name))
    exit(0)        


if __name__ == "__main__":
    main()

