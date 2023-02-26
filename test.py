#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import os
import time
from FTB_dataset import FTB_dataset, FTBDataLoader
from networks import WGVITON, load_checkpoint
from networks import make_grid as mkgrid

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images, save_parses, vis_densepose
import cv2

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "WGVITON")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--front_or_back", default = 0)
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--stage", default = "1ST")
    parser.add_argument("--fine_width", type=int, default = 384)
    parser.add_argument("--fine_height", type=int, default = 512)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('--wearing', type=str, default='', help='wearing_file')    

    opt = parser.parse_args()
    return opt

def test_WGVITON(opt, test_loader, model, board):

    model.eval()
    base_name = os.path.basename(opt.name)
    save_dir = os.path.join(opt.result_dir, base_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)        

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im = inputs['image'].cuda()
        #im_raw = inputs['image_raw'].cuda()

        im_pose = inputs['pose_image'].cuda()
        m_names = inputs['m_name']
        
        ##<< [ SC : LOADING FOR MODEL ] 
        agnostic = inputs['agnostic'].cuda() # [dp, pose, mod_seg] = [25, 17, 18]
        top_m_cloth =  inputs['top_m_cloth'].cuda()        
        bottom_m_cloth =  inputs['bottom_m_cloth'].cuda()
        top_m_seg = inputs['top_m_seg'].cuda()
        bottom_m_seg = inputs['bottom_m_seg'].cuda()        
        
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
        
        #r_mask = inputs['reserve_mask'].cuda()
                  
        input_model = agnostic
        input_top = torch.cat((top_c_img, top_c_seg),dim=1)
        input_bt = torch.cat((bottom_c_img, bottom_c_seg),dim=1)        

        flow_list, fake_images, warped_top, warped_top_m, warped_bt, warped_bt_m = model(input_top, input_bt, input_model)

        N, _, iH, iW = top_c_img.size()
        
        grid = mkgrid(N, iH, iW)        
        flow_top = F.interpolate(flow_list[-1][0].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)
        flow_top_norm = torch.cat([flow_top[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_top[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
        grid_warped_top = F.grid_sample(im_g, flow_top_norm + grid, padding_mode='border')
                        
        grid = mkgrid(N, iH, iW)
        flow_bt = F.interpolate(flow_list[-1][1].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)        
        flow_bt_norm = torch.cat([flow_bt[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_bt[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)        
        grid_warped_bt = F.grid_sample(im_g, flow_bt_norm + grid, padding_mode='border')                       
        
        vis_dp = vis_densepose(agnostic[:,0:25,:,:])                         

        visuals = [[im, im_in, vis_dp], 
                    [top_c_img, warped_top, top_m_cloth],
                    [bottom_c_img, warped_bt, bottom_m_cloth],
                    [grid_warped_top, grid_warped_bt, grid_warped_bt],                    
                    [(warped_top+warped_bt)*0.5, agnostic[:,63:64,:,:], fake_images]]        
       
        save_images(fake_images, m_names, try_on_dir)
        
        if (step+1) % opt.display_count == 0:
            board_add_images(board, m_names, visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

def main():
    opt = get_opt()
    print(opt)
    opt.datamode = 'test'
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
    # create dataset 
    train_dataset = FTB_dataset(opt, is_train=False)

    # create dataloader
    train_loader = FTBDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    model = WGVITON(opt, 6, 64, 3, target_height=opt.fine_height)
    gpus = [int(i) for i in opt.gpu_ids.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    if not os.path.isfile(opt.checkpoint):
        print('FILE DOES NOT EXIST:', opt.checkpoint)
    load_checkpoint(model, opt.checkpoint)
    with torch.no_grad():
        test_WGVITON(opt, train_loader, model,board)           
      
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
