from tensorboardX import SummaryWriter
import torch
from PIL import Image
import os
import numpy as np
import cv2

color_code = [
    [150, 150, 150], #background, gray
    [255, 255, 204], #hat, cream
    [255, 255, 255], #hair, white
    [51, 51, 0], #face, light violet
    [0, 128, 128], #neck and body, dark cyon
    [128, 0, 0], #outer body, dark blue
    [0, 128, 0], # outer Rslv, dark green
    [0, 0, 128], # outer Lslv, dark red
    [255, 0, 0], # inner body, red
    [0, 255, 0], # inner Rslv, green
    [0, 0, 255], # inner Lslv, blue
    [255, 0, 255], # hand R, violet
    [255, 153, 204], # hand L, pink
    [80, 0, 0], # bottom hip, dark red
    [0, 80, 0], # bottom R slv, dark green
    [0, 0, 80], # bottom L slv, dark blue
    [102, 102, 153], # skirt, violet
    [0, 204, 255], # legR, dark red
    [255, 255, 0], # legL, yellow
    [204, 255, 255], # ShoeR, dark red
    [255, 204, 153], # ShoeL, dark red
    [0, 0, 0],
    [255, 255, 204],
    [0, 255, 255],
    [0, 128, 255],
    [0, 255, 128]
]

def vis_densepose(in_dp):
    plates = []
    #if pinfo:
    #    print(att)
    np_dp = in_dp.cpu().detach().numpy()
    for b in range(np_dp.shape[0]):
        plate = np.zeros([np_dp.shape[2], np_dp.shape[3],3]).astype(np.uint8)
        for c in range(0,np_dp.shape[1]):
            tmpmask =(np_dp[b,c,:,:]>0).astype(np.uint8) 
            tmpmask3 = np.stack((tmpmask,tmpmask,tmpmask),axis=2)
            #print(tmpmask3.shape)
            plate = tmpmask3*color_code[c] + (1-tmpmask3)*plate
        plates.append(plate)    

    norm_plate = np.zeros([in_dp.shape[0],3, in_dp.shape[2], in_dp.shape[3]]).astype(np.float32)
    for b in range(len(plates)):               
        norm_plate[b] = ((plates[b].transpose((2,0,1)).astype(np.float32))/255.0)*2.0 - 1.0    
    
    return torch.from_numpy(norm_plate)

def tensor_for_board(img_tensor):
    # map into [0,1]
    tensor = (img_tensor.clone()+1) * 0.5
    tensor.cpu().clamp(0,1)

    if tensor.size(1) == 1:
        tensor = tensor.repeat(1,3,1,1)

    return tensor

def tensor_list_for_board(img_tensors_list):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)
    
    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0][0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            canvas[:, :, offset_h : offset_h + height, offset_w : offset_w + width].copy_(tensor)

    return canvas

def board_add_image(board, tag_name, img_tensor, step_count):
    tensor = tensor_for_board(img_tensor)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)


def board_add_images(board, tag_name, img_tensors_list, step_count):
    tensor = tensor_list_for_board(img_tensors_list)

    for i, img in enumerate(tensor):
        board.add_image('%s/%s' % ('combine', tag_name[i]), img, step_count)

def save_vis_result(img_models, img_tops, img_bottoms, img_tensors, img_names, save_dir, filetype=0):
    for img_model, img_top, img_bt, img_tensor, img_name in zip(img_models, img_tops, img_bottoms, img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)
        
        mimg = (img_model.clone()+1)*0.5 * 255
        mimg = (mimg.cpu().clamp(0,255)).numpy().astype('uint8').transpose(1,2,0)

        timg = (img_top.clone()+1)*0.5 * 255
        timg = (timg.cpu().clamp(0,255)).numpy().astype('uint8').transpose(1,2,0)

        bimg = (img_bt.clone()+1)*0.5 * 255
        bimg = (bimg.cpu().clamp(0,255)).numpy().astype('uint8').transpose(1,2,0)
        
        timg = cv2.resize(timg,(timg.shape[1]//2, timg.shape[0]//2))
        bimg = cv2.resize(bimg,(bimg.shape[1]//2, bimg.shape[0]//2))        
        
        fimg = tensor.numpy().astype('uint8').transpose(1,2,0)
        
        clothes_img = cv2.vconcat([timg,bimg])
        #print('sizecomp', clothes_img.shape, mimg.shape)
        total_img = cv2.hconcat([mimg,clothes_img, fimg])
        #total_img = cv2.hconcat([clothes_img, fimg])
        
        total_img = cv2.cvtColor(total_img,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, img_name), total_img, [cv2.IMWRITE_JPEG_QUALITY,90])
        
        #cv2.imshow('tmp', total_img)
        #cv2.waitKey(0)


        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)


def save_images(img_tensors, img_names, save_dir, filetype=0):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
     
        Image.fromarray(array).save(os.path.join(save_dir, img_name),quality=100,subsampling=0)


def save_parses(img_tensors, c_names, m_names, save_dir, filetype=0):
    for img_tensor, c_name, m_name in zip(img_tensors, c_names, m_names):

        array = img_tensor.cpu().numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
     
        img_name = c_name+'__'+m_name
        img_name = img_name[:-3]+'png'
        Image.fromarray(array).save(os.path.join(save_dir, img_name),quality=100,subsampling=0)

