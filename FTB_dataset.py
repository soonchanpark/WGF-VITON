#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
import copy
import random

import cv2

class FTB_dataset(data.Dataset):
    """Dataset for 2021 AI DATA(65)."""

    def __init__(self, opt, is_train=True):
        super(FTB_dataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        #self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transformG = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])

        #self.front_or_back = opt.front_or_back
        self.ann_list = []
        self.p_aug = 0.5
        self.is_train = is_train
        self.pose_gt_pair = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
        self.pose_op_pair = [[15,16],[17,18],[2,5],[3,6],[4,7],[9,12],[10,13],[11,14], [21, 24],[19,22]]
        self.target_kpts_idx = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

        self.is_unpaired = False
        
        print("***** ETRIDataset Initialized. train=",self.is_train, "FTB_dataset.py")

        if self.is_train == False:
            if 'unpaired' in opt.wearing:
                print('unpaired testset')
                self.is_unpaired = True

            with open(osp.join(opt.dataroot, opt.wearing), 'r') as f:        
                self.ann_list = json.load(f)
        else:
            #print(osp.join(opt.dataroot, 'wearing_info_sf_'+self.datamode+'.json'))
            with open(osp.join(opt.dataroot, 'train_pairs.json'), 'r') as f:
                self.ann_list = json.load(f)
                

            if opt.half:
                new_ann_list = []            
                for i in range(len(self.ann_list)):
                    if i%2 ==0 :
                        new_ann_list.append(self.ann_list[i])
                    
                self.ann_list = new_ann_list
                    
            
                
        print('NUM DATA:', len(self.ann_list))
        #exit(0)
    def name(self):
        return "ETRIDataset"

    def _estimate_joints_mask(self, maskshape, in_pose, from_joint, to_joint, alpha, img=None):
        mask = np.zeros([maskshape[0], maskshape[1]])
        radius = int(maskshape[1]*alpha) 
        #print('heuri distance!', self.heuristic_distance)
        est_hand = [0.0,0.0]
        if in_pose[from_joint*3+2] >0 and in_pose[to_joint*3+2] >0:

            elbowToHand = np.array([in_pose[to_joint*3+0] - in_pose[from_joint*3+0], in_pose[to_joint*3+1] - in_pose[from_joint*3+1]])
            unitEtoH = elbowToHand/np.sqrt(np.sum(elbowToHand**2))

            est_hand = np.array([in_pose[to_joint*3+0], in_pose[to_joint*3+1]]) + unitEtoH*radius*1.5#1.5#0.6#1.0
            cv2.circle(mask,(int(est_hand[0]), int(est_hand[1])), radius, (255,255,255), -1)    

        return mask, [int(est_hand[0]),int(est_hand[1]), int(radius)]

    def _swap_seglabel(self, seg, idx):
        assert len(idx) == 2, 'idx should be 2-dim'
        
        seg = np.where(seg==idx[0], 999, seg)
        seg = np.where(seg==idx[1], idx[0], seg)
        seg = np.where(seg==999, idx[1], seg)   
        
        return seg
        
    def _blur_and_enlarge_mask(self, mask):
        blur_mask = cv2.blur(mask,(21,21))
        enlarged_mask = np.clip(blur_mask*4.0,0.0,1.0)
    
        return enlarged_mask

    def _posegt_lrflip(self, pose):        
        ##-- fliplr for all joints
        for j in range(int(len(pose)/3)):
            #print(j, self.fine_width)
            #print('*before:', pose[j*3:(j+1)*3])
            pose[j*3+0] = self.fine_width - pose[j*3+0] - 1
            #print('*after:', pose[j*3:(j+1)*3])
    
        #print('length pose pair:', len(self.pose_pair))
        ##-- flip index
        for i in range(len(self.pose_gt_pair)):
            f = self.pose_gt_pair[i][0]
            t = self.pose_gt_pair[i][1]
            
            _tmp = pose[f*3:(f+1)*3].copy()
            pose[f*3:(f+1)*3] = pose[t*3:(t+1)*3].copy()
            pose[t*3:(t+1)*3] = _tmp   
        return pose                    

    def _poseop_lrflip(self, pose, fine_width):        

        for j in range(int(len(pose)/3)):
            pose[j*3+0] = fine_width - pose[j*3+0] - 1

        for i in range(len(self.pose_op_pair)):
            f = self.pose_op_pair[i][0]
            t = self.pose_op_pair[i][1]
            
            _tmp = pose[f*3:(f+1)*3].copy()
            pose[f*3:(f+1)*3] = pose[t*3:(t+1)*3].copy()
            pose[t*3:(t+1)*3] = _tmp   
        return pose        
        
    def _handop_lrflip(self, handL, handR, fine_width):
        for i in range(len(handL)//3):
            handL[i*3+0] = fine_width - handL[i*3+0] - 1
            handR[i*3+0] = fine_width - handR[i*3+0] - 1            
        
        tmp_buf = handL.copy()
        handL = handR.copy()
        handR = tmp_buf
        
        return handL, handR                    
        
    def _EstimateOvalForKpts1(self, kpts3, indices=[], offsetXY=[0,0]):
        if len(kpts3)%3 !=0:
            print('ERROR Keypoints should be multiple of 3. : EstimateCircleForKpts')
            exit(0)

        if len(indices)==0:
            indices = list(range(0,len(kpts3)//3))

        sumX, sumY = 0.0, 0.0
        cntX, cntY = 0, 0        

        for i in indices:
            if kpts3[i*3+2]>0.001:
                sumX += kpts3[i*3]
                cntX += 1
                sumY += kpts3[i*3+1]
                cntY += 1        
        if cntX == 0 or cntY == 0:
            #print(kpts3)
            #print(indices)
            #print('ERROR: keypoint count is zero')
            #exit(0)
            return [0,0], 0
        meanpt = [sumX/cntX, sumY/cntY] 
        
        maxdist = 0.0
        for i in indices:
            tmpdist = np.sqrt((meanpt[0]-kpts3[i*3])**2 + (meanpt[1]-kpts3[i*3+1])**2)
            if maxdist < tmpdist:
                maxdist = tmpdist
        if maxdist > 10 :
            maxdist=10
        return [int(meanpt[0]), int(meanpt[1])], [int(maxdist)+offsetXY[0], int(maxdist)+offsetXY[1]] 
        
        
    def _EstimateCircleForKpts1(self,kpts3, indices=[], offset=0):
        if len(kpts3)%3 !=0:
            print('ERROR Keypoints should be multiple of 3. : EstimateCircleForKpts')
            exit(0)

        if len(indices)==0:
            indices = list(range(1,len(kpts3)//3))

        sumX, sumY = 0.0, 0.0
        cntX, cntY = 0, 0        

        for i in indices:
            if kpts3[i*3+2]>0.001:
                sumX += kpts3[i*3]
                cntX += 1
                sumY += kpts3[i*3+1]
                cntY += 1        
        if cntX == 0 or cntY == 0:
            return [0,0], 0
        meanpt = [sumX/cntX, sumY/cntY]
        
        maxdist = 0.0
        for i in indices:
            tmpdist = np.sqrt((meanpt[0]-kpts3[i*3])**2 + (meanpt[1]-kpts3[i*3+1])**2)
            if maxdist < tmpdist:
                maxdist = tmpdist
         
        return [int(meanpt[0]), int(meanpt[1])], int(maxdist)+offset               

    def __getitem__(self, index):
        ann = self.ann_list[index]   
        
        postfix=''    

        s_file = ann['wearing'] #Model-Image file
        w_info = [ann['main_top'], ann['bottom']] #Wearing Information

        Model_Image_file = osp.join(self.root, 'Model-Image'+postfix,s_file)
        Model_Parse_file = osp.join(self.root, 'Model-Parse_png', s_file.replace('.jpg','.png'))
        Model_dp_file = osp.join(self.root, 'Model-dp_png',s_file.replace('.jpg','.png'))            
        #if not self.is_train:
        #    Model_est_Parse_file = osp.join(self.root,str(self.fine_height), 'Model-Parse_'+self.opt.posfix_parse, s_file.replace('.jpg','.png'))
        
        top_Image_file =''
        top_Parse_file =''
        bottom_Image_file =''
        bottom_Parse_file =''
        #if self.stage == 'GMM_ETRI' or self.stage == 'GMMBODY_ETRI':

       
        if self.stage == 'TMP' or self.stage == '1ST':
            top_Image_file = osp.join(self.root,'Item-Image'+postfix,ann['main_top']+'_F.jpg')
            top_Parse_file = osp.join(self.root,'Item-Parse_png',ann['main_top']+'_F.png')

            if ann['bottom'] == None:
                bottom_Image_file = None
                bottom_Parse_file = None
            else:
                bottom_Image_file = osp.join(self.root,'Item-Image'+postfix,ann['bottom']+'_F.jpg')
                bottom_Parse_file = osp.join(self.root,'Item-Parse_png',ann['bottom']+'_F.png')
    
           
        #Keypoints_file = osp.join(self.root,str(self.fine_height),'Model-Pose',s_file[:-4]+'.json')
        Keypoints_file = osp.join(self.root,'Model-Pose_infer',s_file[:-4]+'_keypoints.json')
        #print('HERE!!')
        
        if not osp.isfile(Model_Image_file):

            print('MODEL_IMAGE FILE: ', Model_Image_file, ' DOES NOT EXIST')
            exit(0)
        if not osp.isfile(Model_Parse_file):
            print('MODEL_PARSE FILE: ', Model_Parse_file, ' DOES NOT EXIST')
            exit(0)
        if not osp.isfile(Model_dp_file):
            print('MODEL_DENSEPOSE FILE: ', Model_dp_file, ' DOES NOT EXIST')
            exit(0)            
        if not osp.isfile(top_Image_file):
            print('item:', top_Image_file)
            print('TOP_ITEM_IMAGE FILE: ', top_Image_file, ' DOES NOT EXIST')
            exit(0)
        if not osp.isfile(top_Parse_file):
            print('TOP_ITEM_PARSE FILE: ', top_Parse_file, ' DOES NOT EXIST')
            exit(0)
        if bottom_Image_file != None and not osp.isfile(bottom_Image_file):
            print('BOTTOM_ITEM_IMAGE FILE: ', bottom_Image_file, ' DOES NOT EXIST')
            exit(0)
        if bottom_Parse_file != None and not osp.isfile(bottom_Parse_file):
            print('BOTTOM_ITEM_PARSE FILE: ', bottom_Parse_file, ' DOES NOT EXIST')
            exit(0)
        if not osp.isfile(Keypoints_file):
            print('KPTS_FILE: ', Keypoints_file, ' DOES NOT EXIST')
            exit(0)

        p = random.random()
        if p > self.p_aug and self.is_train:
            ApplyAug = True
        else:
            ApplyAug = False

        ##<< [ SC: LOAD IMAGE AND SEGMENTATION ] 
        top_c_ori = cv2.cvtColor(cv2.imread(top_Image_file),cv2.COLOR_BGR2RGB)
        top_cm = cv2.cvtColor(cv2.imread(top_Parse_file),cv2.COLOR_BGR2RGB)[:,:,0]
        im = cv2.cvtColor(cv2.imread(Model_Image_file),cv2.COLOR_BGR2RGB)                
        parse_array = cv2.cvtColor(cv2.imread(Model_Parse_file),cv2.COLOR_BGR2RGB)[:,:,0] # [h, w, 3]
        dp_array = cv2.cvtColor(cv2.imread(Model_dp_file),cv2.COLOR_BGR2RGB)[:,:,0] # [h, w, 3]
        
        #cv2.imshow('dp', dp_array*10)
        #cv2.waitKey(0)
        
        #if not self.is_train:
        #    gt_parse_array = parse_array.copy()
            #parse_array = cv2.cvtColor(cv2.imread(Model_est_Parse_file),cv2.COLOR_BGR2RGB)[:,:,0] # [h, w, 3]
        
        with open(Keypoints_file,'r') as f:
            keypt = json.load(f)
            #pose = np.array(keypt['landmarks'])
            pose = keypt['pose']
            handL = keypt['handL']
            handR = keypt['handR']

        if ApplyAug:
            #-- flip top image
            top_c_ori = np.fliplr(top_c_ori)
            #-- flip top segmentation label
            top_cm = np.where(top_cm==3, 999, top_cm)
            top_cm = np.where(top_cm==4, 3, top_cm)
            top_cm = np.where(top_cm==999, 4, top_cm)
            #-- flip segmentation            
            top_cm = np.fliplr(top_cm)            
            
            ##-- Flip model info
            im = np.fliplr(im).copy()   
            
            #-- flip model segmentation                     
            parse_array = self._swap_seglabel(parse_array,[9,10])   #-- flip top slvs label
            parse_array = self._swap_seglabel(parse_array,[14,15])  #-- flip top bottoms label            
            parse_array = self._swap_seglabel(parse_array,[11,12])   #-- flip hands label
            parse_array = self._swap_seglabel(parse_array,[17,18])  #-- flip legs label                        
            parse_array = self._swap_seglabel(parse_array,[19,20])  #-- flip shoes label                                    
            parse_array = np.fliplr(parse_array) 

            ## -- flip LR in dp     
            dp_array = self._swap_seglabel(dp_array,[3, 4])
            dp_array = self._swap_seglabel(dp_array,[5, 6])
            dp_array = self._swap_seglabel(dp_array,[7, 8])        
            dp_array = self._swap_seglabel(dp_array,[9, 10])                
            dp_array = self._swap_seglabel(dp_array,[11, 12])                
            dp_array = self._swap_seglabel(dp_array,[13, 14])                                
            dp_array = self._swap_seglabel(dp_array,[15, 16])                                        
            dp_array = self._swap_seglabel(dp_array,[17, 18])
            dp_array = self._swap_seglabel(dp_array,[19, 20])
            dp_array = self._swap_seglabel(dp_array,[21, 22])
            dp_array = self._swap_seglabel(dp_array,[23, 24])                                   
            dp_array = np.fliplr(dp_array)
            
            #-- flip model's pose data
            pose = self._poseop_lrflip(pose,self.fine_width)
            handL, handR = self._handop_lrflip(handL, handR, self.fine_width)            

        top_cm = top_cm[:,:,None]        
        #-- augmentation for bottom data
        if bottom_Image_file != None:
            bottom_c_ori = cv2.cvtColor(cv2.imread(bottom_Image_file),cv2.COLOR_BGR2RGB)
            bottom_cm = cv2.cvtColor(cv2.imread(bottom_Parse_file),cv2.COLOR_BGR2RGB)[:,:,0]
            
            if ApplyAug:
                #-- flip bottom image
                bottom_c_ori = np.fliplr(bottom_c_ori)
                #-- flip bottom segmentation label
                bottom_cm = np.where(bottom_cm==8, 999, bottom_cm)
                bottom_cm = np.where(bottom_cm==9, 8, bottom_cm)
                bottom_cm = np.where(bottom_cm==999, 9, bottom_cm)
                #-- flip bottom segmentation                
                bottom_cm = np.fliplr(bottom_cm)
                
            bottom_cm = bottom_cm[:,:,None]
            #print('bottom_cm:',bottom_cm.shape)            
        else:
            bottom_c_ori = np.ones([self.fine_height,self.fine_width,3], dtype=np.uint8)*255
            bottom_cm = np.zeros([self.fine_height, self.fine_width,1], dtype=np.uint8) #220920 np.int->np.uint8

        #cv2.imshow('img', im)
        #cv2.imshow('dp', (dp_array*10).astype(np.uint8))
        #cv2.waitKey(0)

        m_dp = []
        for i in range(25):
            m_dp.append((dp_array==i).astype(np.float32))
            #cv2.imshow('test', (dp_array==0).astype(np.uint8)*255)
            #cv2.waitKey(0)
            
        m_dp = np.array(m_dp).transpose(1,2,0)
        #print('mdp:', m_dp.shape)
        m_dp = self.transformG(m_dp)

        ##<< [ SC: Arrange masks for cloth items]
        if self.stage=='TMP' or self.stage == '1ST':
            top_c_mask_body = (top_cm==5).astype(np.uint8)   # mask for cloth
            top_c_mask_slvs_left = (top_cm==4).astype(np.uint8)
            top_c_mask_slvs_right = (top_cm==3).astype(np.uint8)
            top_visible_mask = np.clip(top_c_mask_body + top_c_mask_slvs_left + top_c_mask_slvs_right, 0,1)
            top_c = np.multiply(top_c_ori,top_visible_mask)
            top_c += np.multiply(255,(1-top_visible_mask))
            
            top_c_seg = np.concatenate((top_c_mask_body, top_c_mask_slvs_left, top_c_mask_slvs_right), axis=2).astype(np.float32)
            
            bottom_c_mask_body = (bottom_cm==7).astype(np.uint8) + (bottom_cm==11).astype(np.uint8)  # pants_body + skirt
            bottom_c_mask_slvs_left = (bottom_cm==9).astype(np.uint8)  # pants_rsleeve + pants_lsleeve
            bottom_c_mask_slvs_right = (bottom_cm==8).astype(np.uint8)   # pants_rsleeve + pants_lsleeve          
            bottom_visible_mask = np.clip(bottom_c_mask_body + bottom_c_mask_slvs_left + bottom_c_mask_slvs_right, 0,1)
            bottom_c = np.multiply(bottom_c_ori,bottom_visible_mask)
            bottom_c += np.multiply(255,(1-bottom_visible_mask))
            
            bottom_c_seg = np.concatenate((bottom_c_mask_body, bottom_c_mask_slvs_left, bottom_c_mask_slvs_right), axis=2).astype(np.float32)            

        #cv2.imshow('img', im)
        #cv2.imshow('top_m_seg', (top_c_seg*80).astype(np.uint8))
        #cv2.imshow('bottom_m_seg', (bottom_c_seg*80).astype(np.uint8))
        #cv2.waitKey(0)


        #cv2.imshow('top_c', top_c)
        #cv2.imshow('bottom_c', bottom_c)
        ## << [ Numpy to Torch Tensor]
        top_c = self.transform(top_c)
        
        ##220920
        #top_cm = top_cm.transpose(2,0,1).astype(np.float32)
        #top_cm = torch.from_numpy(top_cm)              

        bottom_c = self.transform(bottom_c)
        
        #220920
        #bottom_cm = bottom_cm.transpose(2,0,1).astype(np.float32)
        #bottom_cm = torch.from_numpy(bottom_cm)
        if self.stage == 'TMP' or self.stage == '1ST':
            top_c_seg = self.transformG(top_c_seg)
            bottom_c_seg = self.transformG(bottom_c_seg)
            
        ##<< [ SC: Arrange masks for models]
        # -- Generate wearing agnostics data
        #if self.is_train:
        wagnostic_mask = parse_array
        #else:
        #    wagnostic_mask = gt_parse_array            
        
        #220920
        #parse_shape = (parse_array >0).astype(np.float32)       
        #parse_head = (wagnostic_mask ==2).astype(np.float32) + \
        #             (wagnostic_mask==3).astype(np.float32) + \
        #             (wagnostic_mask==4).astype(np.float32) #hair, face, neck
        #phead = torch.from_numpy(parse_head) # [0,1]
        #mask_p = (wagnostic_mask>0).astype(np.uint8)

        if self.stage == 'TMP' or self.stage == '1ST':
            ## -- arrange structure-aware mask of top and bottom
            top_m_body_mask = (parse_array==8).astype(np.float32) # only torso from studio images
            top_m_slvs_mask_left = (parse_array==10).astype(np.float32) 
            top_m_slvs_mask_right = (parse_array==9).astype(np.float32)
            top_parse_cloth = top_m_body_mask + top_m_slvs_mask_left + top_m_slvs_mask_right
     
            top_m_seg = np.concatenate((top_m_body_mask[:,:,None], top_m_slvs_mask_left[:,:,None],top_m_slvs_mask_right[:,:,None]), axis=2)
            
            ##220929 - top wearing mask
            top_m_wearing = np.zeros(top_m_body_mask.shape).astype(np.float32)[:,:,None]
            if not self.is_unpaired:            
                column_sum_topbody = np.sum(top_m_body_mask, axis=1)
                #print('shape:', column_sum_topbody.shape)
                for k in range(top_m_body_mask.shape[0]-1, -1, -1):
                    if column_sum_topbody[k] != 0:
                        max_height = k
                        break

            else:
                k = int(top_m_body_mask.shape[0]*ann['wg'])  
                ##print('here')          
            top_m_wearing[0:k,:,:] = 1.0                            

            bottom_m_body_mask = (parse_array==13).astype(np.float32) + (parse_array==16).astype(np.float32)# only hip and skirt
            bottom_m_slvs_mask_left = (parse_array==15).astype(np.float32) # only slv of bottom from studio images
            bottom_m_slvs_mask_right = (parse_array==14).astype(np.float32)
            bottom_parse_cloth = bottom_m_body_mask + bottom_m_slvs_mask_left + bottom_m_slvs_mask_right

            bottom_m_seg = np.concatenate((bottom_m_body_mask[:,:,None], bottom_m_slvs_mask_left[:,:,None], bottom_m_slvs_mask_right[:,:,None]), axis=2)
            
            #cv2.imshow('img', im)
            #cv2.imshow('top_m_seg', (top_m_seg*80).astype(np.uint8))
            #cv2.imshow('bottom_m_seg', (bottom_m_seg*80).astype(np.uint8))
            #cv2.waitKey(0)

            ## -- area to exclude for estimating warping loss. hair(2), face(3), r_arm(11), l_arm(12), r_leg(17) l_leg(18)
            parse_occ = (wagnostic_mask==2).astype(np.float32) + \
                     (wagnostic_mask==3).astype(np.float32) + \
                     (wagnostic_mask==11).astype(np.float32) + \
                     (wagnostic_mask==12).astype(np.float32) #+ \
                     #(wagnostic_mask==17).astype(np.float32) + \
                     #(wagnostic_mask==18).astype(np.float32)
                     
            #toptorso_occ = parse_occ.copy()
            #toptorso_occ = toptorso_occ + (wagnostic_mask==9).astype(np.float32) + \
            #         (wagnostic_mask==10).astype(np.float32)  #left leg

            #cv2.imshow('occ', parse_occ)
            #cv2.waitKey(0)
            parse_occ = (1-parse_occ)[None,:,:]            
            parse_occ = torch.from_numpy(parse_occ)
            
            #toptorso_occ = (1-toptorso_occ)[None,:,:]
            #toptorso_occ = torch.from_numpy(toptorso_occ)

            top_pcm = torch.from_numpy(top_parse_cloth)[None,:,:] # [0,1]
            #top_pcm_torso = torch.from_numpy(top_m_body_mask)[None,:,:]
            #top_pcm_slvs = torch.from_numpy(top_m_slvs_mask_left+top_m_slvs_mask_right)[None,:,:]
            bottom_pcm = torch.from_numpy(bottom_parse_cloth)[None,:,:] # [0,1]    

            #top_m_body_mask = torch.from_numpy(top_m_body_mask)
            #top_m_slvs_mask_left = torch.from_numpy(top_m_slvs_mask_left)
            #top_m_slvs_mask_right = torch.from_numpy(top_m_slvs_mask_right)
            #bottom_m_body_mask = torch.from_numpy(bottom_m_body_mask)
            #bottom_m_slvs_mask_left = torch.from_numpy(bottom_m_slvs_mask_left)
            #bottom_m_slvs_mask_right = torch.from_numpy(bottom_m_slvs_mask_right)      
            
            top_m_seg = self.transformG(top_m_seg)  
            bottom_m_seg = self.transformG(bottom_m_seg)
            top_m_wearing = self.transformG(top_m_wearing)            


        #print('make elipse')
        ##<< - Data arrangement for agnostic input for image and mask
        #idx_footL = [19, 20, 21] 
        #idx_footR = [22, 23, 24]
        idx_footL = [19, 20] 
        idx_footR = [22, 23]          
        
        
        if self.fine_height==512:        
            offset_elipse = [30, 6]
            offset_circle = 6
        elif self.fine_height==256:
            offset_elipse = [15,3]
            offset_circle = 3
               
        augratio = (random.random() / 3) + 0.65
        c, r = self._EstimateOvalForKpts1(pose, idx_footL, offset_elipse) 
        footL_mask = np.zeros([self.fine_height, self.fine_width])        
        if r!=0:    
            if self.is_train:
                r[0] = int(augratio*r[0])
                r[1] = int(augratio*r[1])
            footL_mask = cv2.ellipse(footL_mask, c, r, 0, 0, 360, (255,255,255), -1)/255

        c, r = self._EstimateOvalForKpts1(pose, idx_footR, offset_elipse) 
        footR_mask = np.zeros([self.fine_height, self.fine_width])
        if r!=0:
            if self.is_train:        
                r[0] = int(augratio*r[0])
                r[1] = int(augratio*r[1])        
            footR_mask = cv2.ellipse(footR_mask, c, r, 0, 0, 360, (255,255,255), -1)/255                           
            
        c, r = self._EstimateCircleForKpts1(handL,[],offset_circle)
        handL_mask = np.zeros([self.fine_height, self.fine_width])
        if r!=0:
            if self.is_train:        
                r = int(r*augratio)
            handL_mask = cv2.circle(handL_mask, c, r, (255,255,255),-1)/255                
            
        c, r = self._EstimateCircleForKpts1(handR,[],offset_circle)
        handR_mask = np.zeros([self.fine_height, self.fine_width])
        if r!=0:
            if self.is_train:                
                r = int(r*augratio)        
            handR_mask = cv2.circle(handR_mask, c, r, (255,255,255),-1)/255             

        ## Arrange Densepose masks
        #dp_seg = []
        #for k in range(1,25):
        #    dp_seg.append((np_dpmask==k).astype(np.float32))
        #dp_seg = np.array(dp_seg).astype(np.float32)
        
        ##220920 float -> uint8
        ## Arrange masks using masks of hands and feet
        TARGET_INDICES=[0,1,2,3,4,8,9,10,11,12,13,14,15,16,17,18,19,20] ## Except indices of outer_torso, outer_Rslv, outer_Lslv (5, 6, 7)
        #REMAIN_INDICES=[1,2,3,11,12,17,18,19,20] ## Delete Inner clothe 8, 9, 10
        REMAIN_INDICES=[1,2,3,11,12,19,20] ## Delete Inner clothe 8, 9, 10

        remain_seg = np.zeros((wagnostic_mask.shape[0], wagnostic_mask.shape[1])).astype(np.float32)
        delete_seg = np.zeros((wagnostic_mask.shape[0], wagnostic_mask.shape[1])).astype(np.float32)
        #back_seg = (wagnostic_mask==0).astype(np.float32).copy()
        wagnostic_seg = [] #-- input segmentation map
        full_masks=[]
        _amp = 0.001 
        ## -- Arrange Info: Segmentation region of interest
        for c in TARGET_INDICES:
            ftmp_seg = (wagnostic_mask==c).astype(np.float32).copy()

            full_masks.append(ftmp_seg)            
            if c in REMAIN_INDICES:                

                roi_mask = np.ones(wagnostic_mask.shape).astype(np.float32)
                if c == 11:
                    roi_mask = handR_mask
                elif c==12:
                    roi_mask = handL_mask
                elif c==17 or c==19:
                    roi_mask = footR_mask
                elif c==18 or c==20:
                    roi_mask = footL_mask

                tmp_seg = (wagnostic_mask==c).astype(np.float32) * roi_mask
                remain_seg += tmp_seg                    
                #tmp_seg = (tmp_seg - tmp_seg*noiseZ + (1-tmp_seg)*noiseZ).copy()
                wagnostic_seg.append(tmp_seg)
                
                delete_seg += (wagnostic_mask==c).astype(np.float32) * (1-roi_mask)        
                #print('!!')        
            else:
                ##-- Save empty channel
                wagnostic_seg.append(np.zeros((ftmp_seg.shape)))            
                #if c!=0:
                delete_seg += (wagnostic_mask==c).astype(np.float32)           

        ksize = (0,0)            
        if self.fine_height == 256:
            ksize = (15, 15)
        elif self.fine_height == 512:
            ksize = (41, 41)    

        delete_seg_reverse = np.ones((delete_seg.shape)).astype(np.float32) - delete_seg.astype(np.float32) # delete_seg = 0. the other 1
        delete_seg_reverse = np.maximum(delete_seg_reverse, remain_seg.astype(np.float32))

        im_in = im.copy()
        im_in= (im_in*delete_seg_reverse[:,:,None].astype(np.uint8)) + np.ones((im_in.shape)).astype(np.uint8)*200*(1-delete_seg_reverse[:,:,None].astype(np.uint8))
        
        
        #for i in range(18):
        #    print(i)
        #    cv2.imshow('test', wagnostic_seg[i])
        #    cv2.waitKey(0)
        #cv2.imshow('in', im)
        #cv2.imshow('imin', im_in)
        #cv2.waitKey(0)

        remain_seg = np.array(remain_seg).astype(np.float32)  # [ 10, 256, 192]
        mod_m_seg = np.array(wagnostic_seg).astype(np.float32).transpose(1,2,0)
        full_masks = np.array(full_masks).astype(np.float32) # [18, 256, 192] - target (GT)
        
        full_maskmap = np.zeros([full_masks.shape[1], full_masks.shape[2]]).astype(np.long)
        
        for i in range(full_masks.shape[0]):
            full_maskmap = full_maskmap + (full_masks[i]*i)
     
        mod_m_seg = self.transformG(mod_m_seg)
        m_seg = torch.from_numpy(full_masks)
        m_segmap = torch.from_numpy(full_maskmap).type(torch.long)
     

        if self.stage=='TMP' or self.stage == '1ST':            
            im = self.transform(im) # [-1,1]
            im_in = self.transform(im_in)            

        #print('11')
        # upper cloth
        if self.stage=='TMP' or self.stage == '1ST':
            top_im_c = im * top_pcm + (1 - top_pcm) # [-1,1], fill 1 for other parts #cloth region in model shot.
            bottom_im_c = im*bottom_pcm + (1-bottom_pcm)

        pose_data = np.zeros([len(self.target_kpts_idx)*3]).astype(np.float32)
        #print('posedatashape:', pose_data.shape)
        j=0
        for i in self.target_kpts_idx:
            pose_data[j*3+0] = pose[i*3+0]
            pose_data[j*3+1] = pose[i*3+1]
            pose_data[j*3+2] = pose[i*3+2]
            j = j+1
            
        ##<< [SC : load pose points ]
        point_num = pose_data.shape[0]//3
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i*3]
            pointy = pose_data[i*3+1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transformG(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transformG(im_pose)      

        if self.stage=='TMP' or self.stage == '1ST':
            top_m_body_mask = top_m_body_mask[None,:,:]        
            top_m_slvs_mask_left = top_m_slvs_mask_left[None,:,:]
            top_m_slvs_mask_right = top_m_slvs_mask_right[None,:,:]
            
            bottom_m_body_mask = bottom_m_body_mask[None,:,:]        
            bottom_m_slvs_mask_left = bottom_m_slvs_mask_left[None,:,:]
            bottom_m_slvs_mask_right = bottom_m_slvs_mask_right[None,:,:]
            
            im_g = Image.open('grid.png')
            im_g = im_g.resize((self.fine_width, self.fine_height))
            im_g = self.transform(im_g)            
            
            agnostic = torch.cat([m_dp, pose_map, mod_m_seg,im_in,top_m_wearing], 0) # [0:25], [25:42],[42:60]
  
        
        ##<< [ SC : ARRANGE NAMES ]
        top_c_name = ann['main_top']
        if ann['bottom'] is not None:
            bottom_c_name = ann['bottom']
        else:
            bottom_c_name = ''
        m_name = ann['wearing']
        
        if self.stage=='TMP' or self.stage=='1ST':

            result = {
                'top_c_name':   top_c_name,    
                'bottom_c_name':   bottom_c_name,    
                'm_name':  m_name,    # for visualization or ground truth
                'agnostic': agnostic,   # for input v
                'top_c_cloth':    top_c,          # for input v
                'top_c_seg' : top_c_seg,
                'top_m_cloth': top_im_c,    # for ground truth v
                'bottom_c_cloth':    bottom_c,          # for input v
                'bottom_c_seg': bottom_c_seg,
                'bottom_m_cloth': bottom_im_c,    # for ground truth v
                #'model_dp': m_dp,                
                'mod_m_img': im_in,
                'model_seg': m_seg,
                'model_segmap': m_segmap,
                'pose_image': im_pose,  # for visualization v
                'grid_image': im_g,     # for visualization v
                'image':    im,         # for visualization v
                'occ_parse':parse_occ,   # v
                #'occ_toptorso':toptorso_occ,
                'top_m_seg': top_m_seg,
                'bottom_m_seg': bottom_m_seg,
                #'reserve_mask': delete_seg_reverse[None,:,:],
                #'image_raw': im_raw,
                #'bg_mask': back_seg

            }               

        return result

    def __len__(self):
        return len(self.ann_list)

class FTBDataLoader(object):
    def __init__(self, opt, dataset):
        super(FTBDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
    
    def GetLength(self):
        return len(self.dataset.ann_list)


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    #parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

