#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn.utils import spectral_norm
import os

import numpy as np

import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print("Network [{}] was created. Total number of parameters: {:.1f} million. "
              "To see the architecture, do print(network).".format(self.__class__.__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if 'BatchNorm2d' in classname:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif ('Conv' in classname or 'Linear' in classname) and hasattr(m, 'weight'):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method '{}' is not implemented".format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, *inputs):
        pass        

def get_nonspade_norm_layer(norm_type='instance'):
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        # elif subnorm_type == 'sync_batch':
        #     norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer        

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class NLayerDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.no_ganFeat_loss = opt.no_ganFeat_loss
        nf = opt.ndf

        kw = 4
        pw = int(np.ceil((kw - 1.0) / 2))
        norm_layer = get_nonspade_norm_layer(opt.norm_D)

        input_nc = opt.gen_semantic_nc + 3
        # input_nc = opt.gen_semantic_nc + 13
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=pw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=pw)),
                          nn.LeakyReLU(0.2, False)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=pw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.no_ganFeat_loss = opt.no_ganFeat_loss

        for i in range(opt.num_D):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

def make_grid(N, iH, iW):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    grid = torch.cat([grid_x, grid_y], 3).cuda()
    return grid

class WGFVITON(nn.Module):
    def __init__(self, opt, item_nc, model_nc, output_nc, ngf=64, norm_G = 'spectralinstance', norm_layer=nn.BatchNorm2d, target_height=256):
        super(WGFVITON, self).__init__()
        self.warp_feature = 256
        self.out_layer_opt = 256
        self.SPADE_input = 3+3+3 # top img, bt img, Ia img, densepose
        
        if target_height==256:
            self.sH = 8
            self.sW = 6
        elif target_height==512:
            self.sH = 16
            self.sW = 12
            
        #print('h,w:', self.sH, self.sW)
        
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.ClothEncoder = nn.Sequential(
            ResBlock(item_nc, ngf, norm_layer=norm_layer, scale='down'),  # 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),  # 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 8
        )
        
        self.PoseEncoder = nn.Sequential(
            ResBlock(model_nc, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )

        self.conv = ResBlock(ngf * 4 * 3, ngf * 4, norm_layer=norm_layer, scale='same')        
        
        self.SegDecoder = ResBlock(ngf*4*3, ngf * 4, norm_layer=norm_layer, scale='up')


        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )
        self.flow_conv = nn.ModuleList([
            nn.Conv2d(ngf*4*3, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf*4*3, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf*4*3, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf*4*3, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf*4*3, 4, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        )      
        self.FtoW = nn.Sequential(
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True) , nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())            
        )
        
        self.WtoF = nn.Sequential(
            nn.Sequential(nn.Conv2d(ngf * 4*2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4*2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True) , nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4*2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4*2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4*2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())                
        )
        
        ##SPADE
        self.ref_input_nChannel = 49 # masks for top(3) and bottom(3), densepose(25), Sa(18)
        self.ref_nChannel = 16
        self.SPADE_conv = nn.ModuleList([
            #nn.Conv2d(SPADE_input, ngf*4, kernel_size=3, padding=1),
            nn.Conv2d(self.SPADE_input, self.ref_nChannel, kernel_size=3, padding=1),
            nn.Conv2d(self.SPADE_input, self.ref_nChannel, kernel_size=3, padding=1),
            nn.Conv2d(self.SPADE_input, self.ref_nChannel, kernel_size=3, padding=1),
            nn.Conv2d(self.SPADE_input, self.ref_nChannel, kernel_size=3, padding=1)
        ])          
        self.SPADE_blk = nn.ModuleList([
            SPADEResBlock(ngf * 4 +ngf * 4 + self.ref_nChannel, ngf * 4, self.ref_input_nChannel, norm_G),
            SPADEResBlock(ngf * 4 +ngf * 4 + self.ref_nChannel, ngf * 4, self.ref_input_nChannel, norm_G),
            SPADEResBlock(ngf * 4 +ngf * 4 + self.ref_nChannel, ngf * 4, self.ref_input_nChannel, norm_G),
            SPADEResBlock(ngf * 4 +ngf * 4 + self.ref_nChannel, ngf * 2, self.ref_input_nChannel, norm_G, use_mask_norm=False),
            SPADEResBlock(ngf * 2 +ngf * 4 + self.ref_nChannel, ngf * 1, self.ref_input_nChannel, norm_G, use_mask_norm=False)
        ])                          
       
        self.conv_img = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)                             

        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()                

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print("Network [{}] was created. Total number of parameters: {:.1f} million. "
              "To see the architecture, do print(network).".format(self.__class__.__name__, num_params / 1000000))

    def convergeHolisticFeature(self, att, f_top, f_bt, f_m):
    
        attsum = torch.sum(att,1)[:,None,:,:]
        
        #print(f_x.size(), att[:,1:2,:,:].size(), attsum.size())
        f_top = torch.mul(f_top, torch.div(att[:,0:1,:,:],attsum))        
        f_bt = torch.mul(f_bt, torch.div(att[:,1:2,:,:],attsum))
        f_m = torch.mul(f_m, torch.div(att[:,2:3,:,:],attsum))
    
        return f_top + f_bt + f_m

    #def forward(self, input_top, input_bt, input_model, upsample='biliear'):
    def forward(self,input_top, input_bt, input_model, upsample='bilinear'):
        E_Ctop_list = []
        E_Cbt_list = []
        E_m_list = []
        flow_list = []
        #att_list = []
        features=[]

        # Feature Pyramid Network
        for i in range(5):     
            if i == 0:
                E_Ctop_list.append(self.ClothEncoder[i](input_top))
                E_Cbt_list.append(self.ClothEncoder[i](input_bt)) 
                Em = self.PoseEncoder[i](input_model)    
                #E_m_list.append(self.PoseEncoder[i](input_model))
                #features.append(self.SPADE_conv[i](
                #E_m_list.append(self.PoseEncoder[i](input_model))

            else:
                E_Ctop_list.append(self.ClothEncoder[i](E_Ctop_list[i-1]))
                E_Cbt_list.append(self.ClothEncoder[i](E_Cbt_list[i-1]))                
                Em = self.PoseEncoder[i](Em)
                #E_m_list.append(self.PoseEncoder[i](E_m_list[i-1]))
                #E_m_list.append(self.PoseEncoder[i](E_m_list[i-1]))   
                
        for i in range(5):
            N, _, iH, iW = E_Ctop_list[4 - i].size()
            grid = make_grid(N, iH, iW)
            
            if i==0:
                Etop = E_Ctop_list[4-i]
                Ebt = E_Cbt_list[4-i]
                #Em = E_m

                #print('etop:', Etop.size())
                f_in = torch.cat([Em, Etop, Ebt],1)
                flows = self.flow_conv[i](f_in).permute(0,2,3,1)               
                flow_list.append([flows[:,:,:,0:2], flows[:,:,:,2:4]])
                               
                flow_top_norm = torch.cat([flows[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flows[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)                
                flow_bt_norm = torch.cat([flows[:, :, :, 2:3] / ((iW/2 - 1.0) / 2.0), flows[:, :, :, 3:4] / ((iH/2 - 1.0) / 2.0)], 3)            
                
                warped_Etop = F.grid_sample(Etop, flow_top_norm + grid, padding_mode = 'border')     
                warped_Ebt = F.grid_sample(Ebt, flow_bt_norm + grid, padding_mode = 'border')

                #x = self.conv(torch.cat([warped_Etop, warped_Ebt, Em],1)) # [b,256, 8, 6]
                #x = self.up(x)
                x = self.SegDecoder(torch.cat([warped_Etop, warped_Ebt, Em],1)) # [b, 256, 16, 12] 

            else:
                ## Estimate Warping
                Etop = F.interpolate(Etop, scale_factor=2, mode=upsample) + self.conv1[4 - i](E_Ctop_list[4 - i])                
                flow_top = F.interpolate(flow_list[i - 1][0].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)  #
                flow_top_norm = torch.cat([flow_top[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_top[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                warped_Etop = F.grid_sample(Etop, flow_top_norm + grid, padding_mode = 'border')                                
                
                Ebt = F.interpolate(Ebt, scale_factor=2, mode=upsample) + self.conv1[4 - i](E_Cbt_list[4 - i])                                
                flow_bt = F.interpolate(flow_list[i - 1][1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)  #           
                flow_bt_norm = torch.cat([flow_bt[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_bt[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)  
                warped_Ebt = F.grid_sample(Ebt, flow_bt_norm + grid, padding_mode = 'border')

                if i!=1:
                    x = self.up(x)           
                #print(x.size(), warped_Ebt.size())
            
                #print(x.size(), self.FtoW[i](x).size(), warped_Etop.size())
                flows = torch.cat([flow_top, flow_bt],3) + self.flow_conv[i](torch.cat([warped_Etop,warped_Ebt,self.FtoW[i-1](x)],1)).permute(0,2,3,1)
                flow_list.append([flows[:,:,:,0:2], flows[:,:,:,2:4]])                
                
                ## SPADE
                # Resize all information
                resized_input_top = F.interpolate(input_top, size=(self.sH*2**i,self.sW*2**i), mode='nearest')                            
                warped_resized_input_top = F.grid_sample(resized_input_top, flow_top_norm + grid, padding_mode = 'border')
                resized_input_bt = F.interpolate(input_bt, size=(self.sH*2**i,self.sW*2**i), mode='nearest')                            
                warped_resized_input_bt = F.grid_sample(resized_input_bt, flow_bt_norm + grid, padding_mode = 'border')                
                
                resized_input_m = F.interpolate(input_model, size=(self.sH*2**i,self.sW*2**i), mode='nearest')
 
                seg = torch.cat([warped_resized_input_top[:,3:6,:,:], warped_resized_input_bt[:,3:6,:,:], resized_input_m[:,0:25,:,:],resized_input_m[:,42:60,:,:]],1)
                added_feature = self.SPADE_conv[i-1](torch.cat([warped_resized_input_top[:,0:3,:,:], warped_resized_input_bt[:,0:3,:,:],resized_input_m[:,60:63,:,:]],1))  

                WtoF_feature = self.WtoF[i-1](torch.cat([warped_Etop, warped_Ebt],1))
                x = self.SPADE_blk[i-1](torch.cat([x,added_feature, WtoF_feature],1),seg) # [b,256+256+256+16, 32, 24] --> [b, 256, 32, 24 ]
                #x = self.SPADE_blk[i-1](torch.cat([x,added_feature],1),seg) # [b,256+256+256+16, 32, 24] --> [b, 256, 32, 24 ]
        
        N, _, iH, iW = input_top.size()
        grid = make_grid(N, iH, iW)
        
        Etop = F.interpolate(Etop, scale_factor=2, mode=upsample)
        Ebt = F.interpolate(Ebt, scale_factor=2, mode=upsample)
        
        flow_top = F.interpolate(flow_list[-1][0].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_top_norm = torch.cat([flow_top[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_top[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
        warped_top = F.grid_sample(input_top, flow_top_norm + grid, padding_mode='border')
        warped_Etop = F.grid_sample(Etop, flow_top_norm + grid, padding_mode='border')
                        
        grid = make_grid(N, iH, iW)
        flow_bt = F.interpolate(flow_list[-1][1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)        
        flow_bt_norm = torch.cat([flow_bt[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_bt[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)        
        warped_bt = F.grid_sample(input_bt, flow_bt_norm + grid, padding_mode='border')   
        warped_Ebt = F.grid_sample(Ebt, flow_bt_norm + grid, padding_mode='border')                               

        added_feature = self.SPADE_conv[-1](torch.cat([warped_top[:,0:3,:,:],warped_bt[:,0:3,:,:],input_model[:,60:63,:,:]],1))
        seg = torch.cat([warped_top[:,3:6,:,:], warped_bt[:,3:6,:,:], input_model[:,0:25,:,:],input_model[:,42:60,:,:]],1) 
        WtoF_feature = self.WtoF[-1](torch.cat([warped_Etop, warped_Ebt],1))   

        x = self.up(x)                   
        x = self.SPADE_blk[-1](torch.cat([x,added_feature, WtoF_feature],1),seg)
        x = self.tanh(self.conv_img(self.relu(x)))
        #x = self.out_layer(torch.cat([x, input_model, warped_top, warped_bt], 1))

        warped_top_i = warped_top[:, 0:3, :, :]
        warped_top_m = warped_top[:, 3:6, :, :]        
        warped_bt_i = warped_bt[:, 0:3, :, :]        
        warped_bt_m = warped_bt[:, 3:6, :, :]  
                
        return flow_list, x, warped_top_i, warped_top_m, warped_bt_i, warped_bt_m 

class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_nc, out_nc, kernel_size=1,bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
            
        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))

class SPADEResBlock(nn.Module):
    def __init__(self, input_nc, output_nc, semantic_nc, norm_G, use_mask_norm=True):
        super(SPADEResBlock, self).__init__()
        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        self.conv_0 = nn.Conv2d(input_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False)

        subnorm_type = norm_G
        if subnorm_type.startswith('spectral'):
            subnorm_type = subnorm_type[len('spectral'):]
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = SPADE(subnorm_type, input_nc, semantic_nc)
        self.norm_1 = SPADE(subnorm_type, middle_nc, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(subnorm_type, input_nc, semantic_nc)
        
        self.relu = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x,seg))
        else:
            x_s = x
            
        return x_s

    def forward(self, x, seg):#, misalign_mask=None):
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        #if misalign_mask is not None:
        #    misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')

        #x_s = self.shortcut(x, seg, misalign_mask)
        x_s = self.shortcut(x, seg)

        #dx = self.conv_0(self.relu(self.norm_0(x, seg, misalign_mask)))
        #dx = self.conv_1(self.relu(self.norm_1(dx, seg, misalign_mask)))
        dx = self.conv_0(self.relu(self.norm_0(x, seg)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg)))        
        output = x_s + dx
        return output

class PAFGenerator(BaseNetwork):
    def __init__(self, input_nc, outputdim, ngf, semantic_nc, norm_G, height, width):
        super(PAFGenerator, self).__init__()
        #self.num_upsampling_layers = opt.num_upsampling_layers
        self.num_upsampling_layers = 'more'

        self.sh, self.sw = self.compute_latent_vector_size(height, width)

        #nf = opt.ngf
        nf = ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))

        self.head_0 = SPADEResBlock(nf * 16, nf * 16, semantic_nc, norm_G)

        #print('cha:', nf*16+16, nf*16)
        self.G_middle_0 = SPADEResBlock(nf * 16 + 16, nf * 16, semantic_nc, norm_G)
        self.G_middle_1 = SPADEResBlock(nf * 16 + 16, nf * 16, semantic_nc, norm_G)

        self.up_0 = SPADEResBlock(nf * 16 + 16, nf * 8, semantic_nc, norm_G)
        self.up_1 = SPADEResBlock(nf * 8 + 16, nf * 4, semantic_nc, norm_G)
        self.up_2 = SPADEResBlock(nf * 4 + 16, nf * 2, semantic_nc, norm_G, use_mask_norm=False)
        self.up_3 = SPADEResBlock(nf * 2 + 16, nf * 1, semantic_nc, norm_G, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = SPADEResBlock(nf * 1 + 16, nf // 2, semantic_nc, norm_G, use_mask_norm=False)
            nf = nf // 2

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.print_network()
        #self.init_weights(opt.init_type, opt.init_variance)
        self.init_weights()

    def compute_latent_vector_size(self, load_height, load_width):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers '{}' is not recognized".format(self.num_upsampling_layers))

        sh = load_height // 2**num_up_layers
        sw = load_width // 2**num_up_layers
        return sh, sw

    def forward(self, x, seg):
        samples = [F.interpolate(x, size=(self.sh * 2**i, self.sw * 2**i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]
        
        #print(features[0], len(features))

        x = self.head_0(features[0], seg)

        x = self.up(x)
        #x = self.G_middle_0(torch.cat((x, features[1]), 1), seg_div, misalign_mask)
        #print('**', x.size(), features[1].size(), seg.size()) # [3, 1024, 16, 12], [3,16,16,12], [3,18,256,192]
        #x = self.G_middle_0(torch.cat((x, features[1]), 1), seg)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        #x = self.G_middle_1(torch.cat((x, features[2]), 1), seg_div, misalign_mask)
        #print('***', x.size(), features[2].size(), seg.size()) # [3, 1024, 16, 12], [3,16,16,12], [3,18,256,192]
        x = self.G_middle_1(torch.cat((x, features[2]), 1), seg)

        x = self.up(x)
        #x = self.up_0(torch.cat((x, features[3]), 1), seg_div, misalign_mask)
        x = self.up_0(torch.cat((x, features[3]), 1), seg)
        x = self.up(x)
        #x = self.up_1(torch.cat((x, features[4]), 1), seg_div, misalign_mask)
        x = self.up_1(torch.cat((x, features[4]), 1), seg)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), 1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)

        x = self.conv_img(self.relu(x))
        return self.tanh(x)

class SPADE(nn.Module):
   def __init__(self, config_text, norm_nc, label_nc):
       super().__init__()

       self.noise_scale = nn.Parameter(torch.zeros(norm_nc))
       #print(config_text,'*')
       #assert config_text.startswith('spectral')
       param_free_norm_type = config_text

       if param_free_norm_type == 'instance':
           self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
       #elif param_free_norm_type == 'syncbatch':
       #    self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
       elif param_free_norm_type == 'batch':
           self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
       else:
           raise ValueError('%s is not a recognized param-free norm type in SPADE'
                            % param_free_norm_type)
       # The dimension of the intermediate embedding space. Yes, hardcoded.
       nhidden = 128
       ks = 3
       pw = ks // 2

       self.mlp_shared = nn.Sequential(
           nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
           nn.ReLU()
       )
       self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
       self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

   def forward(self, x, segmap):
       # Part 1. generate parameter-free normalized activations
       normalized = self.param_free_norm(x)

       # Part 2. produce scaling and bias conditioned on semantic map
       segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
       actv = self.mlp_shared(segmap)
       gamma = self.mlp_gamma(actv)
       beta = self.mlp_beta(actv)

       # apply scale and bias
       out = normalized * (1 + gamma) + beta
       return out

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()

def save_coarse_refine_checkpoints(coarse, refine, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    params = {}
    params['coarse'] = coarse.cpu().state_dict()
    params['refine'] = refine.cpu().state_dict()
    torch.save(params, save_path)
    coarse.cuda()
    refine.cuda()

