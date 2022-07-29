import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from geffnet import tf_mobilenetv3_large_100, tf_mobilenetv3_small_100
from geffnet.efficientnet_builder import InvertedResidual, Conv2dSame, Conv2dSameExport
from geffnet.activations import HardSwish

#UTILITY FUNCTIONS/CLASSES
def get_trunk(trunk_name):
    """Retrieve the pretrained network trunk and channel counts"""
    if trunk_name == 'mobilenetv3_large':
        backbone = MobileNetV3_Large(pretrained=False)
        s2_ch = 16
        s4_ch = 24
        high_level_ch = 960
    elif trunk_name=='mobilenetv3_small':
        backbone = MobileNetV3_Small(pretrained=False)
        s2_ch = 16
        s4_ch = 16
        high_level_ch = 576
    else:
        raise ValueError('unknown backbone {}'.format(trunk_name))
    return backbone, s2_ch, s4_ch, high_level_ch

class ConvBnRelu(nn.Module):
    """Convenience layer combining a Conv2d, BatchNorm2d, and a ReLU activation.
    Original source of this code comes from
    https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#ENCODERS
class MobileNetV3_Large_Light(nn.Module):
    """Modified MobileNetV3 for use as semantic segmentation feature extractors."""
    def __init__(self, opt, trunk=tf_mobilenetv3_large_100, pretrained=False):
        super(MobileNetV3_Large_Light, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[3][0].conv_dw.stride = (1, 1)
        net.blocks[5][0].conv_dw.stride = (1, 1)

        for block_num in (3, 4, 5, 6):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 5:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(m, Conv2dSameExport):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]

    def forward(self, x, input_dict_extra=None):
        x = self.early(x) # 2x
        x = self.block0(x)
        s2 = x
        x = self.block1(x) # 4x
        s4 = x
        x = self.block2(x) # 8x
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return s2, s4, x, {}

class MobileNetV3_Small_Light(nn.Module):
    def __init__(self, opt, in_channels, trunk=tf_mobilenetv3_small_100, pretrained=False):
        super(MobileNetV3_Small_Light, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.earlyconv = nn.Conv2d(in_channels, 16, 1, 2)
        self.earlybn = nn.BatchNorm2d(16)
        self.earlyact = HardSwish()

        self.early = nn.Sequential(self.earlyconv, self.earlybn, self.earlyact) #nn.Sequential(net.conv_stem, net.bn1, net.act1)
        

        net.blocks[2][0].conv_dw.stride = (1, 1)
        net.blocks[4][0].conv_dw.stride = (1, 1)

        for block_num in (2, 3, 4, 5):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 4:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(m, Conv2dSameExport):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]

    def forward(self, x, input_dict_extra=None):
        x = self.early(x) # 2x
        s2 = x
        x = self.block0(x) # 4x
        s4 = x
        x = self.block1(x) # 8x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return s2, s4, x, {}

#DECODERS
class LRASPP_Light(nn.Module):
    """Lite R-ASPP style segmentation network."""
    def __init__(self, opt, SGNum, mode=0, use_aspp=False, num_filters=128):
        """Initialize a new segmentation model.
        Keyword arguments:
        use_aspp -- whether to use DeepLabV3+ style ASPP (True) or Lite R-ASPP (False)
            (setting this to True may yield better results, at the cost of latency)
        num_filters -- the number of filters in the segmentation head
        """
        super(LRASPP_Light, self).__init__()

        #self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk)
        self.use_aspp = use_aspp
        self.mobilenet_size_large = False
        self.mode = mode
        self.SGNum = SGNum

        out_channel_final = 0
        if (mode == 0):
            out_channel_final = 3 * self.SGNum
        elif (mode == 2):
            out_channel_final = 3 * self.SGNum
        elif (mode == 1):
            out_channel_final = self.SGNum
        else:
            out_channel_final = 3 * self.SGNum

        self.out_channel_final = out_channel_final

        if (self.mobilenet_size_large):
            s2_ch = 16
            s4_ch = 24
            high_level_ch = 960
        else:
            s2_ch = 16
            s4_ch = 16
            high_level_ch = 576

        # Reduced atrous spatial pyramid pooling
        if self.use_aspp:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=12, padding=12),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv3 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=36, padding=36),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            aspp_out_ch = num_filters * 4
        else:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=(10,10)), #changed -- probably change back
                #nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Sigmoid(),
            )
            aspp_out_ch = num_filters
        

        num_layers2 = int(aspp_out_ch/2)
        num_layers3 = int(aspp_out_ch/4)


        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
        self.conv_up2 = ConvBnRelu(aspp_out_ch + 64, num_filters, kernel_size=1)
        self.conv_up3 = ConvBnRelu(aspp_out_ch + 32, num_filters,  kernel_size=1)
        self.dpadFinal = nn.ReplicationPad2d(1)
        self.last = nn.Conv2d(num_filters, self.out_channel_final, kernel_size=1)

    def forward(self, s2, s4, final, im, input_dict_extra=None):

        extra_output_dict = {}

        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                F.interpolate(self.aspp_pool(final), size=final.shape[2:]),
            ], 1)
        else:
            aspp = self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)
        dx1 = y

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)
        dx2 = y

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = F.interpolate(y, size=im.shape[2:], mode='bilinear', align_corners=False)
        dx3 = y


        y = self.last(self.dpadFinal(dx3))
        

        x_orig = y
        x_out = y

        return_dict = {'extra_output_dict': extra_output_dict, 'dx1': dx1, 'dx2': dx2, 'dx3': dx3}
    
        if self.mode == 1 or self.mode == 2:
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, 0, 1)
        elif self.mode == 0:
            bn, _, row, col = x_out.size()
            x_out = x_out.view(bn, self.SGNum, 3, row, col)
            x_out = x_out / torch.clamp(torch.sqrt(torch.sum(x_out * x_out, dim=2).unsqueeze(2) ), min = 1e-6).expand_as(x_out )
        else:
            x_out = x_orig

        #return_dict.update({'x_out': x_out})

        return x_out

class output2env():
    def __init__(self, SGNum, envWidth = 16, envHeight = 8, isCuda = True ):
        self.envWidth = envWidth
        self.envHeight = envHeight

        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi
        El = ( (np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 0)
        ls = ls[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :, :]
        self.ls = Variable(torch.from_numpy(ls.astype(np.float32 ) ) )

        self.SGNum = SGNum
        if isCuda:
            self.ls = self.ls.cuda()

        self.ls.requires_grad = False

    def fromSGtoIm(self, axis, lamb, weight ):
        torch.cuda.empty_cache()
        bn = axis.size(0)
        envRow, envCol = weight.size(2), weight.size(3)

        # Turn SG parameters to environmental maps
        axis = axis.unsqueeze(-1).unsqueeze(-1)

        weight = weight.view(bn, self.SGNum, 3, envRow, envCol, 1, 1)
        lamb = lamb.view(bn, self.SGNum, 1, envRow, envCol, 1, 1)

        mi = lamb.expand([bn, self.SGNum, 1, envRow, envCol, self.envHeight, self.envWidth] )* \
                (torch.sum(axis.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]) * \
                self.ls.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] ), dim = 2).unsqueeze(2) - 1)
        envmaps = weight.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] ) * \
            torch.exp(mi).expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] )
        # print(envmaps.shape)

        envmaps = torch.sum(envmaps, dim=1)

        return envmaps

    def output2env(self, axisOrig, lambOrig, weightOrig, if_postprocessing=True):
        bn, _, envRow, envCol = weightOrig.size()

        axis = axisOrig # torch.Size([B, 12(SGNum), 3, 120, 160])
        
        if if_postprocessing:
            weight = 0.999 * weightOrig
            # weight = 0.8 * weightOrig 
            weight = torch.tan(np.pi / 2 * weight )
        else:
            weight = weightOrig

        if if_postprocessing:
            lambOrig = 0.999 * lambOrig
            lamb = torch.tan(np.pi / 2 * lambOrig )
        else:
            lamb = lambOrig


        envmaps = self.fromSGtoIm(axis, lamb, weight )

        return envmaps, axis, lamb, weight

class renderingLayer():
    def __init__(self, imWidth = 160, imHeight = 120, fov=57, F0=0.05, cameraPos = [0, 0, 0], 
            envWidth = 16, envHeight = 8, isCuda = True):
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envWidth = envWidth
        self.envHeight = envHeight

        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        v = v.astype(dtype = np.float32)

        self.v = Variable(torch.from_numpy(v) ) # for rendering only: virtual camera plane 3D coords (x-y-z with -z forward)
        self.pCoord = Variable(torch.from_numpy(self.pCoord) )

        self.up = torch.Tensor([0,1,0] )

        # azimuth & elevation angles -> dir vector for each pixel
        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi # array([-2.9452, -2.5525, -2.1598, -1.7671, -1.3744, -0.9817, -0.589, -0.1963,  0.1963,  0.589 ,  0.9817,  1.3744,  1.7671,  2.1598,  2.5525,  2.9452])
        El = ( (np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0 # array([0.0982, 0.2945, 0.4909, 0.6872, 0.8836, 1.0799, 1.2763, 1.4726])
        Az, El = np.meshgrid(Az, El)
        Az = Az.reshape(-1, 1)
        El = El.reshape(-1, 1)
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 1) # dir vector for each pixel (local coords)

        envWeight = np.sin(El ) * np.pi * np.pi / envWidth / envHeight

        self.ls = Variable(torch.from_numpy(ls.astype(np.float32 ) ) )
        self.envWeight = Variable(torch.from_numpy(envWeight.astype(np.float32 ) ) )
        self.envWeight = self.envWeight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        if isCuda:
            self.v = self.v.cuda()
            self.pCoord = self.pCoord.cuda()
            self.up = self.up.cuda()
            self.ls = self.ls.cuda()
            self.envWeight = self.envWeight.cuda()

    def forwardEnv(self, normalPred, envmap=None, diffusePred=None, roughPred=None, if_normal_only=False):
        if envmap is not None:
            envR, envC = envmap.size(2), envmap.size(3)
        else:
            envR, envC = self.imHeight, self.imWidth

        # print(normalPred.shape, diffusePred.shape, roughPred.shape)
        if diffusePred is not None and roughPred is not None:
            assert normalPred.shape[-2:] == diffusePred.shape[-2:] == roughPred.shape[-2:]
        
        # print(normalPred.shape)
        normalPred = F.adaptive_avg_pool2d(normalPred, (envR, envC) )
        normalPred = normalPred / torch.sqrt( torch.clamp(
            torch.sum(normalPred * normalPred, dim=1 ), 1e-6, 1).unsqueeze(1) )

        # assert normalPred.shape[2:]==(self.imHeight, self.imWidth)

        ldirections = self.ls.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # torch.Size([1, 128, 3, 1, 1])

        # print(if_normal_only, self.up.shape, normalPred.shape)
        camyProj = torch.einsum('b,abcd->acd',(self.up, normalPred)).unsqueeze(1).expand_as(normalPred) * normalPred # project camera up to normalPred direction https://en.wikipedia.org/wiki/Vector_projection
        camy = F.normalize(self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj) - camyProj, dim=1, p=2)
        camx = -F.normalize(torch.cross(camy, normalPred,dim=1), p=2, dim=1) # torch.Size([1, 3, 120, 160])

        # print(camx.shape, torch.linalg.norm(camx, dim=1))
        # print(camy.shape, torch.linalg.norm(camy, dim=1))
        # ls (local coords), l (cam coords)
        # \sum [1, 128, 1, 1, 1] * [1, 1, 3, 120, 160] -> [1, 128, 3, 120, 160]
        # single vec: \sum [1, 1, 1, 1, 1] * [1, 1, 3, 1, 1] -> [1, 1, 3, 1, 1]
        
        # print(ldirections[:, :, 0:1, :, :].shape, camx.unsqueeze(1).shape) # torch.Size([1, 128, 1, 1, 1]) torch.Size([2, 1, 3, 120, 160])
        
        # [!!!] multiply the local SG self.ls grid vectors (think of as coefficients) with the LOCAL camera-dependent basis (think of as basis..)
        # ... and then you arrive at a hemisphere in the camera cooords
        l = ldirections[:, :, 0:1, :, :] * camx.unsqueeze(1) \
                + ldirections[:, :, 1:2, :, :] * camy.unsqueeze(1) \
                + ldirections[:, :, 2:3, :, :] * normalPred.unsqueeze(1)    
        # print(l.shape) # torch.Size([1, 128, 3, 120, 160])
        # print(ldirections[:, 20, :, :, :].flatten())

        if if_normal_only:
            return l, camx, camy, normalPred

        bn = diffusePred.size(0)
        diffusePred = F.adaptive_avg_pool2d(diffusePred, (envR, envC) )
        roughPred = F.adaptive_avg_pool2d(roughPred, (envR, envC ) )

        temp = Variable(torch.FloatTensor(1, 1, 1, 1,1) )


        if self.isCuda:
            temp = temp.cuda()

        h = (self.v.unsqueeze(1) + l) / 2;
        h = h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=2), min = 1e-6).unsqueeze(2) )

        vdh = torch.sum( (self.v * h), dim = 2).unsqueeze(2)
        temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        diffuseBatch = (diffusePred )/ np.pi
        roughBatch = (roughPred + 1.0)/2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normalPred * self.v.expand_as(normalPred), dim = 1), 0, 1).unsqueeze(1).unsqueeze(2)
        ndh = torch.clamp(torch.sum(normalPred.unsqueeze(1) * h, dim = 2), 0, 1).unsqueeze(2)
        ndl = torch.clamp(torch.sum(normalPred.unsqueeze(1) * l, dim = 2), 0, 1).unsqueeze(2) # [!!!] cos in rendering function; normalPred and l are both in camera coords

        frac = alpha2.unsqueeze(1).expand_as(frac0) * frac0
        nom0 = ndh * ndh * (alpha2.unsqueeze(1).expand_as(ndh) - 1) + 1
        nom1 = ndv * (1 - k.unsqueeze(1).expand_as(ndh) ) + k.unsqueeze(1).expand_as(ndh)
        nom2 = ndl * (1 - k.unsqueeze(1).expand_as(ndh) ) + k.unsqueeze(1).expand_as(ndh)
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom # f_s

        envmap = envmap.view([bn, 3, envR, envC, self.envWidth * self.envHeight ] )
        envmap = envmap.permute([0, 4, 1, 2, 3] )

        brdfDiffuse = diffuseBatch.unsqueeze(1).expand([bn, self.envWidth * self.envHeight, 3, envR, envC] ) * \
                    ndl.expand([bn, self.envWidth * self.envHeight, 3, envR, envC] )
        colorDiffuse = torch.sum(brdfDiffuse * envmap * self.envWeight.expand_as(brdfDiffuse), dim=1) # I_d

        brdfSpec = specPred.expand([bn, self.envWidth * self.envHeight, 3, envR, envC ] ) * \
                    ndl.expand([bn, self.envWidth * self.envHeight, 3, envR, envC] )
        colorSpec = torch.sum(brdfSpec * envmap * self.envWeight.expand_as(brdfSpec), dim=1) # I_s

        return colorDiffuse, colorSpec