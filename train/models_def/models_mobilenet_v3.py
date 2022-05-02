"""Modified MobileNetV3 for use as semantic segmentation feature extractors."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geffnet import tf_mobilenetv3_large_100, tf_mobilenetv3_small_100
from geffnet.efficientnet_builder import InvertedResidual, Conv2dSame, Conv2dSameExport

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
class MobileNetV3_Large(nn.Module):
    """Modified MobileNetV3 for use as semantic segmentation feature extractors."""
    def __init__(self, opt, trunk=tf_mobilenetv3_large_100, pretrained=False):
        super(MobileNetV3_Large, self).__init__()
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

class MobileNetV3_Small(nn.Module):
    def __init__(self, opt, trunk=tf_mobilenetv3_small_100, pretrained=False):
        super(MobileNetV3_Small, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

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
        print("\n\n\n INPUT SIZE: ")
        print(x.size())

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
class LRASPP(nn.Module):
    """Lite R-ASPP style segmentation network."""
    def __init__(self, opt, mode=-1, use_aspp=False, num_filters=128):
        """Initialize a new segmentation model.
        Keyword arguments:
        use_aspp -- whether to use DeepLabV3+ style ASPP (True) or Lite R-ASPP (False)
            (setting this to True may yield better results, at the cost of latency)
        num_filters -- the number of filters in the segmentation head
        """
        super(LRASPP, self).__init__()

        #self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk)
        self.use_aspp = use_aspp
        self.mobilenet_size_large = False
        self.mode = mode

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
                nn.AvgPool2d(kernel_size=(10,10)),
                #nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Sigmoid(),
            )
            aspp_out_ch = num_filters

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
        self.conv_up2 = ConvBnRelu(num_filters + 64, num_filters, kernel_size=1)
        self.conv_up3 = ConvBnRelu(num_filters + 32, num_filters, kernel_size=1)
        
        #self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, s2, s4, final, im, input_dict_extra=None):

        print("\n\n\nINITIAL:")
        print(final.size())

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
        
        print("\n\n\nAFTER ASPP:")
        print(aspp.size())

        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)
        dx1 = y

        print("\n\n\nCONVUP1:")
        print(y.size())

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)
        dx2 = y

        print("\n\n\nCONVUP2")
        print(final.size())

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = F.interpolate(y, size=im.shape[2:], mode='bilinear', align_corners=False)
        dx3 = y

        print("\n\n\nCONVUP3")
        print(final.size())
        
        x_orig = y

        return_dict = {'extra_output_dict': extra_output_dict, 'dx1': dx1, 'dx2': dx2, 'dx3': dx3}
        

        print("\n\n\nMODE: ")
        print(self.mode)
        print("\n\n\n\n")
        if self.mode == 0: # modality='al'
            x_out = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
        elif self.mode == 1: # modality='no'
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1) ).expand_as(x_orig)
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2: # modality='ro'
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = F.softmax(x_orig, dim=1)
        elif self.mode == 4: # modality='de'
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1) # -> [-1, 1]
        # elif self.mode == 5: # clip to 0., inf
        #     x_out = self.relu(torch.mean(x_orig, dim=1).unsqueeze(1)) # -> [0, inf]
        #     # x_out[x_out < 1e-8] = 1e-8
        # elif self.mode == 6: # sigmoid to 0., 1. -> inverse to 0., inf
        #     x_out = torch.sigmoid(torch.mean(x_orig, dim=1).unsqueeze(1))
        #     x_out = 1. / (x_out + 1e-6) # -> [0, inf]
        else:
            x_out = x_orig
        
        y = x_out
        y = F.interpolate(y, size=x_out.shape[2:], mode='bilinear', align_corners=False)
        x_out = y

        print("\n\n\nlast layer:")
        print(x_out.size())
        print("\n\n\n")

        return_dict.update({'x_out': x_out})

        return return_dict


#COMPLETED EXAMPLE WITH SEGMENTATION
'''
class MobileV3Large(LRASPP):
    """MobileNetV3-Large segmentation network."""
    model_name = 'mobilev3large-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Large, self).__init__(
            num_classes,
            trunk='mobilenetv3_large',
            **kwargs
        )


class MobileV3Small(LRASPP):
    """MobileNetV3-Small segmentation network."""
    model_name = 'mobilev3small-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Small, self).__init__(
            num_classes,
            trunk='mobilenetv3_small',
            **kwargs
        )
'''