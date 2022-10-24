import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import math

STAGES = [32, 16, 24, 32, 64, 96, 160, 320] # c
LAST_CHANNEL = 1280
REPEATS = [1, 1, 2, 3, 4, 3, 3, 1] # n
STRIDES = [2, 1, 2, 2, 2, 1, 2, 1] # s
EXPANDS = [1, 1, 6, 6, 6, 6, 6, 6]
inverted_residual_setting = [
                # t, c, n, s
                [1, 32, 1, 2],
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.relu1 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

class lastconv1x1(nn.Module):
    def __init__(self, idx, inp, oup, stride, resolution=None, n_lv=None, l2_vals=None, bit=None):
        super(lastconv1x1, self).__init__()
        self.index = idx
        self.n_lv = n_lv

        self.bn_qconv_bn_relu_0 = nn.Sequential(
            nn.BatchNorm2d(inp),
            ActLsqQuan(bit, in_planes=inp),
            QtConv(inp, oup, kernel_size=1, stride=stride, padding=0, groups=1, w_bit=bit, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=False),
        )
        self.skip_0 = FlexibleSkipConn(inp, oup)
        self.bn0_last = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = self.bn_qconv_bn_relu_0(x)
        out += self.skip_0(x)
        out = self.bn0_last(out)
        return out

class QuantNbit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clamp, bin):
        ctx.save_for_backward(input, clamp)
        if clamp.data <= 0:
            input = torch.clamp(input, 0., 0.)
        else:
            input = torch.clamp(input, 0., clamp.data)
            
        if bin >= 1:
            delta = clamp.data
        elif bin <= 1/255:
            delta = clamp.data / 255
        else:
            delta = clamp.data * bin
        output = torch.round(input/delta)*delta
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clamp, = ctx.saved_tensors
        
        grad_input = grad_output.clone()
        grad_input[input<=0] = 0
        grad_input[input>=clamp] = 0
        
        grad_clamp = grad_output.clone()
        grad_clamp[input<=clamp] = 0

        return grad_input, grad_clamp, None

class QtActivation(nn.Module):
    def __init__(self, idx, subidx, in_channels, resolution, n_lv=None, upper_bound=6.0):
        super(QtActivation, self).__init__()
        self.index = idx
        self.sub_index = subidx
        self.n_lv = n_lv
        self.bin = nn.Parameter(torch.ones(1)*(1 - 1/(n_lv-1)))

        # l2 branch
        self.upper_bound = nn.Parameter(torch.tensor(upper_bound))
        self.in_channels = in_channels
        self.resolution = resolution
        self.input_tensor_size = resolution * resolution * in_channels
        self.a_bin = True

    def forward(self, input):
        if self.a_bin:
            return QuantNbit.apply(input, self.upper_bound, 1-self.bin)
        else:
            return input

    def extra_repr(self):
        s = ('index={index}, sub_index={sub_index}, in_channels={in_channels}, a_bin={a_bin}')
        return s.format(**self.__dict__)

    def change_precision(self, a_bin=False):
        self.a_bin = a_bin

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class WLsqQuan(nn.Module):
    def __init__(self, bit, in_planes=None, symmetric=True, per_kernel=True):
        super(WLsqQuan, self).__init__()

        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1

        self.in_planes = in_planes
        self.per_kernel = per_kernel
        self.s = nn.Parameter(torch.ones(in_planes))

    def forward(self, x):
        g = 1.0 / ((x.numel() * self.thd_pos)**0.5)
        alpha = grad_scale(self.s, g)
        x = round_pass((x / alpha.view(-1,1,1,1)).clamp(self.thd_neg, self.thd_pos)) * alpha.view(-1,1,1,1)

        return x

    def init_from(self, x, *args, **kwargs):
        if self.per_kernel:
            self.s.data.copy_(2 * x.detach().view(self.in_planes,-1).abs().mean(dim=1) / self.thd_pos ** 0.5)
        else:
            self.s.data.copy_(2 * x.detach().abs().mean() / self.thd_pos ** 0.5)
        print(f"initw")

class ActLsqQuan(nn.Module):
    def __init__(self, bit, in_planes=None, per_channel=True):
        super(ActLsqQuan, self).__init__()

        self.thd_neg = 0
        self.thd_pos = 2 ** bit - 1
        self.in_planes = in_planes
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(in_planes))

    def forward(self, x):
        g = 1.0 / ((x.numel() * self.thd_pos)**0.5)
        alpha = grad_scale(self.s, g)
        x = round_pass((x / alpha.view(1,-1,1,1)).clamp(self.thd_neg, self.thd_pos)) * alpha.view(1,-1,1,1)

        return x

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s.data.copy_(2 * x.detach().permute(1,0,2,3).contiguous().view(self.in_planes,-1).abs().mean(dim=1) / self.thd_pos ** 0.5)
        else:
            self.s.data.copy_(2 * x.detach().abs().mean() / self.thd_pos ** 0.5)
        print(f"initact")

class QtConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, w_bit=1, w_bin=False, bias=False):
        super(QtConv, self).__init__()
        self.in_channels = in_chn
        self.out_channels = out_chn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.w_bin = w_bin
        self.bias = bias
        self.number_of_weights = (in_chn // groups) * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, (in_chn // groups), kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        torch.nn.init.uniform_(self.weights, -0.01, 0.01)

        self.w_quantize = WLsqQuan(w_bit, in_planes=out_chn)
        self.w_quantize.init_from(self.weights.view(self.shape))

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        if self.w_bin:
            qt_weights = self.w_quantize(real_weights)
        else:
            qt_weights = real_weights

        y = F.conv2d(x, qt_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
            ', stride={stride}, padding={padding}, dilation={dilation}, groups={groups}, w_bin={w_bin}, bias={bias}')
        return s.format(**self.__dict__)

    def reinit_conv(self):
        self.w_quantize.init_from(self.weights.view(self.shape))

    def change_precision(self, w_bin=False):
        self.w_bin = w_bin
        
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class FlexibleSkipConn(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(FlexibleSkipConn, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x):
        if self.in_planes == self.out_planes:
            out = x
        elif self.in_planes < self.out_planes:
            expansion_ratio = self.out_planes // self.in_planes
            out = torch.cat([x for _ in range(expansion_ratio)], dim=1)
        else: # self.in_planes > self.out_planes
            contraction_ratio = self.in_planes // self.out_planes
            factorized_tensor = [x[:,self.out_planes*i:self.out_planes*(i+1)] for i in range(contraction_ratio)]
            
            # unfit case
            if self.in_planes % self.out_planes != 0:
                remainings = self.in_planes % self.out_planes
                zeros = torch.zeros_like(x[:,:(self.out_planes-remainings)])
                factorized_tensor += [torch.cat([x[:,self.out_planes*contraction_ratio:], zeros], dim=1)]
            out = sum(factorized_tensor)
        
        if self.stride != 1:
            out = F.avg_pool2d(out, self.stride)
        
        return out

    def extra_repr(self):
        s = ('{in_planes}, {out_planes}, stride={stride}')            
        return s.format(**self.__dict__)

class InvertedResidual(nn.Module):
    def __init__(self, idx, in_planes, out_planes, stride=1, expand_ratio=1, resolution=None, n_lv=None, l2_vals=list(), bit=None):
        super(InvertedResidual, self).__init__()
        self.index = idx
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.n_lv = n_lv
        self.l2_vals = l2_vals

        hidden_planes = int(round(in_planes * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        if self.expand_ratio != 1:
            self.bn_qconv_bn_relu_0 = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                ActLsqQuan(bit, in_planes=in_planes),
                QtConv(in_planes, hidden_planes, kernel_size=1, stride=1, padding=0, groups=1, w_bit=bit, bias=False),
                nn.BatchNorm2d(hidden_planes),
                nn.ReLU(inplace=False),
            )
            self.skip_0 = FlexibleSkipConn(in_planes, hidden_planes)
            self.bn0_last = nn.BatchNorm2d(hidden_planes)

        self.branch_bn_q_1 = nn.ModuleList()
        self.conv1 = QtConv(hidden_planes, hidden_planes, kernel_size=3, stride=stride, padding=1, groups=hidden_planes, w_bit=bit, bias=False)
        self.branch_bn_1 = nn.ModuleList()
        self.skip_1 = nn.Sequential()
        self.bn1_last = nn.ModuleList()
        if stride != 1:
            self.skip_1 = nn.Sequential(
                nn.AvgPool2d(2, 2),
            )
        for i in range(len(l2_vals)):
            self.branch_bn_q_1.append(
                nn.Sequential(
                    nn.BatchNorm2d(hidden_planes),
                    QtActivation(idx, 1, hidden_planes, resolution, n_lv=n_lv),
                )
            )
            self.branch_bn_1.append(
                nn.Sequential(
                    nn.BatchNorm2d(hidden_planes),
                    nn.ReLU(inplace=False)
                )
            )
            self.bn1_last.append(
                nn.BatchNorm2d(hidden_planes)
            )

        resolution //= stride

        self.bn_qconv_bn_relu_2 = nn.Sequential(
            nn.BatchNorm2d(hidden_planes),
            ActLsqQuan(bit, in_planes=hidden_planes),
            QtConv(hidden_planes, out_planes, kernel_size=1, stride=1, padding=0, w_bit=bit, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False)
        )
        self.skip_2 = FlexibleSkipConn(hidden_planes, out_planes)
        self.bn2_last = nn.BatchNorm2d(out_planes)

    def forward(self, x, selection):
        if self.expand_ratio != 1:
            out0 = self.bn_qconv_bn_relu_0(x)
            out0 += self.skip_0(x)
            out0 = self.bn0_last(out0)
        else:
            out0 = x

        out1 = self.branch_bn_q_1[selection](out0)
        out1 = self.conv1(out1)
        out1 = self.branch_bn_1[selection](out1)
        out1 += self.skip_1(out0)
        out1 = self.bn1_last[selection](out1)

        out2 = self.bn_qconv_bn_relu_2(out1)
        out2 += self.skip_2(out1)
        out2 = self.bn2_last(out2)

        if self.use_res_connect:
            out2 = out2 + x

        return out2

def make_divisible(v: int, divisor: int = 8, min_value: int = None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # ensure round down does not go down by more than 10%.
        new_v += divisor
    return new_v

def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)

def _scale_stage_depth(depth_multiplier, repeats, depth_trunc='ceil'):    
    stage_repeats = []
    for r in repeats:
        if depth_trunc == 'round':
            stage_repeats.append(max(1, round(r * depth_multiplier)))
        else:
            stage_repeats.append(int(math.ceil(r * depth_multiplier)))
    return stage_repeats

def generate_efficient_mbv2(num_classes=10, channel_multiplier=1.0, channel_divisor=8, depth_multiplier=1.0, resolution_multiplier=1.0, base_resolution=32, n_lv=None, l2_vals=list(), bit=None):
    stage_repeats = _scale_stage_depth(depth_multiplier, REPEATS)
    stage_out_channel = []
    refined_stage_out_channel = []
    stage_strides = []
    expand_ratios = []
    for i, (t,c,n,s) in enumerate(inverted_residual_setting):
        for j in range(n):
            stage_out_channel.append(round_channels(c, channel_multiplier, channel_divisor))
            if j==0: stage_strides.append(s)
            else: stage_strides.append(1)
            refined_stage_out_channel.append(stage_out_channel[-1])
            expand_ratios.append(t)
    stage_resolution = int(resolution_multiplier * base_resolution)
    print(f"==Network Info==")
    print(f"baseline_channels: {STAGES}")
    print(f"stage_resolution: {stage_resolution}")
    print(f"stage_repeats: {stage_repeats}")
    print(f"stage_out_channel: {stage_out_channel}")
    print(f"refined_stage_out_channel: {refined_stage_out_channel}")
    print(f"stage_strides: {stage_strides}")
    return mbv2(stage_repeats, refined_stage_out_channel, stage_strides, expand_ratios, num_classes, channel_multiplier, channel_divisor, depth_multiplier, stage_resolution, n_lv=n_lv, l2_vals=l2_vals, bit=bit), stage_resolution


class mbv2(nn.Module):
    def __init__(self, stage_repeats, stage_out_channel, stage_strides, expand_ratios, num_classes=10, channel_multiplier=1.0, channel_divisor=8, depth_multiplier=1.0, resolution=32, n_lv=None, l2_vals=list(), bit=None):
        super(mbv2, self).__init__()
        self.feature = nn.ModuleList()
        self.resolution = resolution
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], stage_strides[i]))
            else:
                self.feature.append(InvertedResidual(i, stage_out_channel[i-1], stage_out_channel[i], stage_strides[i], expand_ratios[i], self.resolution, n_lv=n_lv, l2_vals=l2_vals, bit=bit))
                self.resolution //= stage_strides[i]
        
        self.feature.append(lastconv1x1(len(stage_out_channel), stage_out_channel[-1], LAST_CHANNEL, 1, resolution=self.resolution, n_lv=n_lv, l2_vals=l2_vals, bit=bit))
                
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(LAST_CHANNEL, num_classes)
        )

    def load_model(self, model_path):
        chkpt = torch.load(model_path)
        self.load_state_dict(chkpt['model_state_dict'], False)
        print(f"model loaded: {model_path}, best_acc={chkpt['best_top1_acc']:.3f}")

    def forward(self, x, selections):
        for i, block in enumerate(self.feature):
            if i==0:
                o = block(x)
            elif i==len(self.feature)-1:
                o = block(o)
            else:
                o = block(o, selections[i-1])

        out_o = self.pool1(o).view(o.size(0), -1)
        out_o = self.classifier(out_o)

        return out_o

    def change_precision(self, a_bin=False, w_bin=False):
        for m in self.modules():
            if isinstance(m, QtActivation):
                m.change_precision(a_bin=a_bin)
            if isinstance(m, QtConv):
                m.change_precision(w_bin=w_bin)

    def reinit_conv(self):
        print(f"reinitializing convolution")
        for m in self.modules():
            if isinstance(m, QtConv):
                m.reinit_conv()