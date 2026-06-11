import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv2d(nn.Module):
    """
    Depthwise convolution wrapper using nn.Conv2d with groups=in_channels.
    Each input channel is convolved independently with its own 3x3 (or kernel_size x kernel_size) kernel.
    Input:  x of shape (N, C, H, W)
    Output: y of shape (N, C, H_out, W_out)
    """
    def __init__(
        self,
        in_channels: int,          # number of input channels (C)
        out_channels: int,         # number of output channels
        kernel_size: int = 3,      # kernel spatial size (kxk). default 3 -> 3x3 kernels.
        stride: int = 1,           # stride of the convolution. stride>1 downsamples spatial dims.
        padding: int | None = None,# zeros added around input. None -> automatic "same" padding for odd kernels.
        dilation: int = 1,         # spacing between kernel elements (dilated conv).
        bias: bool = True,         # whether conv has a learnable bias per output channel.
        activation: str = 'leaky_relu' # activation function to apply after conv (if any)
    ):
        super().__init__()
        if padding is None and kernel_size % 2 == 1 and stride == 1:
            #padding is applied to keep the  spatial size of the transformation same to that of the input.
            padding = dilation*(kernel_size  - 1) //2 # this relationship holds only for stride=1 and odd kernel sizes
        elif kernel_size % 2 != 1 or stride != 1:
            raise ValueError("kernel_size must be odd and stride must be 1 to keep data flow grounded to the SR network architecture.")

        # nn.Conv2d arguments explained:
        # - in_channels: number of input channels
        # - out_channels: number of output channels. For depthwise conv we set it equal to in_channels
        # - kernel_size: spatial kernel size (int or tuple)
        # - stride: movement step of the kernel
        # - padding: zero-padding added to both sides of input
        # - dilation: dilation rate for dilated conv
        # - groups: when groups == in_channels and out_channels == in_channels, each input channel is convolved
        #           with its own filter (depthwise).
        # - bias: include per-channel learnable bias if True
        self.z = nn.Conv2d(
            in_channels,          # input channels
            out_channels,          # output channels == input channels for depthwise
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,   # key: makes the conv depthwise(This means each input channel is convolved with its own filter)
            bias=bias
        )
        if activation is None:
            self.act = None
        elif activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True,negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        - x: tensor shaped (N, C, H, W)
        - returns: tensor shaped (N, C, H_out, W_out)
        PyTorch autograd will compute gradients for self.dw parameters automatically during loss.backward().
        """
        x = self.z(x)
        return x if self.act is None else self.act(x)
    

class ConvBlock(nn.Module):
    """
    Simple convolutional block: Conv2d 
    Input:  x of shape (N, C, H, H)
    Output: y of shape (N, C, H_out, H_out)
    Here we have ensure that H_out = H , to keep this module grounded to our SR model architecture.
    """
    def __init__(
        self,
        in_channels: int,          # number of input channels (C)
        out_channels: int,         # number of output channels
        kernel_size: int = 3,      # kernel spatial size (kxk). default 3 -> 3x3 kernels.
        stride: int = 1,           # stride of the convolution. stride>1 downsamples spatial dims.
        padding: int | None = None,# zeros added around input. None -> automatic "same" padding for odd kernels.
        dilation: int = 1,         # spacing between kernel elements (dilated conv).
        bias: bool = True,         # whether conv has a learnable bias per output channel.
        activation: str = None     # activation function to apply after conv (if any)
    ):
        super().__init__()
        if padding is None and kernel_size % 2 == 1 and stride == 1:
            #padding here is primarly applied to keep the  spatial size of the transformation same to that of the input.
            padding = dilation*(kernel_size  - 1) //2 # this relationship holds only for stride=1 and odd kernel sizes
        elif kernel_size % 2 != 1 or stride != 1:
            raise ValueError("kernel_size must be odd and stride must be 1 to keep data flow grounded to the SR network architecture.")
        
        self.z = nn.Conv2d(
            in_channels,          # input channels
            out_channels,         # output channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1, # standard conv (not depthwise)
            bias=bias
        )

        if activation is None:
            self.act = None
        elif activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True,negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.z(x)
        return x if self.act is None else self.act(x)
        


class Depthwise_residual_block(nn.Module):
    """
    Depthwise separable convolutional block with residual connection.
    input:  x of shape (H, H,channels) -> DepthwiseConv2d with Leaky relu -> ConvBlock
    Applied some constraints to keep it grounded to our SR model architecture.
     - stride=1 (no downsampling)
     - inchannels = out_channels
     - kernel_size odd (3,5,7,...)
    """
    def __init__(
            self,
            kernel_size: int = 3,
            dilation: int = 1,
            in_channels: int = 16,
            output_channel: int =16,
            activation: str = 'leaky_relu'
    ):
        super().__init__()
        self.z1 = DepthwiseConv2d(in_channels=in_channels,out_channels=output_channel ,kernel_size=kernel_size, stride=1, padding=None, dilation=dilation, activation=activation)
        self.z2 = ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size = kernel_size, stride=1, padding=None, dilation=dilation, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.z1(x)  # depthwise conv + LeakyReLU
        x = self.z2(x)  # pointwise conv (1x1)
        x += identity  # residual connection
        return x

class DeptwiseResidualGroup(nn.Module):
    def __init__(
            self,
            in_channels : int = 16,
            out_channels : int = 16,
            kernel_size : int=3,
            dilation : int = 1,
            activation : str = None
    ):
        super().__init__()
        self.NumberOfDWRB = 10
        self.DWRGs = nn.ModuleList([Depthwise_residual_block(in_channels=in_channels,output_channel=out_channels ,kernel_size=kernel_size,dilation=dilation) for _ in range(self.NumberOfDWRB)])
        self.z2 = ConvBlock(in_channels=in_channels ,out_channels=out_channels , kernel_size=kernel_size,dilation=dilation , activation=activation)
    def forward(self ,x:torch.Tensor) ->torch.Tensor:
        identity = x
        for DWRG in self.DWRGs:
            x = DWRG(x)
        x = self.z2(x)
        x =identity + x #establishing residual connection /skip connection
        return x
    
    
class ResidualBlock(nn.Module):
    """
    Standard convolutional block with residual connection.
    input:  x of shape (H, H,channels) -> ConvBlock with Leaky relu -> ConvBlock
    Applied some constraints to keep it grounded to our SR model architecture.
     - stride=1 (no downsampling)
     - inchannels = out_channels
     - kernel_size odd (3,5,7,...)
    """
    def __init__(
            self,
            in_channels: int = 16,
            out_channels: int = 16,
            kernel_size: int = 3,
            dilation: int = 1,
            activation: str = 'leaky_relu',
    ):
        super().__init__()
        self.z1 = ConvBlock(in_channels=in_channels,out_channels=out_channels ,kernel_size=kernel_size, stride=1, padding=None, dilation=dilation, activation=activation)
        self.z2 = ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size = kernel_size, stride=1, padding=None, dilation=dilation, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.z1(x)  # normal conv + LeakyReLU
        x = self.z2(x) 
        x += identity  # residual connection
        return x

class ResidualGroup(nn.Module):
    def __init__(
            self,
            in_channels: int = 16,
            out_channels: int = 16,
            kernel_size: int = 3,
            dilation: int = 1,
            activation: str = 'leaky_relu',
    ):
        super().__init__()
        self.NumberOfRB = 10
        self.RBs = nn.ModuleList([ResidualBlock(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel_size , dilation=dilation , activation=activation) for _ in range(self.NumberOfRB)])
        self.z2 = ConvBlock(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel_size ,dilation=dilation ,activation=None)

    def forward(self ,x:torch.Tensor) -> torch.Tensor:
        identity = x 
        for RB in self.RBs:
            x = RB(x)
        x = self.z2(x)
        x = identity + x
        return x


#SHCN(sparse harmonic convolution network) modules implemented bellow.-------------------------------------------------------
class SHCN_Block(nn.Module):
    def __init__(self, D_d, in_channels=100, H_dim=200, kernel_size_3d=(5, 3, 3)):
        super().__init__()
        
        # 1. Expand Channels: 3x3 keeps spatial constant via padding=1
        self.expansion = nn.Conv2d(in_channels=in_channels, out_channels=H_dim, kernel_size=3, padding=1)
        
        # 2. 3D Harmonic Auditor: Operates on (B, 1, Depth, H, W)
        # Depth is now 200. We use padding in forward() to keep dimensions exact.
        self.auditor = nn.Conv3d(1, 1, kernel_size=kernel_size_3d, dilation=(D_d, 1, 1), bias=True)
        
        # 3. Projection/Squeeze: Back to 100, spatial constant via padding=1
        self.projection = nn.Conv2d(in_channels=H_dim, out_channels=in_channels, kernel_size=3, padding=1)

        # Volumetric Padding Logic to keep (200, 4, 4) constant through 3D conv
        self.p_d = ((kernel_size_3d[0] - 1) * D_d) // 2 
        self.p_s = (kernel_size_3d[1] - 1) // 2
        
        self.prelu_res = nn.PReLU(in_channels)
        self.prelu_skip = nn.PReLU(in_channels)

    def forward(self, x):
        # x shape: (B, 100, 4, 4)
        identity = x 
        
        # Phase 1: Expand to 200 channels
        # Spatial remains (4, 4) due to padding=1
        feat = self.expansion(x) # (B, 200, 4, 4)
        
        # Phase 2: Volumetric Audit
        # Treat channel dim as 'Depth' for Conv3d
        vol = feat.unsqueeze(1) # (B, 1, 200, 4, 4)
        
        # Pad Depth (D_d), Height (1), and Width (1)
        # F.pad format: (left, right, top, bottom, front, back)
        vol = F.pad(vol, (self.p_s, self.p_s, self.p_s, self.p_s, self.p_d, self.p_d))
        
        # Audit maintains (1, 200, 4, 4)
        audit_raw = self.auditor(vol).squeeze(1) 
        
        # Phase 3: Project back to 100 channels
        # Spatial remains (4, 4)
        projected = self.projection(audit_raw) 
        
        # Final Paths
        res_out = identity + self.prelu_res(projected)
        skip_out = self.prelu_skip(projected)
        
        return res_out, skip_out


class SHCN(nn.Module):
    def __init__(self,in_channels , out_channels ,  repeats=3, layers_per_repeat=7):
        super(SHCN, self).__init__()
        self.layers = nn.ModuleList([])
        
        # Build the 9-Block Grid
        for r in range(repeats):
            for l in range(layers_per_repeat):
                # Slope = 1 Schedule: D_d = 1, 2, 3
                d_d = l + 1 
                self.layers.append(SHCN_Block(D_d=d_d))
                
    def forward(self, x):
        # x: Upsampled DCT Features (B, 100, 4, 4)
        current_backbone = x
        global_skip_sum = 0.
        
        for block in self.layers:
            # Dual output: Backbone flows to next block, Skip flows to the end
            current_backbone, skip_contribution = block(current_backbone)
            global_skip_sum += skip_contribution
            
        # Returning both ensures the "Fusion" can happen in the MDSR wrapper
        return current_backbone, global_skip_sum





#The SEN module implemented bellow ------------------------------------------------------------------------------------------
from torchvision.ops import DeformConv2d
class DeformableConv2d(nn.Module):
        def __init__(
                self,
                in_channels: int = 1,
                kernel_size: int = 3,
                dilation: int = 1,
                ):
            super().__init__()
            if kernel_size % 2 != 1:
                raise ValueError("kernel_size must be odd to keep data flow grounded to the SR network architecture.")
            # compute padding for "same" behavior when stride=1
            padding = dilation * (kernel_size - 1) // 2

            # offset conv predicts 2*K*K offsets per spatial location
            offset_channels = 2 * kernel_size * kernel_size
            self.offset_conv = nn.Conv2d(in_channels, offset_channels, kernel_size=kernel_size, padding=padding)
            # deformable convolution: keeps in_channels == out_channels for residual
            self.deform = DeformConv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=True)

            # initialize offset conv biases to zero to start near regular convolution
            nn.init.constant_(self.offset_conv.weight, 0.)
            if self.offset_conv.bias is not None:
                nn.init.constant_(self.offset_conv.bias, 0.)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # predict offsets
            offset = self.offset_conv(x)  # shape: (N, 2*K*K, H, W)
            # apply deformable conv
            x = self.deform(x, offset)
            return x 


class DeformableResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            kernel_size: int=3,
            dilation: int = 1,
            activation: str = 'leaky_relu'
    ):
        super().__init__()
        self.z1 = ConvBlock(in_channels=in_channels , out_channels=in_channels ,kernel_size=kernel_size,dilation=dilation,activation=activation)
        self.z2 = DeformableConv2d(in_channels=in_channels,kernel_size=kernel_size,dilation=dilation)
    def forward(self , x:torch.Tensor)->torch.Tensor:
        identity = x 
        x = self.z1(x)
        x = self.z2(x)
        x = x + identity # establishing residual connection
        return x

class DeformableResidualGroup(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int=3,
            dilation: int = 1,
            activation: str = 'leaky_relu'
    ):
        super().__init__()
        self.DERBs = nn.ModuleList([DeformableResidualBlock(in_channels=in_channels, kernel_size=kernel_size,dilation=dilation , activation=activation) for _ in range(10)]) 
        self.z2 = ConvBlock(in_channels=in_channels ,out_channels=out_channels , kernel_size=kernel_size,dilation=dilation , activation=None)
    def forward(self ,x:torch.Tensor) ->torch.Tensor:
        identity = x
        for DERB in self.DERBs:
            x = DERB(x)
        #add x = self.z1(x) again and again to increase the number of deformable residual block in this DeformableResidualGroup
        x =self.z2(x)
        x = x+identity
        return x

class ShallowConv2d(nn.Module):
    def __init__(self ,
                 in_channels = 16,
                 out_channels = 100,
                 kernel_size =3,
                 stride = 1 ,
                 dilation = 1 ,
                 activation = None,
                 padding_ = True     
                ):
        super().__init__()
        if (padding_ == True):
            padding = dilation*(kernel_size  - 1) //2 
        else:
            padding = None
        self.z = ConvBlock(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel_size ,dilation = dilation, stride=stride , padding=padding ,activation = activation)

    def forward(self , x:torch.Tensor) -> torch.Tensor:
        x = self.z(x)
        return x
        
class ShrinkingTrunk(nn.Module):
    def __init__(self , 
                in_channels ,
                out_channels ,
                kernel_size = 3,
                dilation = 1,
                stride = 2 ,
                activation = "leaky_relu"
                ):
        super().__init__()
        self.act = nn.LeakyReLU(inplace=True,negative_slope=0.01)
        self.z1 = nn.Conv2d(in_channels=in_channels,out_channels = out_channels,kernel_size=kernel_size , dilation=dilation , stride=stride, padding=1)
        self.z2 = nn.Conv2d(in_channels= out_channels, out_channels=out_channels, kernel_size=kernel_size , dilation=dilation , stride=stride, padding=1)
        self.z3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels , kernel_size=kernel_size , dilation=dilation , stride=stride, padding=1)
        

    def forward(self , x:torch.Tensor) -> torch.Tensor:
        #with each transformation spatial size of the input tensor will decrease (keeping the no. of channels maintained) to match the size of predicted dct feature map
        #spatial size 32→16→8→4
        x = self.z1(x)
        x = self.act(x)
        x = self.z2(x)
        x = self.act(x)
        x = self.z3(x)
        x = self.act(x)
        return x 
    

