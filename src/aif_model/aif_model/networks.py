"""Majority of code is by pi-tau on github: https://github.com/pi-tau/vae"""

import torch
import torch.nn as nn
import aif_model.config as c

class PositionalNorm(nn.LayerNorm):
    # Author: pi-tau
    """PositionalNorm is a normalization layer used for 3D image inputs that
    normalizes exclusively across the channels dimension.
    https://arxiv.org/abs/1907.04312
    """

    def forward(self, x):
        # The input is of shape (B, C, H, W). Transpose the input so that the
        # channels are pushed to the last dimension and then run the standard
        # LayerNorm layer.
        x = x.permute(0, 2, 3, 1).contiguous()
        out = super().forward(x)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out


class ResBlock(nn.Module):
    # Author: pi-tau
    """Residual block following the "bottleneck" architecture as described in
    https://arxiv.org/abs/1512.03385. See Figure 5.
    The residual blocks are defined following the "pre-activation" technique
    as described in https://arxiv.org/abs/1603.05027.

    The residual block applies a number of convolutions represented as F(x) and
    then skip connects the input to produce the output F(x)+x. The residual
    block can also be used to upscale or downscale the input by doubling or
    halving the spatial dimensions, respectively. Scaling is performed by the
    "bottleneck" layer of the block. If the residual block changes the number of
    channels, or the spatial dimensions are up- or down-scaled, then the input
    is also transformed into the desired shape for the addition operation.
    """

    def __init__(self, in_chan, out_chan, scale="same"):
        """Init a Residual block.

        Args:
            in_chan: int
                Number of channels of the input tensor.
            out_chan: int
                Number of channels of the output tensor.
            scale: string, optional
                One of ["same", "upscale", "downscale"].
                Upscale or downscale by half the spatial dimensions of the
                input tensor. Default is "same", i.e., no scaling.
        """
        super().__init__()
        assert scale in ["same", "upscale", "downscale"]
        if scale == "same":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, padding="same")
            stride = 1
        elif scale == "downscale":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, stride=2, padding=1)
            stride = 2
        elif scale == "upscale":
            bottleneck = nn.ConvTranspose2d(in_chan//2, in_chan//2, kernel_size=4, stride=2, padding=1)
            stride = 1

        # The residual block employs the bottleneck architecture as described
        # in Sec 4. under the paragraph "Deeper Bottleneck Architectures" of the
        # original paper introducing the ResNet architecture.
        # The block uses a stack of three layers: `1x1`, `3x3` (`4x4`), `1x1`
        # convolutions. The first `1x1` reduces (in half) the number of channels
        # before the expensive `3x3` (`4x4`) convolution. The second `1x1`
        # up-scales the channels to the requested output channel size.
        self.block = nn.Sequential(
            # 1x1 convolution
            PositionalNorm(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, in_chan//2, kernel_size=1),

            # 3x3 convolution if same or downscale, 4x4 transposed convolution if upscale
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            bottleneck,

            # 1x1 convolution
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            nn.Conv2d(in_chan//2, out_chan, kernel_size=1),
        )

        # If channels or spatial dimensions are modified then transform the
        # input into the desired shape, otherwise use a simple identity layer.
        self.id = nn.Identity()
        if in_chan != out_chan or scale == "downscale":
            # We will downscale by applying a strided `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride),
            )
        if scale == "upscale":
            # We will upscale by applying a nearest-neighbor upsample.
            # Channels are again modified using a `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

    def forward(self, x):
        return self.block(x) + self.id(x)


class Encoder(nn.Module):
    # Author: pi-tau
    """Encoder network used for encoding the input space into a latent space.
    The encoder maps a vector(tensor) from the input space into a distribution
    over latent space. This distribution is assumed to be Normal and is
    parametrized by mu and std.
    """

    def __init__(self, in_chan, latent_dim):
        """Init an Encoder module.

        Args:
            in_chan: int
                Number of input channels of the images.
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()

        # The encoder architecture follows the design of ResNet stacking several
        # residual blocks into groups, operating on different scales of the image.
        # The first residual block from each group is responsible for downsizing
        # the image and increasing the channels.
        self.net = nn.Sequential(
            # Stem.
            nn.Conv2d(in_chan, 32, kernel_size=3, padding="same"),

            # Body.
            ResBlock(in_chan=32, out_chan=64, scale="downscale"),           
            
            ResBlock(in_chan=64, out_chan=128, scale="downscale"),
            
            ResBlock(in_chan=128, out_chan=256, scale="downscale"),       

            ResBlock(in_chan=256, out_chan=512, scale="downscale"),            

            # Head.
            PositionalNorm(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2*latent_dim),
        )

    def forward(self, x):
        """Forward the input through the encoder and return the parameters `mu`
        and `std` of the Normal distribution.
        """
        enc = self.net(x)
        mu, log_std = torch.chunk(enc, chunks=2, dim=-1)
        return mu, log_std


class Decoder(nn.Module):
    # Author: pi-tau
    """Decoder network used for decoding the latent space back into the input
    space. The decoder maps a vector(tensor) from the latent space into a
    distribution over the input space. This distribution is assumed to be Normal
    and is parametrized by mu and std. We will assume the std to be constant.
    """

    def __init__(self, out_chan, latent_dim):
        """Init an Decoder module.

        Args:
            out_chan: int
                Number of output channels of the images.
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()

        # The decoder architecture follows the design of a reverse ResNet
        # stacking several residual blocks into groups, operating on different
        # scales of the image. The first residual block from each group is
        # responsible for up-sizing the image and reducing the channels.
        self.net = nn.Sequential(
            # Inverse head.
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 512 * c.height//16 * c.width//16),
            nn.Unflatten(dim=-1, unflattened_size=(512, c.height//16, c.width//16)),     # 8x8 #4x4

            # Body.
            ResBlock(in_chan=512, out_chan=256, scale="upscale"),

            ResBlock(in_chan=256, out_chan=128, scale="upscale"), 

            ResBlock(in_chan=128, out_chan=64, scale="upscale"),       
            
            ResBlock(in_chan=64, out_chan=32, scale="upscale"),             

            # Inverse stem.
            PositionalNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, out_chan, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        """Forward the input through the encoder and return the parameters `mu`
        and `std` of the Normal distribution. Note that we will not be learning
        the covariance matrix and instead will be using a constant identity
        matrix as the covariance matrix.
        """
        mu = self.net(x)
        log_std = torch.zeros_like(mu)
        return mu, log_std


class FullyConnected(nn.Module):
    # Author: Tin M.
    """Fully connected network that translates from the latent space of the VAE
    into an interpetable space of either pixel coordinates of individual objects
    or camera angles"""

    def __init__(self, in_size, out_size):
        """Initialize the fully connected network
        Args:
            in_size: int
            out_size: int
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            # nn.Linear(1024, 1024),
            # nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            # nn.Linear(512, 512),
            # nn.PReLU(),
            # nn.Linear(512, 512),
            # nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512,out_size)
        )

    def forward(self, x):
        """Forward the input through the fully connected network, 
        translating from the latent space into the pixel/angle space"""
        return self.net(x)
    
    def loss(self, x, y):
        outputs = self.forward(x)
        # diffs = (32/2*(np.concatenate((outputs.detach().numpy(),y.detach().numpy()),axis=1)+1)).astype(int)
        # # print(diffs.shape)
        # np.savetxt("differences.csv", diffs, delimiter=',')
        return nn.MSELoss()(outputs, y)
    
    def save(self, path):
        """
        Save network to file
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load network from file
        """
        self.load_state_dict(torch.load(path))
        self.eval()
