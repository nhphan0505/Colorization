import torch
import torch.nn as nn

#====================Generator====================
def conv_block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm: layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))  # encoder use LeakyReLU
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, norm=True, dropout=False):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm: layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.ReLU(inplace=True))            # decoder use ReLU
    if dropout: layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

def conv3x3_halve(in_c, out_c):
    # 3x3 stride=1 + BN + ReLU (paper: "3×3 conv, stride 1, halving channels, BN+ReLU")
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        # Encoder (green, 4x4 s2; first has no BN)
        self.e1 = nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1)   # 32->16
        self.e2 = conv_block(64, 128)                                        # 16->8
        self.e3 = conv_block(128, 256)                                       # 8->4
        self.e4 = conv_block(256, 512)                                       # 4->2
        self.e5 = conv_block(512, 512)                                       # 2->1 (bottleneck 1x1)

        # Decoder (orange, 4x4 deconv s2) + concat + 3x3 conv (BN+ReLU)
        self.d1 = deconv_block(512, 512, dropout=True)                       # 1->2
        self.cat1 = conv3x3_halve(512 + 512, 512)                            # [d1||e4] -> 512

        self.d2 = deconv_block(512, 256, dropout=True)                       # 2->4
        self.cat2 = conv3x3_halve(256 + 256, 256)                            # [d2||e3] -> 256

        self.d3 = deconv_block(256, 128, dropout=True)                       # 4->8
        self.cat3 = conv3x3_halve(128 + 128, 128)                            # [d3||e2] -> 128

        self.d4 = deconv_block(128, 64, dropout=False)                       # 8->16
        self.cat4 = conv3x3_halve(64 + 64, 64)                               # [d4||e1] -> 64  (16x16)

        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # 16->32
        self.final    = nn.Conv2d(64, out_ch, kernel_size=1, stride=1)       # 1x1 conv
        self.tanh     = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)          # [B, 64, 16, 16]
        e2 = self.e2(e1)         # [B,128,  8,  8]
        e3 = self.e3(e2)         # [B,256,  4,  4]
        e4 = self.e4(e3)         # [B,512,  2,  2]
        b  = self.e5(e4)         # [B,512,  1,  1]

        # Decoder + skips
        d1 = self.d1(b)                          # -> [B,512, 2, 2]
        c1 = self.cat1(torch.cat([d1, e4], 1))   # -> [B,512, 2, 2]

        d2 = self.d2(c1)                         # -> [B,256, 4, 4]
        c2 = self.cat2(torch.cat([d2, e3], 1))   # -> [B,256, 4, 4]

        d3 = self.d3(c2)                         # -> [B,128, 8, 8]
        c3 = self.cat3(torch.cat([d3, e2], 1))   # -> [B,128, 8, 8]

        d4 = self.d4(c3)                         # -> [B, 64,16,16]
        c4 = self.cat4(torch.cat([d4, e1], 1))   # -> [B, 64,16,16]

        u  = self.up_final(c4)                   # -> [B, 64,32,32]
        out = self.final(u)                      # -> [B,  2,32,32]
        return self.tanh(out)
    

#====================Dicriminator====================
def d_block(in_c, out_c, k=4, s=2, p=1, use_bn=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class PatchDiscriminator(nn.Module):
    """
    Paper-faithful PatchGAN for colorization (L + ab):
      - First conv: no BN
      - Middle blocks: BN
      - One stride=1 block before head to increase RF
      - Head: 1x1 conv -> logits map (no BN, no sigmoid)
    For 32x32 input, output is 3x3.
    """
    def __init__(self, in_ch=3, nf=64):
        super().__init__()
        self.c1 = d_block(in_ch,   nf,   k=4, s=2, p=1, use_bn=False)  # 32 -> 16
        self.c2 = d_block(nf,      nf*2, k=4, s=2, p=1, use_bn=True)   # 16 -> 8
        self.c3 = d_block(nf*2,    nf*4, k=4, s=2, p=1, use_bn=True)   # 8  -> 4
        self.c4 = d_block(nf*4,    nf*8, k=4, s=1, p=1, use_bn=True)   # 4  -> 3
        self.head = nn.Conv2d(nf*8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.head(x)   # logits
        return x