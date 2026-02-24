import torch
import torch.nn as nn

#====================Generator====================
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.latent = latent
        
    def forward(self, x):
        conv = self.conv(x)
        conv = self.relu(conv)
        if self.latent==False:
            pool = self.pool(conv)
        else:
            pool = None
        return conv, pool

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.uconv = nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding='same')
        self.relu = nn.ReLU()
        
    def forward(self, x, skip_block):
        uconv = self.uconv(x)
        con = torch.concat([skip_block, uconv], dim=1)
        conv = self.conv(con)
        conv = self.relu(conv)
        return conv
        
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoding
        self.enc_block_1 = Encoder(1, 32)
        self.enc_block_2 = Encoder(32, 64)
        self.enc_block_3 = Encoder(64, 128)
        self.enc_block_4 = Encoder(128, 256)
        self.enc_block_5 = Encoder(256, 512, latent=True)

        # Decoding
        self.dec_block_4 = Decoder(512, 256)
        self.dec_block_3 = Decoder(256, 128)
        self.dec_block_2 = Decoder(128, 64)
        self.dec_block_1 = Decoder(64, 32)

        # Output block
        self.compute = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding='same'),
        )
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # input shape (B, 1, 32, 32)
        s1, p1 = self.enc_block_1(x) # (B, 32, 32, 32), (B, 32, 16, 16)
        s2, p2 = self.enc_block_2(p1) # (B, 64, 32, 32), (B, 64, 8, 8)
        s3, p3 = self.enc_block_3(p2) # (B, 128, 8, 8), (B, 128, 4, 4)
        s4, p4 = self.enc_block_4(p3) # (B, 256, 4, 4), (B, 256, 2, 2)
        latent, _ = self.enc_block_5(p4) # (B, 512, 2, 2)

        d4 = self.dec_block_4(latent, s4) # (B, 256, 4, 4)
        d3 = self.dec_block_3(d4, s3) # (B, 128, 8, 8)
        d2 = self.dec_block_2(d3, s2) # (B, 64, 16, 16)
        d1 = self.dec_block_1(d2, s1) # (B, 32, 32, 32)

        ab = self.compute(d1) #(B, 2, 32, 32)
        ab = self.tanh(ab) # [-1,1]
        return ab
    

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