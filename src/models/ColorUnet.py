#====================Model====================
import torch.nn as nn
# ---------------- Encoder Block ---------------- #
class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self, x):
        x = self.block(x)
        return x

    

# ---------------- Decoder Block ---------------- #
class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, bottle_neck = False):
        super().__init__()
        self.bottle_neck = bottle_neck
        self.up = nn.ConvTranspose2d(in_ch*(2-bottle_neck), in_ch*(2-bottle_neck), kernel_size=3, stride=2, padding = 1, output_padding = 1)
        self.block = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Conv2d(in_ch*(2-bottle_neck), in_ch*(2-bottle_neck), kernel_size=3, stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch*(2-bottle_neck), out_ch, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x, skip = None):
        if self.bottle_neck == False:
            x = torch.cat([skip, x], dim=1)

        # print(x.shape)
        x = self.up(x)
        # print(x.shape)
        
        return self.block(x)


#----------------- Output Block ---------------- #
class OutputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=3, stride=2, padding = 1, output_padding = 1)
        self.block = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        )

    def forward(self, x):
        x = self.up(x)
        return self.block(x)

# ---------------- U-Net ---------------- #
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):  # input L channel, output ab channels
        super().__init__()
        # ---------------- Encoder ----------------
        self.enc1 = DownConv(in_ch, 16)    
        self.enc2 = DownConv(16, 32)      
        self.enc3 = DownConv(32, 64)     
        

        # ---------------- Decoder ----------------
        self.dec3 = UpConv(64, 32, bottle_neck = True)     
        self.dec2 = UpConv(32, 16)
        

        # ---------------- Output ----------------
        self.final = OutputBlock(16, 32)   

    def forward(self, x):
        # ----- Encoding path -----
        s1 = self.enc1(x)   
        s2 = self.enc2(s1)  
        s3 = self.enc3(s2)  
    
        # ----- Decoding path -----
        
        d3 = self.dec3(s3)  
        d2 = self.dec2(d3, s2)  
        
          
        # ----- Output -----
        out = self.final(d2)    
        return out
