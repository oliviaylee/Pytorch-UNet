""" Full assembly of the parts to form the complete network """

from .unet_parts import *

# https://github.com/gabolsgabs/cunet/blob/master/cunet/train/models/cunet_model.py

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, condition=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.condition = condition

        self.inc = (DoubleConv(n_channels, 64, condition))
        self.down1 = (Down(64, 128, condition))
        self.down2 = (Down(128, 256, condition))
        self.down3 = (Down(256, 512, condition))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, condition))

        # Fully connected decoding for post-contact trajectory
        # TO-DO: Tweak hidden layer sizes
        self.fc1 = nn.Linear(1024 // factor, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2*10) 
        #(x, y), either autoregressively output subsequent (x,y)'s or just at once
        self.relu = nn.ReLU(inplace=True)

        self.up1 = (Up(1024, 512 // factor, bilinear, condition))
        self.up2 = (Up(512, 256 // factor, bilinear, condition))
        self.up3 = (Up(256, 128 // factor, bilinear, condition))
        self.up4 = (Up(128, 64, bilinear, condition))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Fully connected decoding for post-contact trajectory
        # to import Flatten
        trajectory = self.fc3(self.relu(self.fc2(self.relu(self.fc1(Flatten(x5))))))
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        contact_pts = self.outc(x) # logits = self.outc(x)
        return trajectory, contact_pts # logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)