import torch
import torch.nn as nn


class TriplaneDiscriminator2D(nn.Module):
    def __init__(self, opts):
        super(TriplaneDiscriminator2D, self).__init__()
        self.opts = opts

        self.load_weights()

        # Each depth slice adds to the channel dimension, so input channels = C * D = 3 * 96 = 288
        self.main = nn.Sequential(
            # Input size: (B x 288 x 256 x 256)
            nn.Conv2d(288, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (B x 64 x 128 x 128)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (B x 128 x 64 x 64)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (B x 256 x 32 x 32)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (B x 512 x 16 x 16)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (B x 512 x 8 x 8)
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0, bias=False),
            # State size: (B x 1 x 6 x 6)
            nn.AdaptiveAvgPool2d(1),
            # nn.Sigmoid(),
        )

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            checkpoint = torch.load(self.opts.checkpoint_path, map_location="cpu")["discriminator_state_dict"]
            self.load_state_dict(checkpoint, strict=False)
            print(f"Loading discriminator weights from {self.opts.checkpoint_path}")

    def forward(self, x):
        # Reshape x from (B, C, D, W, H) to (B, C*D, W, H)
        B, C, D, W, H = x.size()
        x = x.reshape(B, C * D, W, H)
        return self.main(x)
