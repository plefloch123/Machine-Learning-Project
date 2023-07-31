import torch
import torch.nn as nn #nn stands for neural network
import torchvision.transforms.functional as TF #give fine-grained control over the transformations.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()       #used to give access to methods and properties of a parent or sibling class
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,3, 1, 1, bias = False), #kernel size = 3, stride = 1, padding = 1
            nn.BatchNorm2d(out_channels), #BatchNorm2d is the number of dimensions/channels that output from the last layer and come in to the batch norm layer
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )    #Sequential is a construction which is used when you want to run certain layers sequentially

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 161 x 161 -> 80 x 80 so output = 160 x 160 (lost pixels)

        # Down part fo UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,  # if a neural network's stride is set to 1, the filter will move one pixel, or unit,  at a time
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck layer of UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Last final conv (1*1 conv) at the end of UNET
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        #For dow part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        #For the bottleneck layer
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # link thins together
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()