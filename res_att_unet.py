import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, in_channel=1024, num_class=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.bn = nn.BatchNorm1d(1024)
        #self.relu = nn.ReLU()
        self.f1 = nn.Linear(in_channel, in_channel//2)
        self.f2 = nn.Linear(in_channel//2, num_class)

    def forward(self, x):
        x = self.gap(x)
        y = self.f1(x.squeeze())
        #y = self.relu(y)
        #y = self.bn(y)
        y = self.f2(y)

        return y
    

# Ali Revised
class ResAttBlock_Revised(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResAttBlock_Revised, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        # 3 Convolution Layer, 1x1, 3x3, 1x1
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Residual Convolution kernel_size=1
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax2d = nn.Softmax2d()
        self.act = nn.GELU()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], device=device), requires_grad=True)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        out_1 = self.conv_block(x)

        if x.shape != out_1.shape:
            out_res = self.act(self.residual_conv(x))
            out_c = self.sigmoid(self.adaptive_pool(out_1)) * out_res
            out_s = self.softmax2d(out_c) * out_res
            output = out_1 + (self.w[0] * out_c) + (self.w[1] * out_s)
        else:
            out_c = self.sigmoid(self.adaptive_pool(out_1)) * x
            out_s = self.softmax2d(out_c) * x
            output = out_1 + (self.w[0] * out_c) + (self.w[1] * out_s)
        return self.conv_out(output)


class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResAttBlock, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        # 3 Convolution Layer, 1x1, 3x3, 1x1
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Residual Convolution kernel_size=1
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        # Output Convolution of F_out
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax2d = nn.Softmax2d()
        self.act = nn.GELU()

    def forward(self, x):
        out_1 = self.conv_block(x)

        if x.shape != out_1.shape:
            out_res = self.residual_conv(x)
            out_c = self.sigmoid(self.adaptive_pool(out_1)) * out_res
            out_s = self.softmax2d(out_c) * out_res
            output = out_1 + out_c + out_s
        else:
            out_c = self.sigmoid(self.adaptive_pool(out_1)) * x
            out_s = self.softmax2d(out_c) * x
            output = out_1 + out_c + out_s
        return self.conv_out(output)


class DoubleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleResidualBlock, self).__init__()

        self.res_block_1 = ResAttBlock(in_channels, out_channels)
        self.res_block_2 = ResAttBlock(out_channels, out_channels)

    def forward(self, x):
        y = self.res_block_1(x)
        y = self.res_block_2(y)
        return y


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResAttBlock(in_channels, out_channels)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], dim=1)
        y = self.res_block(x)
        return y


class ResAttEncoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256]):
        super(ResAttEncoder, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleResidualBlock(features[-1], features[-1]*2)

        for idx, feature in enumerate(features):
            if idx == 0:
                self.downs.append(DoubleResidualBlock(in_channels, feature))
            else:
                self.downs.append(DoubleResidualBlock(features[idx - 1], feature))

    def forward(self, x):
        skip_conns = []

        for down in self.downs:
            x = down(x)
            skip_conns.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        return x, skip_conns[::-1]


class ResAttDecoder(nn.Module):
    def __init__(self, features=[64, 128, 256]):
        super(ResAttDecoder, self).__init__()
        self.ups = nn.ModuleList()

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ResAttBlock(feature * 2, feature))

    def forward(self, x, skip_connections):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                #x = TF.resize(x, size=skip_connection.shape[2:])
                x = TF.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return x


class ResAttUnet_1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256]):
        super(ResAttUnet_1, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for idx, feature in enumerate(features):
            if idx == 0:
                self.encoder.append(DoubleResidualBlock(in_channels, feature))
            else:
                self.encoder.append(DoubleResidualBlock(features[idx - 1], feature))

        self.bottleneck = DoubleResidualBlock(features[-1], features[-1] * 2)

        # Up part
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResAttBlock_Revised(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1, padding=0)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        skip_conn = []
        for down in self.encoder:
            x = down(x)
            skip_conn.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_conns = skip_conn[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_conn = skip_conns[idx // 2]

            if x.shape != skip_conn.shape:
                x = TF.interpolate(x, size=skip_conn.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_conn, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        x = self.final_conv(x)
        x = self.activation(x)

        return x


class ResAttUnet(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256]):
        super(ResAttUnet, self).__init__()
        self.encoder = ResAttEncoder(in_channels=in_channels, features=features)
        self.decoder = ResAttDecoder(features=features)

    def forward(self, x):
        y, skip = self.encoder(x)
        y = self.decoder(y, skip)
        return y


def test():
    x = torch.randn((4, 1, 256, 256)).to(device)
    input_size = (3, 256, 256)
    model = ResAttUnet(in_channels=1).to(device)
    preds = model(x)
    print("output: ", preds.shape)
    print(count_parameters(model))

    for names, params in model.encoder.named_parameters():
        if params.requires_grad:
            params.requires_grad = False
    print(count_parameters(model))
    print(model)
    #summary(model, input_size)


if __name__ == "__main__":
    test()
