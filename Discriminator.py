import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1):   # 默认1通道，条件GAN时设为2
        super(PatchDiscriminator, self).__init__()
        # print(f"🚀🚀🚀 成功初始化判别器！当前接收通道数定为: {in_channels} 🚀🚀🚀")
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)),
        )

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3, downsample_mode="avg"):
        super().__init__()
        self.discriminators = nn.ModuleList([PatchDiscriminator() for _ in range(num_scales)])
        self.downsample = {
            "avg": nn.AvgPool2d(2, stride=2),
            "max": nn.MaxPool2d(2, stride=2),
            "conv": nn.Conv2d(1, 1, 3, stride=2, padding=1)
        }[downsample_mode]

    def forward(self, x):
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        return outputs


def discriminator_loss(output_real, output_fake, real_labels=None, fake_labels=None):
    criterion = nn.BCEWithLogitsLoss()
    real_labels = real_labels if real_labels is not None else torch.ones_like(output_real)
    fake_labels = fake_labels if fake_labels is not None else torch.zeros_like(output_fake)
    return (criterion(output_real, real_labels) + criterion(output_fake, fake_labels)) / 2


def multi_discriminator_loss(fake_outputs, real_outputs, weights=None):
    criterion = nn.BCEWithLogitsLoss()
    weights = [1.0] * len(fake_outputs) if weights is None else weights
    loss = sum(
        w * (criterion(fake, torch.ones_like(fake)) + criterion(real, torch.zeros_like(real)))
        for (fake, real), w in zip(zip(fake_outputs, real_outputs), weights)
    )
    return loss / (2 * sum(weights))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc = PatchDiscriminator(in_channels=2).to(device)
    fake = torch.randn(2, 2, 512, 512).to(device)
    print("Single discriminator output shape:", disc(fake).shape)