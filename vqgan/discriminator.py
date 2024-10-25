import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channels:int, num_of_filters_last:int = 64, num_of_layers:int = 3):
        super(Discriminator, self).__init__()

        layers =[
            nn.Conv2d(image_channels, num_of_filters_last, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        ]

        num_filters_mult = 1

        for i in range(1, num_of_layers+1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)

            layers += [
                nn.Conv2d(num_of_filters_last*num_filters_mult_last, num_of_filters_last*num_filters_mult,
                        kernel_size=4,stride= 2 if i < num_of_layers else 1, padding=1, bias=False),
                nn.BatchNorm2d(num_of_filters_last*num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers.append(nn.Conv2d(num_of_filters_last*num_filters_mult, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)