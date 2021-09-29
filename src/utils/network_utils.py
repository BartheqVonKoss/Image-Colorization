import torch


class Conv2D(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            batch_normalization,
            bias,
            kernel_size=3,
            stride=1,
            activation=torch.nn.ReLU(),
    ):
        super(Conv2D, self).__init__()

        self.layer = []
        self.layer.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            ))
        if batch_normalization:
            self.layer.append(torch.nn.BatchNorm2d(out_channels))
        if activation:
            self.layer.append(activation)
        self.layer = torch.nn.Sequential(*self.layer)

    def forward(self, input_op):
        return self.layer(input_op)


def convolution(in_channels: int,
                out_channels: int,
                batch_normalization: bool,
                bias: bool,
                pooling=False,
                kernel_size=3,
                stride=1,
                activation=torch.nn.ReLU()):
    layer = []

    conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
    )
    layer.append(conv)
    if batch_normalization:
        layer.append(torch.nn.BatchNorm2d(num_features=out_channels))
    if pooling:
        layer.append(torch.nn.AvgPool2d(2, 2))
    if activation:
        layer.append(activation)

    return layer


def get_flatten_size(model: torch.nn.Module) -> int:
    in_h, in_w = cfg.HEIGHT, cfg.WIDTH
    for layer in model.modules():
        if type(layer) is torch.nn.Conv2d:
            in_h, in_w = calc_out_conv_layers(
                in_h,
                in_w,
                layer.kernel_size,
                layer.padding,
                layer.stride,
            )
            last_conv = layer

    return in_h * in_w * last_conv.out_channels


def calc_out_conv_layers(in_h, in_w, kernels, paddings, strides):

    for kernel, padding, stride in zip(kernels, paddings, strides):
        out_h = (in_h - kernel + 2 * padding) / stride + 1
        out_w = (in_w - kernel + 2 * padding) / stride + 1

    return int(out_h), int(out_w)
