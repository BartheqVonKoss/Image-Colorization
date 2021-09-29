from torch.utils.tensorboard import SummaryWriter
import os
import torch

from collections import OrderedDict
from torchsummary import summary

from src.utils.network_utils import get_flatten_size, convolution, Conv2D


class Network(torch.nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.config = config
        self.module_1 = []
        self.module_2 = []
        self.module_3 = []
        self.module_4 = []
        self.module_5 = []

        self.module_1 += convolution(
            in_channels=1,
            out_channels=int(self.config.div * 16),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_1 += convolution(
            in_channels=int(self.config.div * 16),
            out_channels=int(self.config.div * 16),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )

        self.module_2 += convolution(
            in_channels=int(self.config.div * 16),
            out_channels=int(self.config.div * 32),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_2 += convolution(
            in_channels=int(self.config.div * 32),
            out_channels=int(self.config.div * 32),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )

        self.module_3 += convolution(
            in_channels=int(self.config.div * 32),
            out_channels=int(self.config.div * 64),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_3 += convolution(
            in_channels=int(self.config.div * 64),
            out_channels=int(self.config.div * 64),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_3 += convolution(
            in_channels=int(self.config.div * 64),
            out_channels=int(self.config.div * 64),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )

        self.module_4 += convolution(
            in_channels=int(self.config.div * 64),
            out_channels=int(self.config.div * 128),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_4 += convolution(
            in_channels=int(self.config.div * 128),
            out_channels=int(self.config.div * 128),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_4 += convolution(
            in_channels=int(self.config.div * 128),
            out_channels=int(self.config.div * 128),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )

        self.module_5 += convolution(
            in_channels=int(self.config.div * 128),
            out_channels=int(self.config.div * 64),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_5 += convolution(
            in_channels=int(self.config.div * 64),
            out_channels=int(self.config.div * 16),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.module_5 += convolution(
            in_channels=int(self.config.div * 16),
            out_channels=int(self.config.div * 16),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )

        self.module_1 = torch.nn.Sequential(*self.module_1)
        self.module_2 = torch.nn.Sequential(*self.module_2)
        self.module_3 = torch.nn.Sequential(*self.module_3)
        self.module_4 = torch.nn.Sequential(*self.module_4)
        self.module_5 = torch.nn.Sequential(*self.module_5)

        self.conv_2 = Conv2D(
            in_channels=int(self.config.div * 32),
            out_channels=int(self.config.div * 16),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )

        self.conv_3 = Conv2D(
            in_channels=int(self.config.div * 64),
            out_channels=int(self.config.div * 32),
            batch_normalization=self.config.batch_normalization,
            bias=self.config.bias,
        )
        self.conv_4 = Conv2D(
            in_channels=int(self.config.div * 128),
            out_channels=int(self.config.div * 64),
            batch_normalization=self.config.batch_normalization,
            kernel_size=1,
            bias=self.config.bias,
            stride=1,
        )

        self.conv_final_U = Conv2D(
            in_channels=int(self.config.div * 16),
            out_channels=50,
            batch_normalization=False,
            bias=False,
            kernel_size=1,
            activation=None,
        )

        self.conv_final_V = Conv2D(
            in_channels=int(self.config.div * 16),
            out_channels=50,
            batch_normalization=False,
            bias=False,
            kernel_size=1,
            activation=None,
        )

    def forward(self, input_op):
        output_module_1 = self.module_1(input_op)
        pooled_output_module_1 = torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)(output_module_1)
        output_module_2 = self.module_2(pooled_output_module_1)
        pooled_output_module_2 = torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)(output_module_2)
        output_module_3 = self.module_3(pooled_output_module_2)
        pooled_output_module_3 = torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)(output_module_3)
        output_module_4 = self.module_4(pooled_output_module_3)
        output_conv_4 = self.conv_4(output_module_4)
        upsampled_output_conv_4 = torch.nn.functional.interpolate(
            output_conv_4, (output_module_3.size(2), output_module_3.size(3)))
        elemwise_sum_1 = upsampled_output_conv_4 + output_module_3
        output_conv_3 = self.conv_3(elemwise_sum_1)
        upsampled_output_conv_3 = torch.nn.functional.interpolate(
            output_conv_3, (output_module_2.size(2), output_module_2.size(3)))
        elemwise_sum_2 = upsampled_output_conv_3 + output_module_2
        output_conv_2 = self.conv_2(elemwise_sum_2)

        upsampled_output_conv_4 = torch.nn.functional.interpolate(
            upsampled_output_conv_4,
            (output_module_1.size(2), output_module_1.size(3)))
        upsampled_output_conv_3 = torch.nn.functional.interpolate(
            upsampled_output_conv_3,
            (output_module_1.size(2), output_module_1.size(3)))
        upsampled_output_conv_2 = torch.nn.functional.interpolate(
            output_conv_2, (output_module_1.size(2), output_module_1.size(3)))
        concatenated_output = torch.cat(
            (upsampled_output_conv_4, upsampled_output_conv_3,
             upsampled_output_conv_2, output_module_1),
            dim=1)

        output_conv_pre_final = self.module_5(concatenated_output)
        output_U = self.conv_final_U(output_conv_pre_final)
        output_V = self.conv_final_V(output_conv_pre_final)

        output_U = torch.nn.functional.interpolate(
            output_U, (input_op.size(2), input_op.size(3)))
        output_V = torch.nn.functional.interpolate(
            output_V, (input_op.size(2), input_op.size(3)))
        return output_U, output_V


def test_network(model):

    writer = SummaryWriter(os.path.join(self.config.work_dir, 'model'))
    model = model.to('cuda')
    print(summary(model, (1, self.config.HEIGHT, self.config.WIDTH)))
    input_ = torch.randn((2, 1, self.config.HEIGHT, self.config.WIDTH)).to('cuda')
    print(model(input_)[0].size())
    print(model(input_)[1].size())
    writer.add_graph(model, input_)
    writer.flush()
    writer.close()


def main():
    net = Network()
    test_network(net)


if __name__ == '__main__':
    main()
