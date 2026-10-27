"""Convolutional building blocks of the DVAE feature encoder and decoder.

Two families are provided. The *strided* encoder/decoder downsamples time and frequency
by strided convolutions. The *dilated* variant downsamples only frequency and covers the
same temporal receptive field with dilated convolutions, so the latent sequence keeps the
time resolution of the spectrogram.

Attribute names (``net``, ``Network``) are kept stable because they appear in saved
state dicts.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

Pair = tuple[int, int]


def _check_depth(layer_channels: tuple[int, ...]) -> None:
    if len(layer_channels) < 2:
        raise ValueError("layer_channels needs at least two entries")


def conv_output_size(
    size: int, kernel_size: int, stride: int, padding: int, dilation: int = 1
) -> int:
    """Size of one axis after a convolution."""
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class Conv2dLayer(nn.Module):
    """``Conv2d`` followed by optional LeakyReLU and BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Pair = (3, 3),
        stride: Pair = (2, 2),
        padding: Pair = (1, 1),
        dilation: Pair = (1, 1),
        batchnorm: bool = True,
        activation: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        ]
        if activation:
            layers.append(nn.LeakyReLU())
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTranspose2dLayer(nn.Module):
    """``ConvTranspose2d`` followed by optional BatchNorm and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Pair = (3, 3),
        stride: Pair = (2, 2),
        padding: Pair = (1, 1),
        dilation: Pair = (1, 1),
        batchnorm: bool = True,
        activation: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, dilation=dilation
            )
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1dLayer(nn.Module):
    """``Conv1d`` followed by optional LeakyReLU and BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.LeakyReLU(),
        ]
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTranspose1dLayer(nn.Module):
    """``ConvTranspose1d`` followed by LeakyReLU and optional BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, dilation=dilation
            ),
            nn.LeakyReLU(),
        ]
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StridedEncoder(nn.Module):
    """Stack of strided 2-D convolutions, one per entry of ``layer_channels``.

    Maps ``(B, in_channels, F, T)`` to ``(B, layer_channels[-1], F', T')``. With stride 2
    and kernel 3 an axis of ``2m + 1`` samples becomes ``m + 1``.
    """

    def __init__(
        self,
        in_channels: int,
        layer_channels: tuple[int, ...],
        kernel_size: Pair = (3, 3),
        stride: Pair = (2, 2),
        padding: Pair = (1, 1),
    ) -> None:
        super().__init__()
        kwds = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        channels = (in_channels, *layer_channels)
        self.Network = nn.Sequential(
            *[Conv2dLayer(channels[i], channels[i + 1], **kwds) for i in range(len(layer_channels))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Network(x)


class StridedDecoder(nn.Module):
    """Transposed counterpart of :class:`StridedEncoder`.

    ``in_channels`` is the encoder output width (``layer_channels[-1]``). The last layer has
    no batch norm, and its activation is optional. Channel widths mirror the encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_channels: tuple[int, ...],
        kernel_size: Pair = (3, 3),
        stride: Pair = (2, 2),
        padding: Pair = (1, 1),
        activate_final_layer: bool = True,
    ) -> None:
        super().__init__()
        _check_depth(layer_channels)
        kwds = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        channels = (in_channels, *layer_channels[-2::-1])
        layers = [
            ConvTranspose2dLayer(channels[i], channels[i + 1], **kwds)
            for i in range(len(channels) - 1)
        ]
        layers.append(
            ConvTranspose2dLayer(
                layer_channels[0],
                out_channels,
                activation=activate_final_layer,
                batchnorm=False,
                **kwds,
            )
        )
        self.Network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Network(x)


def _dilated_geometry(
    n_layers: int, kernel_size: Pair, padding: Pair
) -> tuple[list[Pair], list[Pair]]:
    """Per-layer ``(padding, dilation)`` that double the temporal dilation at every layer.

    The time padding is chosen so that the time axis keeps its length.
    """
    dilations = 2 ** np.arange(n_layers)
    fields = (kernel_size[-1] - 1) * dilations + 1
    paddings = [(padding[0], int(f // 2)) for f in fields]
    return paddings, [(1, int(d)) for d in dilations]


class StridedDilatedEncoder(nn.Module):
    """Encoder that strides along frequency only and dilates along time."""

    def __init__(
        self,
        in_channels: int,
        layer_channels: tuple[int, ...],
        kernel_size: Pair = (3, 3),
        stride: Pair = (2, 1),
        padding: Pair = (1, 1),
    ) -> None:
        super().__init__()
        paddings, dilations = _dilated_geometry(len(layer_channels), kernel_size, padding)
        channels = (in_channels, *layer_channels)
        self.Network = nn.Sequential(
            *[
                Conv2dLayer(
                    channels[i], channels[i + 1], kernel_size, stride, paddings[i], dilations[i]
                )
                for i in range(len(layer_channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Network(x)


class StridedDilatedDecoder(nn.Module):
    """Transposed counterpart of :class:`StridedDilatedEncoder`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_channels: tuple[int, ...],
        kernel_size: Pair = (3, 3),
        stride: Pair = (2, 1),
        padding: Pair = (1, 1),
        activate_final_layer: bool = True,
    ) -> None:
        super().__init__()
        _check_depth(layer_channels)
        n = len(layer_channels)
        paddings, dilations = _dilated_geometry(n, kernel_size, padding)
        kwds = dict(kernel_size=kernel_size, stride=stride)
        channels = (in_channels, *layer_channels[-2::-1])
        # Dilations follow the trained models: (2**(n-1), 2**(n-3), ..., 2, 1), then 1 again.
        indices = [n - 1, *range(n - 3, -1, -1)]
        layers = [
            ConvTranspose2dLayer(
                channels[j], channels[j + 1], padding=paddings[k], dilation=dilations[k], **kwds
            )
            for j, k in enumerate(indices)
        ]
        layers.append(
            ConvTranspose2dLayer(
                layer_channels[0], out_channels, padding=paddings[0], dilation=dilations[0],
                batchnorm=False, activation=activate_final_layer, **kwds,
            )
        )  # fmt: skip
        self.Network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Network(x)


def _temporal_geometry(
    n_layers: int, kernel_size: int, first_layer: int
) -> tuple[np.ndarray, np.ndarray]:
    dilations = 2 ** (np.arange(n_layers) + first_layer)
    paddings = ((kernel_size - 1) * dilations + 1) // 2
    return dilations, paddings


class TConvSeparator(nn.Module):
    """Dilated temporal convolutions mapping encoder features to per-source features.

    Maps ``(B, in_channels, T)`` to ``(B, out_channels, T)``. Dilation doubles at every
    layer, starting at ``2 ** first_layer`` so that it continues where the 2-D encoder stops.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        layers: int = 1,
        hidden_channels: int = 32,
        first_layer: int = 0,
    ) -> None:
        super().__init__()
        dilations, paddings = _temporal_geometry(layers, kernel_size, first_layer)

        def layer(i: int, cin: int, cout: int, batchnorm: bool = True) -> Conv1dLayer:
            return Conv1dLayer(
                cin, cout, kernel_size, 1, int(paddings[i]), int(dilations[i]), batchnorm=batchnorm
            )

        if layers == 1:
            mlist = [layer(0, in_channels, out_channels)]
        else:
            mlist = [layer(0, in_channels, hidden_channels)]
            mlist += [layer(i, hidden_channels, hidden_channels) for i in range(1, layers - 1)]
            mlist.append(layer(layers - 1, hidden_channels, out_channels, batchnorm=False))
        self.Network = nn.Sequential(*mlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Network(x)


class TConvTranspose(nn.Module):
    """Transposed counterpart of :class:`TConvSeparator`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        layers: int = 1,
        hidden_channels: int = 32,
        first_layer: int = 0,
    ) -> None:
        super().__init__()
        dilations, paddings = _temporal_geometry(layers, kernel_size, first_layer)

        def layer(i: int, cin: int, cout: int) -> ConvTranspose1dLayer:
            return ConvTranspose1dLayer(
                cin, cout, kernel_size, 1, int(paddings[i]), int(dilations[i])
            )

        if layers == 1:
            mlist = [layer(0, in_channels, out_channels)]
        else:
            mlist = [layer(layers - 1, in_channels, hidden_channels)]
            mlist += [layer(i, hidden_channels, hidden_channels) for i in range(layers - 3, -1, -1)]
            mlist.append(layer(0, hidden_channels, out_channels))
        self.Network = nn.Sequential(*mlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Network(x)
