import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad

from model.embedder import *


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad


# class ImplicitNet(nn.Module):
#     def __init__(
#         self,
#         d_in,
#         dims,
#         skip_in=(),
#         geometric_init=True,
#         radius_init=1,
#         beta=100
#     ):
#         super().__init__()

#         dims = [d_in] + dims + [1]

#         self.num_layers = len(dims)
#         self.skip_in = skip_in

#         for layer in range(0, self.num_layers - 1):

#             if layer + 1 in skip_in:
#                 out_dim = dims[layer + 1] - d_in
#             else:
#                 out_dim = dims[layer + 1]

#             lin = nn.Linear(dims[layer], out_dim)

#             # if true preform preform geometric initialization
#             if geometric_init:

#                 if layer == self.num_layers - 2:

#                     torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
#                     torch.nn.init.constant_(lin.bias, -radius_init)
#                 else:
#                     torch.nn.init.constant_(lin.bias, 0.0)

#                     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

#             setattr(self, "lin" + str(layer), lin)

#         if beta > 0:
#             self.activation = nn.Softplus(beta=beta)

#         # vanilla relu
#         else:
#             self.activation = nn.ReLU()

#     def forward(self, input):

#         x = input

#         for layer in range(0, self.num_layers - 1):

#             lin = getattr(self, "lin" + str(layer))

#             if layer in self.skip_in:
#                 x = torch.cat([x, input], -1) / np.sqrt(2)

#             x = lin(x)

#             if layer < self.num_layers - 2:
#                 x = self.activation(x)

#         return x


# PASTED from IDR code
class ImplicitNet(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        dims,
        geometric_init=True,
        bias=1.0,
        skip_in=(),
        weight_norm=True,
        multires=0,
        weights=None,
    ):
        super().__init__()

        # Added for conf shenanigans
        bias = int(bias)
        d_out = int(d_out)
        multires = int(multires)

        # input = latent + xyz
        # output = sdf
        dims = [d_in] + dims + [d_out]

        # Embed xyz
        latent_dim = d_in - 3
        self.embed_fn = None
        if multires > 0:
            embed_fn, embed_dim = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = latent_dim + embed_dim

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init and weights is None:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, latent_dim+3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :latent_dim+3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

        if weights:
            self.load_state_dict(torch.load(weights)['model_state_dict'])

    def forward(self, input):

        if self.embed_fn is not None:
            embd = self.embed_fn(input[:,-3:])
            input = torch.cat([input[:,:-3], embd], dim=-1)

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x