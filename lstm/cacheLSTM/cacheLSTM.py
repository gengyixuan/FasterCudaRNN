import math
import torch

# Our module!
import cache_lstm

class cache_lstmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, XU, W, h0, c0):
        # XU: (T, B, 4Dh)
        # W:  (4Dh, Dh)
        # h0: (1, B, Dh)
        # c0: (B, Dh)
        h, cT = cache_lstm.forward(XU, W, h0, c0)
        variables = [cT, W]
        ctx.save_for_backward(*variables)

        return h, cT

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = cache_lstm.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class cacheLSTM(torch.nn.Module):
    def __init__(self, Di, Dh):
        super(cacheLSTM, self).__init__()
        self.Di = Di
        self.Dh = Dh # Dh should be smaller than 70 !!!!
        self.U = torch.nn.Parameter(torch.empty(Di, 4 * Dh)) # (Di, 4Dh)
        self.W = torch.nn.Parameter(torch.empty(4 * Dh, Dh))
        self.reset_parameters()

    def setUW(self, init_U, init_W):
        self.U = torch.nn.Parameter(init_U)
        self.W = torch.nn.Parameter(init_W)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Dh)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    # X: (T, B, Di)
    # h0: (B, Dh)
    # c0: (B, Dh)
    def forward(self, X, h0, c0):
        h0 = h0.unsqueeze(0)
        XU = torch.matmul(X, self.U) # (T, B, Di) x (Di, 4Dh) -> (T, B, 4Dh)
        return cache_lstmFunction.apply(XU, self.W, h0, c0)