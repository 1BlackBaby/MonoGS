import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x):
        if not self.same:
            return self.padding

        ih, iw = x.size()[2:]
        ph = max(self.k[0] - self.stride[0], 0) if ih % self.stride[0] == 0 else max(
            self.k[0] - (ih % self.stride[0]), 0
        )
        pw = max(self.k[1] - self.stride[1], 0) if iw % self.stride[1] == 0 else max(
            self.k[1] - (iw % self.stride[1]), 0
        )
        pl = pw // 2
        pr = pw - pl
        pt = ph // 2
        pb = ph - pt
        return pl, pr, pt, pb

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        return x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]

