import torch
import pprint
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from distributions import PiecewiseExponentialDistribution
import math
from bisect import bisect_left, insort
import matplotlib.pyplot as plt

C = torch.tensor(1.1)
INF = torch.tensor(math.inf)
torch.manual_seed(0)
rng = np.random.RandomState(0)

def derivative (f, x) : 
    if isinstance(x, float) : 
        x = torch.tensor(x)
    var = Variable(x, requires_grad=True)
    y = f(var)
    y.backward()
    return var.grad

def pieces (xs, h, xRange) : 
    ys = [h(x) for x in xs] 
    ms = [derivative(h, x) for x in xs]
    zs = []
    for x, x_, y, y_, m, m_ in zip(xs, xs[1:], ys, ys[1:], ms, ms[1:]) : 
        z = (y_ - y - x_ * m_ + x * m) / (m - m_)
        zs.append(z)
    cs = [y - m * x for x, y, m in zip(xs, ys, ms)]
    lo, hi = xRange
    intervals = zip([lo] + zs, zs + [hi])
    lines = zip(ms, cs)
    pieces = list(zip(intervals, lines))
    return pieces

def adaptiveRejectionSampling (h, nSamples, xRange) : 
    samples = []
    lo, hi = xRange 
    dConstant = 0.1
    if lo == -math.inf and hi == math.inf : 
        lo = -C
        while derivative(h, lo) <= dConstant : 
            lo *= C
        hi = C
        while derivative(h, hi) >= -dConstant : 
            hi *= C
    elif lo == -math.inf : 
        lo = -torch.abs(hi)
        while derivative(h, lo) <= dConstant : 
            lo *= C
    else : 
        hi = torch.abs(lo)
        while derivative(h, hi) >= -dConstant: 
            hi *= C

    xs = [t for t in torch.linspace(lo, hi, 10)]
    s = PiecewiseExponentialDistribution(pieces(xs, h, xRange))

    def lower_hull (x0) : 
        i = bisect_left(xs, x0)
        if i == 0 or i == len(xs): 
            return -INF
        else : 
            x, x_ = xs[i-1], xs[i]
            y, y_ = h(x), h(x_)
            return ((x_ - x0) * y + (x0 - x) * y_) / (xs[i] - xs[i - 1])

    while len(samples) < nSamples : 
        w = torch.rand([])
        xStar = s.sample()
        if w <= torch.exp(lower_hull(xStar) - s.logP(xStar)): 
            samples.append(xStar)
        else : 
            if w <= torch.exp(h(xStar) - s.logP(xStar)):
                samples.append(xStar)
            insort(xs, xStar)
            s = PiecewiseExponentialDistribution(pieces(xs, h, xRange))
    return samples

def gibbs (x, samplers, nSteps, **kwargs) : 
    for i in tqdm(range(nSteps)) : 
        for k, v in x.items() : 
            x[k] = samplers[k](**x, **kwargs)
    return x

