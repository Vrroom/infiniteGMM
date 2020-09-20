import torch
from bisect import bisect_left

torch.manual_seed(0)

def slope (p1, p2) :
    x1, y1 = p1
    x2, y2 = p2
    return (y2 - y1) / (x2 - x1)

def intercept (p1, p2) :
    x1, y1 = p1
    m = slope(p1, p2)
    return y1 - m * x1

class PiecewiseExponentialDistribution () :

    def __init__ (self, pieces) : 
        """
        Constructor.

        ln u(x) is piecewise linear. pieces
        is a list of ((x_l, x_r), (m, c)) where 
        m and c define the line y = m*x + c on the
        interval [x_l, x_r).
        """
        self.max = max([max(m*xl+c, m*xr+c) for ((xl, xr), (m, c)) in pieces])
        self.pieces = [(interval, (m, c - self.max)) for (interval, (m, c)) in pieces]
        self.weights = [self.integrate(piece) for piece in self.pieces]
        
    def integrate (self, piece) : 
        """
        Find the area under the curve for the piece.
        """
        ((x_l, x_r), (m, c)) = piece
        return (torch.exp(m * x_r + c) - torch.exp(m * x_l + c)) / m

    def inverseCDF (self, y) : 
        """
        Calculate the inverse CDF within a piece.
        """
        cumsum = torch.cumsum(torch.tensor(self.weights), 0).tolist()
        i = bisect_left(cumsum, y)
        ((x_l, x_r), (m, c)) = self.pieces[i]
        y = y - (cumsum[i - 1] if i > 0 else 0)
        return (torch.log(m * y + torch.exp(m * x_l + c)) - c) / m

    def logP (self, x) : 
        """
        Return the probability of x under this distribution.
        """
        (_, (m, c)), *_ = [((x_l, x_r), _)for ((x_l, x_r), _) in self.pieces if x_l <= x <= x_r]
        return m * x + c + self.max

    def sample (self) : 
        """
        Sample a point from the distribution.
        """
        totalWeight = sum(self.weights)
        u = totalWeight * torch.rand([])
        return self.inverseCDF(u)



