
import os
import argparse
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import torch.optim as optim
from torchdiffeq._impl.adjoint import OdeintAdjointMethod

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args()

class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, z):

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

        return dz_dt


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.
    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


def get_batch(num_samples):
    x = np.random.rand(num_samples * 2) * 2 - 1
    x = x.reshape((num_samples,2))

    x = torch.from_numpy(x).type(torch.float32).to(device)

    return x

if __name__ == '__main__':

    asd1 = torch.eye(3, requires_grad=True).unsqueeze(0).repeat(3,1,1)
    asd2 = torch.eye(3, requires_grad=True).unsqueeze(0).repeat(3,1,1)
    asd3 = torch.bmm(asd1, asd2)
    asd3 = asd3[-1]


    t0 = 0
    t1 = 10
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)
    optimizer = optim.Adam(func.parameters(), lr=args.lr)

    lossf = nn.MSELoss

    for itr in range(1, 100 + 1):
        optimizer.zero_grad()

        x = get_batch(args.num_samples)
        x.requires_grad = True
        z_t = odeint(
            func,
            x,
            torch.tensor([t1, t0]).type(torch.float32).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

        J = torch.zeros((z_t[0].shape[0], z_t[0].shape[1], z_t[0].shape[1]))
        for i in range(2):
            dzt_dx = torch.autograd.grad(z_t[-1][:, i].sum(), x, retain_graph=True, create_graph=True)[0].contiguous()
            J[:, i, :] = dzt_dx

        dets = torch.linalg.det(J)

        loss = ((dets - torch.ones_like(dets) * 2) ** 2).mean()

        print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss))

        loss.backward()
        optimizer.step()


    amt_viz = 50

    x = get_batch(amt_viz)

    viz_timesteps = 10

    z_t = odeint(
        func,
        x,
        torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
        atol=1e-5,
        rtol=1e-5,
        method='dopri5',
    )

    paths = []

    z_t = z_t.detach().numpy()

    for k in range(amt_viz):
        path = []
        for i in range(viz_timesteps):
            path.append(z_t[i,k,:])
        path = np.vstack(path).T
        paths.append(path)

    for path in paths:
        plt.scatter(path[0,0], path[1,0])
        plt.plot(path[0], path[1])
    plt.show()
