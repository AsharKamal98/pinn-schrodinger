import torch
import numpy as np
import math
from scipy.special import hermite, factorial

class tiSchrodingerII(torch.nn.Module):
    def __init__(self, 
                 device, 
                 num_points, 
                 state=4,
                loss_pde_factor = 3,
                loss_bc_factor = 1,
                loss_norm_factor = 1,
                loss_norm_exp_factor = 1
        ):
        super().__init__()

        self.device = device
        n = state
        self.n = state

        # loss parameters
        self.loss_pde_factor = loss_pde_factor
        self.loss_bc_factor = loss_bc_factor
        self.loss_norm_factor = loss_norm_factor
        self.loss_norm_exp_factor = loss_norm_exp_factor

        # neural network architecture
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,1),
        ).to(device)

        
        # boundaries
        x_boundary = [-5.0, 5.0]
        u_boundary = [0.0, 0.0]

        self.x_bc = torch.tensor([[x_boundary[0]], [x_boundary[1]]]).to(device)
        self.u_bc = torch.tensor([[u_boundary[0]], [u_boundary[1]]]).to(device)
        self.x_interior = torch.linspace(x_boundary[0], x_boundary[1], num_points).reshape(-1,1).to(device)
        self.dx = self.x_interior[1] - self.x_interior[0]

        # points for testing
        num_points_testing = 200
        self.x_test = torch.linspace(x_boundary[0], x_boundary[1], num_points_testing).reshape(-1, 1).to(device)
        # true u for test points
        Hn = hermite(n)
        x_test = self.x_test.cpu().numpy()
        self.u_true = (1.0 / np.sqrt(2**n * factorial(n))) * (1 / np.pi**0.25) * Hn(x_test) * np.exp(-x_test**2 / 2)
        
        # make E learnable, init near 1/2 * (2n + 1)
        #E0 = n + 0.5
        E0 = n+1/2
        #E0 = 5.5
        #self.E = torch.nn.Parameter(torch.tensor(E0, dtype=torch.float32, device=device))
        self.E = E0
    
    def forward(self, x):
        return self.net(x)
    
    def V(self, x):
        return 0.5 * x**2
    
    def pde_residual(self, x_interior):
        x = x_interior.clone().detach().requires_grad_(True)
        
        u = self.forward(x)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]

        Vx = self.V(x)
        return -0.5*d2u + Vx * u - self.E * u
    
    def loss(self, x):
        # loss PDE
        res = self.pde_residual(x)
        loss_pde = torch.mean(res**2)

        # loss boundary conditions
        u_pred_bc = self(self.x_bc)
        loss_bc = torch.mean((u_pred_bc - self.u_bc)**2)

        # loss normalization
        u_pred = self(x)
        norm = torch.sum(u_pred**2) * self.dx
        loss_norm = torch.exp(self.loss_norm_exp_factor * (norm - 1.0)**2) - 1.0

        loss_dict = {
            'loss_pde': self.loss_pde_factor * loss_pde,
            'loss_bc': self.loss_bc_factor * loss_bc,
            'loss_norm': self.loss_norm_factor * loss_norm,
            'norm': norm,
        }
        
        return self.loss_pde_factor*loss_pde + self.loss_bc_factor*loss_bc + self.loss_norm_factor*loss_norm, loss_dict
        #return self.loss_pde_factor*loss_pde + self.loss_bc_factor*loss_bc, loss_norm, norm