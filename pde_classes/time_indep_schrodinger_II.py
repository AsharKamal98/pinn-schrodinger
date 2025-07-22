import torch
import numpy as np
import math
from scipy.special import hermite, factorial

class tiSchrodingerI(torch.nn.Module):
    def __init__(self, device, num_points, state):
        super().__init__()

        self.device = device

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

        n = state
        self.n = state
        
        # boundaries
        x_boundary = [0.0, 1.0]
        u_boundary = [0.0, 0.0]

        self.x_bc = torch.tensor([[x_boundary[0]], [x_boundary[1]]]).to(device)
        self.u_bc = torch.tensor([[u_boundary[0]], [u_boundary[1]]]).to(device)
        self.x_interior = torch.linspace(x_boundary[0], x_boundary[1], num_points).reshape(-1,1).to(device)

        # points for testing
        num_points_testing = 200
        self.x_test = torch.linspace(x_boundary[0], x_boundary[1], num_points_testing).reshape(-1, 1).to(device)
        # true u for test points
        # self.u_true = np.sqrt(2) * np.sin(n * self.x_test.cpu().numpy() * np.pi)
        
        # Hermite polynomial H_n(x)
        Hn = hermite(n)
        # Analytical solution: psi_n(x)
        x_test = self.x_test.cpu().numpy()
        self.u_true = (1.0 / np.sqrt(2**n * factorial(n))) * (1 / np.pi**0.25) * Hn(x_test) * np.exp(-x_test**2 / 2)
        
        # make E learnable, init near n²π²
        E0 = (n**2) * math.pi**2
        #E0 = 1.0
        self.E = torch.nn.Parameter(torch.tensor(E0, dtype=torch.float32, device=device))
        #self.E = n**2 * np.pi**2
        
        self.x_t = np.linspace(0,1,10)
        self.y = np.sqrt(2) * np.sin(self.x_t * np.pi)

        #self.E = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))

    
    def forward(self, x):
        return self.net(x)
    
    
    def V(self, x):
        return 0.5 * x**2
    
    def pde_residual(self, x_interior):
        #x = self.x_interior.clone().detach().requires_grad_(True)
        x = x_interior.clone().detach().requires_grad_(True)
        #x = x_interior.requires_grad(True)
        
        u = self.forward(x)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
        #E = (np.pi)**2

        Vx = self.V(x)
        return -d2u + Vx * u - self.E * u
    
    def loss(self, x):
        # loss parameters
        lam1 = 50
        lam2 = 1
        k = 15

        # Loss: PDE and boundary conditions
        res = self.pde_residual(x)
        loss_pde = torch.mean(res**2)

        u_pred_bc = self(self.x_bc)
        loss_bc = torch.mean((u_pred_bc - self.u_bc)**2)

        norm = torch.mean(self(x)**2)
        loss_norm = torch.exp(k * (norm - 1.0)**2) - 1.0
    
        # x_anchor = torch.tensor([[0.5]], device=device)
        # val = np.sqrt(2) * np.sin(self.n * np.pi * 0.5)
        # u_anchor = torch.tensor([[val]], dtype=torch.float32, device=device)
        # loss_anchor = torch.mean((pinn(x_anchor) - u_anchor) ** 2)

        

        loss = loss_pde + lam1*loss_bc + lam2*loss_norm
        #loss = loss_pde + lam1*loss_bc + lam2*loss_norm
        return loss