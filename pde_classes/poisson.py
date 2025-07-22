import torch

class Poisson(torch.nn.Module):
    def __init__(self, device, num_points):
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
            torch.nn.Linear(40,1),
        ).to(device)
        
        # boundaries
        x_boundary = [0.0, 1.0]
        u_boundary = [0.0, 0.0]

        self.x_bc = torch.tensor([[x_boundary[0]], [x_boundary[1]]], requires_grad=False).to(device)
        self.u_bc = torch.tensor([u_boundary[0], u_boundary[1]]).to(device)
        self.x_interior = torch.linspace(x_boundary[0], x_boundary[1], num_points).reshape(-1,1).to(device)

        # points for testing
        self.x_test = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
        # true u for test points
        self.u_true = 0.5 * self.x_test.cpu().numpy() * (1 - self.x_test.cpu().numpy())
    
    def forward(self, x):
        return self.net(x)
    
    def pde_residual(self, x_interior):
        #x = self.x_interior.clone().detach().requires_grad_(True)
        x = x_interior.clone().detach().requires_grad_(True)
        
        u = self.forward(x)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
        
        # source term 
        f = torch.tensor(1.0).to(self.device)

        return d2u+f

    def loss(self, x):
        # loss PDE
        res = self.pde_residual(x)
        loss_pde = torch.mean(res**2)

        # loss boundary conditions
        u_pred_bc = self(self.x_bc)
        loss_bc = torch.mean((u_pred_bc - self.u_bc)**2)
        
        loss = loss_pde + loss_bc
        return loss