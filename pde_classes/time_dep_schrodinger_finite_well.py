import torch
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class tdSchrodinger_fw(torch.nn.Module):
    def __init__(self, 
                 device, 
                 num_x_points, 
                 num_t_points,
                 state=4,
                 loss_pde_factor = 1,
                 loss_bc_factor = 100,
                 loss_norm_factor = 1,
                 loss_norm_exp_factor = 10,
                 loss_init_factor = 1,
                 loss_energy_factor = 1,
                 V0=5000.0,
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
        self.loss_init_factor = loss_init_factor
        self.loss_energy_factor = loss_energy_factor

        # neural network architecture
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2,40), # input [x, t]
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,2), # output [real(psi), imag(psi)]
        ).to(device)

        # boundaries
        x_boundary = [-5.0, 5.0]
        t_boundary = [0.0, 1.0]
        u_boundary = [0.0, 0.0]
        self.well_edges = (-1.0, 1.0)
        self.V0 = V0

        # grid spaxing
        self.dx = (x_boundary[1] - x_boundary[0]) / (num_x_points - 1)

        # points for training
        x_boundary = torch.tensor(x_boundary).to(device)
        x_interior = torch.linspace(x_boundary[0], x_boundary[1], num_x_points).to(device)
        t_vals = torch.linspace(t_boundary[0], t_boundary[1], num_t_points).to(device)

        x_interior_grid, t_interior_grid = torch.meshgrid(x_interior, t_vals, indexing='ij')  # shape (num_x_points, num_t_points) each
        self.x_t_interior = torch.stack([x_interior_grid.flatten(), t_interior_grid.flatten()], dim=1).to(device)  # shape (num_x_points*num_t_points, 2)
        
        # boundary conditions for loss
        x_boundary_grid, t_boundary_grid = torch.meshgrid(x_boundary, t_vals, indexing='ij')  # shape (2, num_t_points) each    
        self.x_t_bc = torch.stack([x_boundary_grid.flatten(), t_boundary_grid.flatten()], dim=1).to(device) # shape (2*num_t_points, 2)
        self.u_bc = torch.zeros((2*num_t_points, 2), device=device)

        # x-t grid
        x_init_grid, t_init_grid = torch.meshgrid(x_interior, torch.tensor([t_boundary[0]]).to(device), indexing='ij')  # shape (num_x_points, 1)
        self.x_t_init = torch.stack([x_init_grid.flatten(), t_init_grid.flatten()], dim=1).to(device)  # shape (num_x_points, 2)
        
        # initial condition for loss
        # psi_init = torch.sqrt(torch.tensor(2.0, device=device)) * torch.sin(n * x_interior * math.pi).reshape(-1, 1)
        # self.u_init = torch.cat([psi_init, torch.zeros_like(psi_init)], dim=1)  # [Re, Im]

        x_interior_np = x_interior.cpu().numpy()
        psi_init_np = np.zeros_like(x_interior_np)
        inside = (x_interior_np > self.well_edges[0]) & (x_interior_np < self.well_edges[1])
        psi_init_np[inside] = np.sin(self.n * np.pi * (x_interior_np[inside] + 1) / 2)
        psi_init = torch.tensor(psi_init_np, device=self.device).reshape(-1, 1)
        self.u_init = torch.cat([psi_init, torch.zeros_like(psi_init)], dim=1)  # [Re, Im]

        # points for testing
        num_x_points_testing = 100
        num_t_points_testing = 100
        
        x_interior_test = torch.linspace(x_boundary[0], x_boundary[1], num_x_points_testing).to(device)
        t_vals_test = torch.linspace(t_boundary[0], t_boundary[1], num_t_points_testing).to(device)

        x_interior_test_grid, t_interior_test_grid = torch.meshgrid(x_interior_test, t_vals_test, indexing='ij')
        self.x_t_test = torch.stack([x_interior_test_grid.flatten(), t_interior_test_grid.flatten()], dim=1).to(device)
        # plotting points
        self.x_interior_test_grid = x_interior_test_grid
        self.t_interior_test_grid = t_interior_test_grid

        # # true u for test points
        # x_test = x_interior_test_grid.cpu().numpy().flatten()
        # t_test = t_interior_test_grid.cpu().numpy().flatten()
        
        # E_n = (1/8) * n**2 * np.pi**2
        # self.E_n = E_n
        # psi_true = np.sqrt(2) * np.sin(n * np.pi * x_test) * np.exp(-1j * E_n * t_test)
        # self.u_true = np.stack([np.real(psi_true), np.imag(psi_true)], axis=1)  # shape (N, 2)
        
        # make E learnable, init near n²π²
        #E0 = E_n
        #self.E = torch.nn.Parameter(torch.tensor(E0, dtype=torch.float32, device=device))
        #self.E = E0

        # compute E0 from analytic solution
        x0 = self.x_t_init[:, 0:1]
        psi0_re = torch.sin(self.n * math.pi * (x0 + 1) / 2)
        dpsi0_dx = self.n * math.pi / 2 * torch.cos(self.n * math.pi * (x0 + 1) / 2)

        energy_density0 = 0.5 * (dpsi0_dx**2)  # V=0
        E0 = (energy_density0 * self.dx).sum()
        self.register_buffer("E0", E0.detach())  # fixed, no grad
    
    
    def forward(self, x):
        return self.net(x)

    def energy_per_time(self, x):
        """ 
            compute energy per time step
        """

        dx = self.dx
        
        xg = x.detach().clone().requires_grad_(True)
        psi = self.forward(xg)
        u, v = psi[:, 0:1], psi[:, 1:2]

        du_dx = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        dv_dx = torch.autograd.grad(v, xg, torch.ones_like(v), create_graph=True)[0][:, 0:1]

        energy_density = 0.5 * (du_dx**2 + dv_dx**2)  # shape (N, 1)

        t_vals = xg[:, 1]   # all t's in the batch
        uniq_t, inv = torch.unique(t_vals, sorted=True, return_inverse=True)

        sumE_per_t = torch.zeros_like(uniq_t).scatter_add(0, inv, energy_density.view(-1))
        E_per_t = sumE_per_t * dx

        return uniq_t, E_per_t
    
    def V(self, x):
        # x: [N, 2], x[:,0] is spatial coordinate
        x_pos = x[:, 0]
        V = torch.where(
            (x_pos > self.well_edges[0]) & (x_pos < self.well_edges[1]),
            torch.zeros_like(x_pos),
            torch.full_like(x_pos, self.V0)
        )
        return V.unsqueeze(1)  # shape [N, 1]

    def pde_residual(self, x_interior):
        x = x_interior.clone().detach().requires_grad_(True)
        
        # psi = u + iv
        psi = self.forward(x)
        u, v = psi[:, 0:1], psi[:, 1:2]

        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
        dv_dx = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0][:, 0:1]
        d2v_dx2 = torch.autograd.grad(dv_dx, x, torch.ones_like(dv_dx), create_graph=True)[0][:, 0:1]

        du_dt = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0][:, 1:2]
        dv_dt = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0][:, 1:2]

        Vx = self.V(x)  # shape [N, 1]

        res_real  = dv_dt - (1/2) * d2u_dx2 + Vx * u
        res_imag = du_dt + (1/2) * d2v_dx2 - Vx * v

        return torch.cat([res_real, res_imag], dim=1)  # shape (N, 2)

    def loss(self, x):
        # PDE
        res = self.pde_residual(x)
        loss_pde = torch.mean(res**2)

        # boundary conditions
        u_pred_bc = self(self.x_t_bc)
        loss_bc = torch.mean((u_pred_bc - self.u_bc)**2)

        # initial conditions
        u_pred_init = self(self.x_t_init)
        loss_init = torch.mean((u_pred_init - self.u_init)**2)

        # loss normalization per time step
        psi_pred = self(x) # [N, 2]
        prob = (psi_pred**2).sum(dim=1) # |psi|^2 = u^2+v^2, shape [N]

        t_vals = x[:, 1]                                            # all t's in the batch
        uniq_t, inv = torch.unique(t_vals, sorted=True, return_inverse=True)
        sum_per_t = torch.zeros_like(uniq_t).scatter_add(0, inv, prob)  # sum_x |psi|^2 per t
        mean_prob_per_t = sum_per_t * self.dx  # approximate integral over x for each t

        # penalize deviation from 1 at each time, then average over t
        loss_norm = torch.mean(torch.exp(self.loss_norm_exp_factor * (mean_prob_per_t - 1.0)**2) - 1.0)

        # energy drift penalty
        uniq_t_energy, E_per_t = self.energy_per_time(x)
        #loss_energy = torch.mean((E_per_t - self.E0)**2 / (self.E0**2 + 1e-12))
        loss_energy = torch.mean((E_per_t - self.E0)**2)


        loss_dict = {
            'loss_pde': self.loss_pde_factor * loss_pde,
            'loss_bc': self.loss_bc_factor * loss_bc,
            'loss_norm': self.loss_norm_factor * loss_norm,
            'loss_init': self.loss_init_factor * loss_init,
            'loss_energy': self.loss_energy_factor * loss_energy,
            'norm': mean_prob_per_t.mean(),
            'E0': self.E0,
            'E': E_per_t.mean(),
        }

        #return self.loss_pde_factor * loss_pde + self.loss_bc_factor * loss_bc + self.loss_norm_factor * loss_norm + self.loss_init_factor * loss_init + self.loss_energy_factor * loss_energy, loss_dict
        return self.loss_pde_factor * loss_pde + self.loss_init_factor * loss_init + self.loss_norm_factor * loss_norm, loss_dict
    
    def predict(self):
        u_pred = self(self.x_t_test).detach().cpu().numpy()
        return u_pred
    
    def make_plot(self, u_pred):
        num_x_points_testing = 100
        num_t_points_testing = 100

        X = self.x_interior_test_grid.cpu().numpy()   # (Nx, Nt)
        T = self.t_interior_test_grid.cpu().numpy()   # (Nx, Nt)

        # --- Predicted ---
        u_pred_real = u_pred[:, 0].reshape(num_x_points_testing, num_t_points_testing)
        u_pred_imag = u_pred[:, 1].reshape(num_x_points_testing, num_t_points_testing)
        prob_pred = u_pred_real**2 + u_pred_imag**2

        # --- True ---
        # u_true_real = self.u_true[:, 0].reshape(num_x_points_testing, num_t_points_testing)
        # u_true_imag = self.u_true[:, 1].reshape(num_x_points_testing, num_t_points_testing)
        # prob_true = u_true_real**2 + u_true_imag**2

        # Axes
        x_vals = X[:, 0]
        t_vals = T[0, :]

        # --- Set up figure ---
        fig, ax = plt.subplots(figsize=(8, 5))
        # line_pred, = ax.plot([], [], lw=2, label="Predicted")  # solid line
        line_prob, = ax.plot([], [], lw=2, label="Pred |ψ|²", color='black')
        line_real, = ax.plot([], [], lw=2, label="Pred Re(ψ)", color='tab:blue')
        line_imag, = ax.plot([], [], lw=2, label="Pred Im(ψ)", color='tab:orange')

        title_template = f"State={self.n}   |   t={{t:.2f}}"
        ax.set_xlim(x_vals.min(), x_vals.max())
        y_max = max(prob_pred.max(), np.abs(u_pred_real).max(), np.abs(u_pred_imag).max()) * 1.1
        #ax.set_ylim(0, ymax)
        #ax.set_xlim(-3,3)
        ax.set_ylim(-y_max, y_max)
        ax.set_xlabel("x")
        #ax.set_ylabel("|ψ(x,t)|² (Probability Densityyy)")
        ax.set_ylabel("Value")
        title = ax.set_title("")
        #ax.legend()
        ax.legend(loc='upper right')


        # Init
        def init():
            #line_pred.set_data([], [])
            line_prob.set_data([], [])
            line_real.set_data([], [])
            line_imag.set_data([], [])
            title.set_text("")
            return line_prob, line_real, line_imag, title

        # Update
        def update(frame):
            # line_pred.set_data(x_vals, prob_pred[:, frame])
            line_prob.set_data(x_vals, prob_pred[:, frame])
            line_real.set_data(x_vals, u_pred_real[:, frame])
            line_imag.set_data(x_vals, u_pred_imag[:, frame])
            # line_true.set_data(x_vals, prob_true[:, frame])
            #title.set_text(f"|ψ|² at t={t_vals[frame]:.2f}")        
            title.set_text(title_template.format(t=t_vals[frame]))
            return line_prob, line_real, line_imag, title

        ani = animation.FuncAnimation(
            fig, update, frames=len(t_vals),
            init_func=init, blit=True, interval=100
        )

        plt.close(fig)  # prevent static plot in Jupyter
        return ani
