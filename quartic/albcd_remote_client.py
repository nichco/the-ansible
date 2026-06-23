import grpc
import numpy as np
from philote_mdo.general import ExplicitClient
from typing import List, Callable
import time
from combo import combo
import matplotlib.pyplot as plt


class AugmentedLagrangianBlockCoordinateDescent():
    def __init__(self, 
                 clients: List,            # Philote clients for each subproblem
                 x_init: List[np.ndarray], # initial variable values
                 con: Callable = lambda v: np.zeros(0),
                 mu: np.ndarray | float = 1.0, # positive penalty parameter(s)
                 max_mu: float = 1e3, # maximum penalty parameter
                 rho: float = 1.2, # penalty increase factor
                 tau: float = 0.5, # factor for increasing mu based on constraint violation
                 tol: float = 1e-3, # outer loop feasibility tolerance
                 eps: float = 1e-2, # initial inner loop convergence tolerance
                 eta: float = 1e-5, # final inner loop convergence tolerance
                 max_y: float = 1e6, # maximum Lagrange multiplier value
                 ):

        self.clients = clients
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        # self.history = [self.x.copy()]
        # self.x_time = [0.0]
        self.tf = None
        self.con = con
        self.mu = np.asarray(mu) # penalty parameter(s)
        self.max_mu = max_mu
        self.rho = rho
        self.tau = tau
        self.tol = tol
        self.eps = eps
        self.eta = eta
        self.max_y = max_y
        self.history = [self.x.copy()]

        self.initial_constraint_values = self.con(self.x)
        self.y = np.zeros_like(self.initial_constraint_values) # Lagrange multipliers

        # setup the clients
        for client in self.clients:
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()


    def _update_mu(self, c_new, c_old) -> None:

        # if isinstance(self.mu, np.ndarray):

        #     for i, (f_new, f_old) in enumerate(zip(abs(c_new), abs(c_old))):
        #         if f_new > self.tau * f_old and f_new > self.tol:
        #             self.mu[i] = min(self.rho * self.mu[i], self.max_mu)

        # elif isinstance(self.mu, (float, int)):

        if max(abs(c_new)) > self.tau * max(abs(c_old)) and max(abs(c_new)) > self.tol:
            self.mu = min(self.rho * self.mu, self.max_mu)

        return None


    def solve(self, 
              max_outer_iter: int=100, # maximum number of outer iterations
              max_inner_iter: int=10,  # maximum number of inner iterations
              ) -> None:
        
        phase = 0 # two-phase inner loop tolerance
        t0 = time.perf_counter()

        # augmented Lagrangian outer loop
        for k in range(max_outer_iter):

            c_old = self.con(self.x)

            # block coordinate descent inner loop
            for j in range(max_inner_iter):

                z_old = np.concatenate([xi.ravel() for xi in self.x])

                # Gauss-Seidel loop
                for client in self.clients:
                    inputs = {"x": np.concatenate([xi.ravel() for xi in self.x]), "y": self.y, "mu": np.atleast_1d(self.mu)}
                    outputs = client.run_compute(inputs)
                    self.x = [np.asarray(outputs["x"][i]) for i in range(len(self.x))]
                    self.history.append(self.x.copy())

                z_new = np.concatenate([xi.ravel() for xi in self.x])
                step = abs(z_new - z_old)
                denom = np.maximum(abs(z_old), abs(z_new))
                denom = np.maximum(denom, 1e-8) # floor
                rel_step = max(step / denom)

                print(f"pr_itr={j:03d} | "f"rel_stp={rel_step:.3e} | ")

                if phase == 0 and rel_step <= self.eps:
                    print('-(Phase 0) primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                    break

                if phase == 1 and rel_step <= self.eta:
                    print('-(Phase 1) primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                    break

                # if rel_step <= self.eps:
                #     print('-Primal loop converged with rel step: ', rel_step, ' in ', j, ' iterations!-')
                #     break


            c_new = self.con(self.x)
            feas = np.max(abs(c_new))

            if feas <= self.tol and phase == 1:
                print('-Dual loop converged with feasibility: ', feas, ' in ', k, ' dual iterations!-')
                break

            if feas <= self.tol and phase == 0:
                print('-Phase 1 complete. Starting phase 2 with feasibility: ', feas)
                phase = 1

            # self.y += np.diag(self.mu) @ c_new # always update the multipliers
            # self.y_history.append(self.y.copy())
            # if isinstance(self.mu, np.ndarray):
            #     self.y += np.diag(self.mu) @ c_new
            # if isinstance(self.mu, (float, int)):
            self.y += self.mu * c_new

            # self.y_history.append(self.y)

            # update mu on a per-scalar-constraint basis
            self._update_mu(c_new, c_old)
            c_old = c_new

            print(f"du_itr={k:03d} | "
                  f"feas={feas:.3e} | "
                  f"max mu={np.max(self.mu):.3e} | "
                  f"min mu={np.min(self.mu):.3e} | "
                  f"y={np.linalg.norm(self.y):.3e} | "
                  )

        self.tf = time.perf_counter() - t0
        return None




def con(v_init):
    
    x1_1 = v_init[0]
    x2_1 = v_init[1]
    x1_2 = v_init[2]
    x2_2 = v_init[3]

    c_1 = combo([x1_1, x1_2])
    c_2 = combo([x2_1, x2_2])
    return np.concatenate([c_1, c_2])



if __name__ == "__main__":
    # client1 = ExplicitClient(channel=grpc.insecure_channel("localhost:50052"))
    # client2 = ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))
    client1 = ExplicitClient(channel=grpc.insecure_channel("192.168.0.43:50052"))
    client2 = ExplicitClient(channel=grpc.insecure_channel("192.168.0.43:50051"))

    x_init = [-0.5, 1.0, -0.5, 1.0]

    opt = AugmentedLagrangianBlockCoordinateDescent(clients=[client1, client2], 
                                                    x_init=x_init, 
                                                    con=con,
                                                    mu=10,#1,
                                                    max_mu=1e3,
                                                    rho=1.2,
                                                    tau=0.5,
                                                    tol=1e-4,
                                                    eps=1e-2, # initial inner loop convergence tolerance
                                                    eta=1e-5, # final inner loop convergence tolerance
                                                    max_y=1e6
                                                    )
    opt.solve(max_outer_iter=100,
              max_inner_iter=10,
              )
    print(opt.x)
    print('total time (s): ', opt.tf)

    x1_1_history = [h[0] for h in opt.history]
    x2_1_history = [h[1] for h in opt.history]
    x1_2_history = [h[2] for h in opt.history]
    x2_2_history = [h[3] for h in opt.history]


    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 - 1.5 * X * Y
    levels = np.linspace(0, max(Z.flatten()), 30)
    plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
    plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

    plt.plot(x1_1_history, x2_2_history, 's-', color='tab:orange', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_1, y_2)$')
    plt.plot(x1_2_history, x2_1_history, 'o-', color='tab:purple', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_2, y_1)$')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('y')

    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 * np.cos(theta)
    circle_y = 0.5 * np.sin(theta)
    plt.plot(circle_x, circle_y, '--', color='black', linewidth=2, alpha=0.5)
    plt.fill(circle_x, circle_y, color='black', alpha=0.3)

    ticks = [-1, 0, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.legend()
    plt.gca().set_aspect('equal')

    # plt.savefig('augmented_lagrangian_circle_constraint.pdf', bbox_inches='tight')
    plt.show()