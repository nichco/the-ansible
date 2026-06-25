import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv

class LiftingLine:
    def __init__(self, N, b, c_root, c_tip, v_inf, rho, cl_alpha=2*jnp.pi):

        self.N = N
        self.alpha = 0.0
        self.b = b
        self.S = 0.5*self.b*(c_root + c_tip)
        self.AR = self.b**2/self.S
        self.cl_alpha = cl_alpha
        self.v_inf = v_inf
        self.rho = rho
        self.c_root = c_root
        self.c_tip = c_tip
        delta = 0.5*jnp.pi/self.N
        self.theta = jnp.linspace(delta, jnp.pi - delta, self.N)
        self.y = 0.5*self.b*jnp.cos(self.theta)
        u = jnp.absolute(2*self.y/self.b)
        self.chord = (1.0 - u)*c_root + u*c_tip
        self.n_idx = jnp.arange(1, self.N + 1)
        self.sin_mat = jnp.sin(self.n_idx[None, :] * self.theta[:, None])

    def solve_lifting_line_model(self, x):
        # A = jnp.zeros((self.N, self.N), dtype=x.dtype)
        # b = jnp.zeros(self.N)

        # for m in range(self.N):
        #     for n in range(self.N):
        #         cval = 0.25*(self.chord[m]/self.b)*self.cl_alpha
        #         A = A.at[m, n].set(
        #             jnp.sin((n+1)*self.theta[m])*jnp.sin(self.theta[m]) +
        #             # cval*jnp.sin((n+1)*self.theta[m])
        #             (n+1)*cval*jnp.sin((n+1)*self.theta[m])
        #         )

        #     b = b.at[m].set(
        #         0.25*(self.chord[m]/self.b)*jnp.sin(self.theta[m])*self.cl_alpha*(self.alpha + x[m])
        #     )

        n_vec = jnp.arange(1, self.N + 1)
        mu    = (0.25 * self.chord / self.b * self.cl_alpha)
        A     = self.sin_mat * (jnp.sin(self.theta)[:, None]
                                + n_vec[None, :] * mu[:, None])
        b     = mu * jnp.sin(self.theta) * (self.alpha + x)

        return jnp.linalg.solve(A, b)

    def compute_drag_coefficient(self, coef):

        # s = 0.0
        # for n in range(self.N):
        #     s += (n+1)*coef[n]**2
        # return jnp.pi*self.AR*s

        return jnp.pi * self.AR * jnp.dot(self.n_idx * coef, coef)

    def compute_lift_coefficient(self, coef):
        return jnp.pi * self.AR * coef[0]
    
    def circulation(self, coef):

        # Gamma = jnp.zeros(self.N)
        # for n in range(self.N):
        #     Gamma = Gamma + 2.0 * self.b * self.v_inf * coef[n] * jnp.sin((n + 1) * self.theta)
        # return Gamma

        return 2.0 * self.b * self.v_inf * (self.sin_mat @ coef)
    
    def compute_forces(self, coef):
        """
        Returns panel forces, shape (N, 3) [F_x, F_y, F_z].
        N collocation points = N panels (each point is a panel center).
        F_x: induced drag (streamwise, positive downstream)
        F_y: zero
        F_z: lift (normal)
        """
        # coef = self.solve_lifting_line_model(x)
        Gamma = self.circulation(coef) # Actual circulation Gamma = 2 b v_inf sum A_n sin(n theta)

        # # Induced angle of attack: αᵢ = Σ n Aₙ sin(nθ)/sin(θ)
        # alpha_i = jnp.zeros(self.N)
        # for n in range(self.N):
        #     alpha_i = (alpha_i
        #             + (n + 1) * coef[n]
        #             * jnp.sin((n + 1) * self.theta)
        #             / sin_theta)
        # Induced angle of attack: alpha_i = sum n A_n sin(n theta)/sin(theta) 
        alpha_i = jnp.zeros(self.N) 
        for n in range(self.N): 
            alpha_i = (alpha_i + (n + 1) * coef[n] 
                       * jnp.sin((n + 1) * self.theta) 
                       / jnp.sin(self.theta))

        # Induced downwash velocity w_i = -v_inf alpha_i (negative = downward)
        w_i = -self.v_inf * alpha_i

        # Panel widths: N+1 boundaries and N strips
        theta_bnd = jnp.linspace(0.0, jnp.pi, self.N + 1)
        y_bnd     = 0.5 * self.b * jnp.cos(theta_bnd)
        delta_y   = jnp.abs(jnp.diff(y_bnd))          # shape (N,)

        # Kutta-Joukowski: dF = rho (V_eff * Gamma y_hat) dy
        F_x = -self.rho * Gamma * w_i * delta_y   # induced drag (positive downstream)
        F_y =  jnp.zeros(self.N)                  # always zero
        F_z =  self.rho * self.v_inf * Gamma * delta_y   # lift

        return jnp.stack([F_x, F_y, F_z], axis=1)   # (N, 3)


    def plot_3d(self, x, plotter, cmap="viridis", disp=None):

        coef   = np.array(self.solve_lifting_line_model(x))
        y_span = np.array(self.y)

        Gamma = self.circulation(coef)

        u = np.abs(2.0 * y_span / float(self.b))
        chord_span = (1.0 - u) * float(self.c_root) + u * float(self.c_tip)

        disp = np.array(disp[:, :3]) if disp is not None else np.zeros((self.N, 3))

        pts, scalars, centers = [], [], []
        for j in range(self.N):
            x_le = -0.25 * chord_span[j] + disp[j, 0]
            x_te =  0.75 * chord_span[j] + disp[j, 0]
            y_j  = y_span[j]             + disp[j, 1]
            z_j  =                         disp[j, 2]
            pts.append([x_le, y_j, z_j])
            pts.append([x_te, y_j, z_j])
            scalars.extend([Gamma[j], Gamma[j]])
            centers.append(np.array([0.5*(x_le + x_te), y_j, z_j])) # for arrows

        pts     = np.array(pts)
        scalars = np.array(scalars)

        faces = []
        for j in range(self.N - 1):
            faces.extend([4, 2*j, 2*j+1, 2*j+3, 2*j+2])

        mesh = pv.PolyData(pts, np.array(faces))
        mesh.point_data["Gamma"] = scalars
        plotter.add_mesh(mesh, scalars="Gamma", cmap=cmap, show_edges=True)


        forces = np.array(self.compute_forces(coef)) # (N,3) for arrows
        mag = np.linalg.norm(forces, axis=1)

        cloud = pv.PolyData(np.array(centers))
        cloud["forces"] = forces

        glyphs = cloud.glyph(
            orient="forces",
            scale="forces",
            factor=0.2 * self.b / (np.max(mag) + 1e-12)
        )

        plotter.add_mesh(glyphs, color="red")



if __name__ == "__main__":

    N = 31
    b = 15.0
    c_root = 1.0
    c_tip = 0.65
    v_inf = 100.0
    rho = 1.225

    lifting_line = LiftingLine(N, b, c_root, c_tip, v_inf, rho)

    x = jnp.ones(N) * jnp.deg2rad(5)

    coef = lifting_line.solve_lifting_line_model(x)

    CD = lifting_line.compute_drag_coefficient(coef)
    print('CD: ', CD)

    Gamma = lifting_line.circulation(coef)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(lifting_line.y, x, linewidth=2)
    ax[0].set_title('Twist')
    ax[1].plot(lifting_line.y, Gamma, linewidth=2)
    ax[1].set_title('Gamma')
    plt.show()


    plotter = pv.Plotter()
    lifting_line.plot_3d(x, plotter)
    plotter.view_isometric()
    plotter.show()

    forces = lifting_line.compute_forces(coef)
    print('Forces (F_x, F_y, F_z) at each panel: \n', forces)

    q_inf      = 0.5 * rho * v_inf**2
    CL_series  = lifting_line.compute_lift_coefficient(coef)
    CL_panels  = jnp.sum(forces[:, 2]) / (q_inf * lifting_line.S)
    CDi_series = lifting_line.compute_drag_coefficient(coef)
    CDi_panels = jnp.sum(forces[:, 0]) / (q_inf * lifting_line.S)

    print(f"CL  — series: {CL_series:.4f}  |  panels: {CL_panels:.4f}")
    print(f"CDi — series: {CDi_series:.4f}  |  panels: {CDi_panels:.4f}")