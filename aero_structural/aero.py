import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv


class LiftingLine:
    def __init__(self, le, te, v_inf, rho, cl_alpha=2 * jnp.pi):
        """
        Parameters
        ----------
        le : np.ndarray, shape (M, 3)
            Leading-edge points, ordered left wingtip → right wingtip.
        te : np.ndarray, shape (M, 3)
            Trailing-edge points at each LE station.
        v_inf     : float   Free-stream velocity.
        rho       : float   Air density.
        cl_alpha  : float   Lift-curve slope.
        """
        self.N = le.shape[0]

        y_mesh = le[:, 1]
        self.b = y_mesh[-1] - y_mesh[0]

        chord_mesh = jnp.linalg.norm(te - le, axis=1)
        panel_areas = 0.5 * (chord_mesh[:-1] + chord_mesh[1:]) * jnp.abs(jnp.diff(y_mesh))
        self.S = jnp.sum(panel_areas)
        self.AR = self.b ** 2 / self.S

        self.cl_alpha = cl_alpha
        self.v_inf = v_inf
        self.rho = rho

        delta = 0.5 * jnp.pi / self.N
        self.theta = jnp.linspace(delta, jnp.pi - delta, self.N)
        self.y = 0.5 * self.b * jnp.cos(self.theta)

        self.chord = jnp.interp(self.y, y_mesh, chord_mesh)
        le_x_mesh = le[:, 0]
        self._le_x = jnp.interp(self.y, y_mesh, le_x_mesh)

        self.alpha = 0.0

        # ── Precompute all geometry-dependent constants once ──────────────────
        #
        # These arrays are fixed for a given wing geometry and do NOT change
        # between solver calls. Computing them here avoids redundant work on
        # every call to solve_lifting_line_model.

        n_idx   = jnp.arange(1, self.N + 1)                            # (N,)

        sin_th  = jnp.sin(self.theta)                                   # (N,)

        # sin_mat[m, n] = sin((n+1) * theta[m])                         # (N, N)
        sin_mat = jnp.sin(n_idx[None, :] * self.theta[:, None])

        # nsin_mat[m, n] = (n+1) * sin((n+1) * theta[m])               # (N, N)
        # Used for the induced-angle sum: alpha_i = nsin_mat @ coef / sin_th
        nsin_mat = sin_mat * n_idx[None, :]

        # cval[m] = 0.25 * chord[m] / b * cl_alpha                      # (N,)
        cval = 0.25 * (self.chord / self.b) * self.cl_alpha

        # Cosine-spaced panel widths                                     # (N,)
        theta_bnd = jnp.linspace(0.0, jnp.pi, self.N + 1)
        delta_y   = jnp.abs(jnp.diff(0.5 * self.b * jnp.cos(theta_bnd)))

        # Expose for external use / plot_3d
        self._n_idx    = n_idx
        self._sin_th   = sin_th
        self._sin_mat  = sin_mat
        self._nsin_mat = nsin_mat
        self._cval     = cval
        self._delta_y  = delta_y
        

    def solve_lifting_line_model(self, x):
        """Solve the lifting-line system for twist distribution x (radians)."""
        sin_mat  = self._sin_mat
        nsin_mat = self._nsin_mat
        sin_th   = self._sin_th
        n_idx    = self._n_idx
        cval     = self._cval
        delta_y  = self._delta_y

        # ── System matrix (N×N) ──────────────────────────────────────────────
        # A[m, n] = sin_mat[m, n] * (sin_th[m] + (n+1) * cval[m])
        A = sin_mat * (sin_th[:, None] + n_idx[None, :] * cval[:, None])

        # ── RHS vector (N,) ──────────────────────────────────────────────────
        b_vec = cval * sin_th * (self.alpha + x)

        # ── Solve for Fourier coefficients ────────────────────────────────────
        coef = jnp.linalg.solve(A, b_vec)

        # ── Aerodynamic coefficients ──────────────────────────────────────────
        CL = jnp.pi * self.AR * coef[0]
        CD = jnp.pi * self.AR * jnp.dot(n_idx * coef, coef)   # Σ (n+1)*An²

        # ── Circulation distribution (N,) ─────────────────────────────────────
        # Gamma[m] = 2 b V∞ Σ_n coef[n] sin((n+1)θ_m)  =  2 b V∞ (sin_mat @ coef)
        Gamma = 2.0 * self.b * self.v_inf * (sin_mat @ coef)

        # ── Induced angle & downwash (N,) ─────────────────────────────────────
        # alpha_i[m] = Σ_n (n+1) coef[n] sin((n+1)θ_m) / sin(θ_m)
        #            = (nsin_mat @ coef) / sin_th
        alpha_i = (nsin_mat @ coef) / sin_th
        w_i     = -self.v_inf * alpha_i

        # ── Panel forces (N, 3) ───────────────────────────────────────────────
        F_x = -self.rho * Gamma * w_i * delta_y
        F_y = jnp.zeros(self.N)
        F_z =  self.rho * self.v_inf * Gamma * delta_y

        return {
            "coef":  coef,
            "CD":    CD,
            "CL":    CL,
            "Gamma": Gamma,
            "F":     jnp.stack([F_x, F_y, F_z], axis=1),
        }


    def plot_3d(self, Gamma, forces, plotter, cmap="viridis", disp=None):
        y_span     = np.array(self.y)
        chord_span = np.array(self.chord)
        le_x       = self._le_x

        disp = np.array(disp[:, :3]) if disp is not None else np.zeros((self.N, 3))

        pts, scalars, centers = [], [], []
        for j in range(self.N):
            x_le = le_x[j]                 + disp[j, 0]
            x_te = le_x[j] + chord_span[j] + disp[j, 0]
            y_j  = y_span[j]               + disp[j, 1]
            z_j  =                           disp[j, 2]
            pts.append([x_le, y_j, z_j])
            pts.append([x_te, y_j, z_j])
            scalars.extend([Gamma[j], Gamma[j]])
            centers.append(np.array([0.5 * (x_le + x_te), y_j, z_j]))

        pts     = np.array(pts)
        scalars = np.array(scalars)

        faces = []
        for j in range(self.N - 1):
            faces.extend([4, 2*j, 2*j+1, 2*j+3, 2*j+2])

        mesh = pv.PolyData(pts, np.array(faces))
        mesh.point_data["Gamma"] = scalars
        plotter.add_mesh(mesh, scalars="Gamma", cmap=cmap, show_edges=True)

        mag   = np.linalg.norm(forces, axis=1)
        cloud = pv.PolyData(np.array(centers))
        cloud["forces"] = forces
        glyphs = cloud.glyph(
            orient="forces",
            scale="forces",
            factor=0.2 * self.b / (np.max(mag) + 1e-12),
        )
        plotter.add_mesh(glyphs, color="red")


if __name__ == "__main__":
    import time

    v_inf = 200.0
    rho   = 0.5

    from mesh import build_crm_mesh

    ns       = 33
    crm_mesh = build_crm_mesh(ns=ns, span_cos_spacing=0)
    le_pts   = crm_mesh[0, :, :]
    te_pts   = crm_mesh[1, :, :]

    lifting_line = LiftingLine(le_pts, te_pts, v_inf, rho)
    twist = jnp.ones(lifting_line.N) * jnp.deg2rad(5)

    t0  = time.perf_counter()
    sol = lifting_line.solve_lifting_line_model(twist)
    t1  = time.perf_counter()
    print(f"Solve time: {(t1 - t0)*1e3:.1f} ms")

    print("CL:", sol["CL"])
    print("CD:", sol["CD"])

    Gamma  = sol["Gamma"]
    forces = sol["F"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(lifting_line.y, twist, linewidth=2);  ax[0].set_title("Twist")
    ax[1].plot(lifting_line.y, Gamma, linewidth=2);  ax[1].set_title("Gamma")
    plt.show()

    plotter = pv.Plotter()
    lifting_line.plot_3d(Gamma, forces, plotter)
    plotter.view_isometric()
    plotter.show()

    q_inf      = 0.5 * rho * v_inf ** 2
    CL_panels  = jnp.sum(forces[:, 2]) / (q_inf * lifting_line.S)
    CDi_panels = jnp.sum(forces[:, 0]) / (q_inf * lifting_line.S)
    CL_series  = sol["CL"]
    CDi_series = sol["CD"]

    print(f"CL  — series: {CL_series:.4f}  |  panels: {CL_panels:.4f}")
    print(f"CDi — series: {CDi_series:.4f}  |  panels: {CDi_panels:.4f}")