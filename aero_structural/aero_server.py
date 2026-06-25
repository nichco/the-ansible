import philote_mdo.general as pmdo
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from modopt import JaxProblem, SLSQP
from combo import combo
from aero import LiftingLine

# aero problem setup
N = 31
b = 10.0
c_root = 1.0
c_tip = 0.65
rho_atm = 1.225
v_inf = 60
lifting_line = LiftingLine(N, b, c_root, c_tip, v_inf, rho_atm)
tip_disp_target = 0.1

# constraint scalers
lw_scale = 1e-4
f_scale = 2e-4
disp_scale = 3e-1

class Aero(pmdo.ExplicitDiscipline):

    def setup(self):
        self.add_input("x", shape=(4,))  # input variables
        self.add_input("y", shape=(2,))  # lagrange multipliers
        self.add_input("mu", shape=(1,)) # penalty parameter(s)
        self.add_input("p", shape=(3,))  # input parameters

        self.add_output("x", shape=(4,)) # output variables
        self.add_output("p", shape=(3,)) # output parameters

    def compute(self, inputs, outputs):
        x_init = inputs["x"]
        y = inputs["y"]
        mu = inputs["mu"]
        params = inputs["p"]

        print('x_init: ', x_init)

        input_twist = x_init[0:N]
        input_thickness = x_init[N:2 * N - 1]
        input_f_hat = x_init[2 * N - 1:]
        v0 = input_twist

        input_right_tip_disp = params[0]
        input_left_tip_disp = params[1]
        input_weight = params[2]
        input_cd = params[3]
        input_lift = params[4]
        input_f = params[4:]

        def jax_obj(v):

            coef = lifting_line.solve_lifting_line_model(v)
            CD = lifting_line.compute_drag(coef)
            f = lifting_line.compute_forces(coef)
            f = jnp.linalg.norm(f, axis=1)
            CL = lifting_line.compute_lift_coefficient(coef)
            lift = 0.5 * rho_atm * v_inf**2 * CL * lifting_line.S
            
            # compute the global constraints
            f_con = (input_f_hat - f) * f_scale
            l_equals_w = (lift - input_weight) * lw_scale
            right_disp_con = (input_right_tip_disp - tip_disp_target) * disp_scale
            left_disp_con = (input_left_tip_disp - tip_disp_target) * disp_scale
            c = jnp.concatenate([f_con, jnp.array([right_disp_con, left_disp_con, l_equals_w])])

            return 1e3 * CD + y.T @ c + 0.5 * c.T @ jnp.diag(mu) @ c
        
        x_scaler = np.ones(N) * 10 # twist scaler

        jaxprob = JaxProblem(x0=v0, jax_obj=jax_obj, x_scaler=x_scaler)
        optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        # optimizer.print_results()
        twist_solution = optimizer.results['x'] / x_scaler

        output_coef = lifting_line.solve_lifting_line_model(twist_solution)
        output_cd = lifting_line.compute_drag(output_coef)
        output_f = lifting_line.compute_forces(output_coef)
        output_f = jnp.linalg.norm(output_f, axis=1)
        output_cl = lifting_line.compute_lift_coefficient(output_coef)
        output_lift = 0.5 * rho_atm * v_inf**2 * output_cl * lifting_line.S

        outputs["x"] = np.concatenate([twist_solution, 
                                       input_thickness, 
                                       input_f_hat
                                       ])
        
        outputs["p"] = np.array([input_right_tip_disp, 
                                 input_left_tip_disp, 
                                 input_weight, 
                                 output_cd, 
                                 output_lift, 
                                 output_f
                                 ])



if __name__ == "__main__":
    from concurrent import futures
    import grpc
    import philote_mdo.general as pmdo

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    discipline = pmdo.ExplicitServer(discipline=Aero())
    discipline.attach_to_server(server)

    server.add_insecure_port("[::]:50052")
    server.start()
    print("Server started. Listening on port 50052.")
    server.wait_for_termination()