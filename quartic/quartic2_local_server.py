import philote_mdo.general as pmdo
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from modopt import JaxProblem, SLSQP
from combo import combo


class Quartic2(pmdo.ExplicitDiscipline):
    """
    Basic ALBCD subproblem template.
    This subproblem minimizes the Quartic function with respect to x1.
    """

    def setup(self):
        self.add_input("x", shape=(4,)) # variables
        self.add_input("y", shape=(2,)) # lagrange multipliers
        self.add_input("mu", shape=(1,)) # penalty parameter(s)

        self.add_output("x", shape=(4,)) # variables

    def compute(self, inputs, outputs):
        x_init = inputs["x"]
        y = inputs["y"]
        mu = inputs["mu"]

        print('x_init: ', x_init)

        x1_1 = x_init[0]
        x2_1 = x_init[1]
        x1_2 = x_init[2]
        x2_2 = x_init[3]

        v0 = np.array([x1_2, x2_2])

        def jax_obj(v):
            x1_2 = v[0]
            x2_2 = v[1]
            obj = jnp.squeeze(x1_2**2 + x2_2**2 - 1.5 * x1_2 * x2_2)

            c_1 = combo([x1_1, x1_2])
            c_2 = combo([x2_1, x2_2])
            c = jnp.concatenate([c_1, c_2])

            return jnp.squeeze(obj + y.T @ c + 0.5 * mu * jnp.sum(c**2))
        
        def jax_con(v):
            x1_2, x2_2 = v[0], v[1]
            con = x1_2**2 + x2_2**2
            return con.flatten()
        
        jaxprob = JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0.5**2, cu=np.inf)

        optimizer = SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
        optimizer.solve()
        optimizer.print_results()
        ans = optimizer.results['x']

        print('outputs: ', np.array([x1_1, x2_1, ans[0], ans[1]]))
        outputs["x"] = np.array([x1_1, x2_1, ans[0], ans[1]])



if __name__ == "__main__":
    from concurrent import futures
    import grpc
    import philote_mdo.general as pmdo

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    discipline = pmdo.ExplicitServer(discipline=Quartic2())
    discipline.attach_to_server(server)

    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()