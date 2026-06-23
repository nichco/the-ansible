import philote_mdo.general as pmdo
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from modopt import JaxProblem, SLSQP

class Rosenbrock2(pmdo.ExplicitDiscipline):
    """
    Basic ALBCD subproblem template.
    This subproblem minimizes the Rosenbrock function with respect to x2.
    """

    def setup(self):
        self.add_input("x", shape=(2,)) # variables
        # self.add_input("y", shape=(2,)) # lagrange multipliers
        # self.add_input("mu", shape=(2,)) # penalty parameter(s)

        self.add_output("x", shape=(2,)) # variables

    def compute(self, inputs, outputs):
        x_init = inputs["x"]
        # y = inputs["y"]
        # mu = inputs["mu"]

        print('x_init: ', x_init)

        x1 = x_init[0]
        x2 = x_init[1]
        v0 = np.atleast_1d(x2)

        def jax_obj(v):
            x2 = v[0]
            return jnp.squeeze((1 - x1)**2 + 1 * (x2 - x1**2)**2)
        
        jaxprob = JaxProblem(x0=v0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)
        optimizer = SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
        optimizer.solve()
        optimizer.print_results()

        ans = optimizer.results['x'][0]

        print('outputs: ', np.array([x1, ans]))

        outputs["x"] = np.array([x1, ans])



if __name__ == "__main__":
    from concurrent import futures
    import grpc
    import philote_mdo.general as pmdo

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    discipline = pmdo.ExplicitServer(discipline=Rosenbrock2())
    discipline.attach_to_server(server)

    # server.add_insecure_port("[::]:50051") # localhost only
    server.add_insecure_port('0.0.0.0:50052') # accepts connections from other machines
    server.start()
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()