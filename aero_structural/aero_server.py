import philote_mdo.general as pmdo
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from modopt import JaxProblem, SLSQP
from combo import combo


class Aero(pmdo.ExplicitDiscipline):

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

        
        outputs["x"] = np.array([])



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