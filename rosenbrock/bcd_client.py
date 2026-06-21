import grpc
import numpy as np
from philote_mdo.general import ExplicitClient
from typing import List
import time

# client1 = ExplicitClient(channel=grpc.insecure_channel("localhost:50052"))
# client2 = ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

# # transfer the stream options to the server
# client1.send_stream_options()
# client2.send_stream_options()

# # run setup
# client1.run_setup()
# client1.get_variable_definitions()
# client2.run_setup()
# client2.get_variable_definitions()

# # define some inputs
# x = np.array([-1, -1])
# inputs = {"x": x}
# # outputs = {}

# # run a function evaluation
# outputs1 = client1.run_compute(inputs)
# outputs2 = client2.run_compute(inputs)

# print(outputs1)
# print(outputs2)






class BlockCoordinateDescent():
    def __init__(self, 
                 clients: List,            # Philote clients for each subproblem
                 x_init: List[np.ndarray], # initial variable values
                 eps: float = 1e-5,        # BCD convergence tolerance
                 ):

        self.clients = clients
        self.x = [np.asarray(xi) for xi in x_init] # cast to numpy arrays
        self.n = sum(xi.size for xi in self.x) # dimension
        # self.history = [self.x.copy()]
        # self.x_time = [0.0]
        self.t0 = None
        self.tf = None
        self.eps = eps

        # setup the clients
        for client in self.clients:
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()


    def solve(self, 
              max_iter: int=100, # max number of BCD iterations
              ) -> None:
        
        self.t0 = time.perf_counter()

        for i in range(max_iter):

            z_old = np.concatenate([xi.ravel() for xi in self.x])

            for client in self.clients:
                inputs = {"x": np.concatenate([xi.ravel() for xi in self.x])}
                outputs = client.run_compute(inputs)
                self.x = [np.asarray(outputs["x"][i]) for i in range(len(self.x))]

            z_new = np.concatenate([xi.ravel() for xi in self.x])
            step = abs(z_new - z_old)
            denom = np.maximum(abs(z_old), abs(z_new))
            denom = np.maximum(denom, 1e-5) # floor
            rel_step = max(step / denom)

            print(f"pr_itr={i:03d} | "f"rel_stp={rel_step:.3e} | ")

            if rel_step <= self.eps:
                print("Converged with rel step: ', rel_step, ' in ', i, ' iterations")
                break

        self.tf = time.perf_counter() - self.t0

        return None






if __name__ == "__main__":
    client1 = ExplicitClient(channel=grpc.insecure_channel("localhost:50052"))
    client2 = ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

    x_init = [np.array([-1.0]), np.array([-1.0])]
    opt = BlockCoordinateDescent(clients=[client1, client2], x_init=x_init, eps=1e-5)
    opt.solve(max_iter=100)
    print(opt.x)