import grpc
import numpy as np
from philote_mdo.general import ExplicitClient


client1 = ExplicitClient(channel=grpc.insecure_channel("localhost:50052"))
client2 = ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

# transfer the stream options to the server
client1.send_stream_options()
client2.send_stream_options()

# run setup
client1.run_setup()
client1.get_variable_definitions()
client2.run_setup()
client2.get_variable_definitions()

# define some inputs
x = np.array([-1, -1])
inputs = {"x": x}
outputs = {}

# run a function evaluation
outputs1 = client1.run_compute(inputs)
outputs2 = client2.run_compute(inputs)

print(outputs1)
print(outputs2)