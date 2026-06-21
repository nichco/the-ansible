import grpc
import numpy as np
from philote_mdo.general import ExplicitClient


client = ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

# transfer the stream options to the server
client.send_stream_options()

# run setup
client.run_setup()
client.get_variable_definitions()
client.get_partials_definitions()

# define some inputs
x = np.array([0, 0])
inputs = {"x": x}
outputs = {}

# run a function evaluation
outputs = client.run_compute(inputs)

print(outputs)