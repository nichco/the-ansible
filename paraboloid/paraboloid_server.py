from concurrent import futures
import grpc
import philote_mdo.general as pmdo
# from philote_mdo.examples import Paraboloid
from paraboloid import Paraboloid


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

discipline = pmdo.ExplicitServer(discipline=Paraboloid())
discipline.attach_to_server(server)

server.add_insecure_port("[::]:50051")
server.start()
print("Server started. Listening on port 50051.")
server.wait_for_termination()