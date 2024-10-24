# src/api/server.py

import logging
import socket
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceMgr
from src.api.experiment_mgmt import ExperimentManager
from src.utils.ml_utils import DataUtils
from src.utils.system_utils import read_yaml_file, get_repo_root

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SERVER")

class Server:
    CONFIG_PATH: Path = get_repo_root() / "config" / "model_config.yaml"

    def __init__(self):
        self.config = read_yaml_file(self.CONFIG_PATH)
        self.device_mgr = DeviceMgr()
        self.experiment_mgr = ExperimentManager(self.CONFIG_PATH)
        self.data_utils = DataUtils()

    def start(self):
        server_devices = self.device_mgr.get_devices(available_only=True, device_type="SERVER")
        if not server_devices:
            logger.error("No available server devices found.")
            return

        server_device = server_devices[0]
        host = server_device.working_cparams.host
        port = self.config['experiment']['port']

        try:
            server_socket = self.device_mgr.create_server_socket(host, port)
            logger.info(f"Server is listening on {host}:{port}...")
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            return

        try:
            while True:
                conn, addr = server_socket.accept()
                logger.info(f"Connected by {addr}")
                self.handle_connection(conn)
        finally:
            server_socket.close()
            logger.info("Server socket closed.")

    def handle_connection(self, conn: socket.socket):
        try:
            experiment_config = self.data_utils.receive_data(conn)
            if experiment_config is None:
                logger.error("Received None experiment_config")
                return

            experiment = self.experiment_mgr.setup_experiment(experiment_config)

            while True:
                data = self.data_utils.receive_data(conn)
                if data is None:
                    logger.info("Client disconnected or sent None data")
                    break

                result = experiment.process_data(data)
                if result is None:
                    logger.warning("process_data returned None result")
                    result = {'error': 'Processing failed'}

                self.data_utils.send_result(conn, result)

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            conn.close()

if __name__ == "__main__":
    server = Server()
    server.start()
