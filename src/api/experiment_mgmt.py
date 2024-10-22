# src/api/experiment_mgmt.py

import sys
from typing import Any, Dict
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from src.utils.system_utils import read_yaml_file
from src.api.device_mgmt import DeviceMgr
from src.utils.logger import setup_logger, DeviceType
from src.interface.bridge import ExperimentManagerInterface, ExperimentInterface

logger = setup_logger(device=DeviceType.SERVER)

class ExperimentManager(ExperimentManagerInterface):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.config = read_yaml_file(config_path)
        self.device_mgr = DeviceMgr()
        server_devices = self.device_mgr.get_devices(device_type="SERVER")
        if not server_devices:
            raise ValueError("No SERVER device found in the configuration")
        self.server_device = server_devices[0]
        self.host = self.server_device.working_cparams.host if self.server_device.working_cparams else None
        self.port = self.config.get('experiment', {}).get('port', 12345)

    def setup_experiment(self, experiment_config: Dict[str, Any]) -> ExperimentInterface:
        experiment_type = experiment_config.get('type', self.config['experiment']['type'])
        if experiment_type == 'yolo':
            from src.experiment_design.experiments.yolo_experiment import YOLOExperiment
            return YOLOExperiment(self.config, self.host, self.port)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    def run_experiment(self, experiment):
        experiment.run()
