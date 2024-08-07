"""
This module implements basic split inference behavior between a single "client"
node and a single "edge" node. It simulates a scenario where a client node
(e.g., an autonomous vehicle) initiates an inference and passes intermediary
data to a nearby edge server for completion.
"""

import logging
import rpyc
import threading
import uuid
from typing import Any
from rpyc.utils.server import ThreadedServer
from rpyc.utils.registry import REGISTRY_PORT, DEFAULT_PRUNING_TIMEOUT
from rpyc.utils.registry import UDPRegistryClient

from .base import ParticipantService
from ..tasks import FinishSignalTask, SimpleInferenceTask, SingleInputInferenceTask

logger = logging.getLogger("tracr_logger")


class ClientService(ParticipantService):
    """
    To define the way our client behaves, there are only three parts of the BaseExecutor class we
    need to overwrite:
        1.) The `partners` class attribute should list the names of the nodes we want to handshake
            with before the executor starts
        2.) The inference_sequence_per_input method tells the node to try a split at each possible
            layer for each input image it receives using the "cycle" type partitioner class,
            sending a SimpleInferenceTask to the "EDGE" node each time
        3.) The `on_finish` method should send a `FinishSignalTask` instance to the edge node's
            inbox so it knows it's done after it has finished all the inference tasks we sent it

    We also add a DOWNSTREAM_PARTNER class attribute for readability/adaptability, although this
    isn't strictly necessary.

    When the experiment runs, this executor will actually respond to an instance of
    `InferOverDatasetTask`, but because the base class's corresponding `infer_dataset` method just
    calls `inference_sequence_per_input` repeatedly, we don't have to change it directly.
    """

    DOWNSTREAM_PARTNER = "EDGE1"
    ALIASES: list[str] = ["CLIENT1", "PARTICIPANT"]
    partners: list[str] = ["OBSERVER", "EDGE1"]

    def __init__(self):
        super().__init__()
        logger.info("ClientService initialized")

    @rpyc.exposed
    def get_ready(self) -> None:
        """Prepare the client node for the experiment."""
        logger.info("ClientService get_ready method called")
        super().get_ready()
        logger.info("ClientService is ready")

    def inference_sequence_per_input(self, task: SingleInputInferenceTask) -> None:
        """
        Perform a sequence of inferences for a single input, trying splits at each possible layer.

        This method implements the core logic of the client's split inference strategy.
        It attempts to split the inference at different layers and delegates work to
        the edge node as necessary.

        Args:
            task (tasks.SingleInputInferenceTask): The task containing the input for inference.
        """
        assert self.model is not None, "Model must be initialized before inference"
        input_data = task.input
        
        try:
            splittable_layer_count = self.model.splittable_layer_count
        except AttributeError:
            logger.warning("Model does not have splittable_layer_count attribute. Assuming 1 layer.")
            splittable_layer_count = 1

        for current_split_layer in range(splittable_layer_count):
            inference_id = str(uuid.uuid4())
            start, end = 0, current_split_layer

            if end == 0:
                self._perform_full_inference(input_data, inference_id)
            elif end == splittable_layer_count - 1:
                self._delegate_full_inference(input_data, inference_id)
            else:
                self._perform_split_inference(input_data, inference_id, start, end)

    def _perform_full_inference(self, input_data: Any, inference_id: str) -> None:
        """Perform a full inference without involving the edge node."""
        logger.info("Completing full inference without help.")
        self.model(input_data, inference_id)

    def _delegate_full_inference(self, input_data: Any, inference_id: str) -> None:
        """Delegate the entire inference task to the edge node."""
        logger.info(f"Sending full job to {self.DOWNSTREAM_PARTNER}")
        downstream_task = SimpleInferenceTask(
            self.node_name, input_data, inference_id=inference_id, start_layer=0
        )
        self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)

    def _perform_split_inference(
        self, input_data: Any, inference_id: str, start: int, end: int
    ) -> None:
        """Perform a split inference, delegating part of the work to the edge node."""
        logger.info(f"Running split inference from layers {start} to {end}")
        out = self.model(input_data, inference_id, start=start, end=end)
        downstream_task = SimpleInferenceTask(
            self.node_name, out, inference_id=inference_id, start_layer=end
        )
        self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)

    def on_finish(self, _: Any) -> None:
        """
        Handle the completion of all tasks.

        This method sends a finish signal to the edge node and calls the
        superclass's on_finish method.

        Args:
            _ (Any): Unused argument for compatibility with superclass method.
        """
        downstream_finish_signal = FinishSignalTask(self.node_name)
        self.send_task(self.DOWNSTREAM_PARTNER, downstream_finish_signal)
        super().on_finish(_)


class EdgeService(ParticipantService):
    ALIASES: list[str] = ["EDGE1", "PARTICIPANT"]
    partners: list[str] = ["OBSERVER", "CLIENT1"]

    def __init__(self):
        super().__init__()
        logger.info("EdgeService initialized")
        self._start_service()

    def _start_service(self):
        try:
            logger.info("Starting EdgeService")
            self.server = ThreadedServer(self, port=18812, auto_register=False)
            self.server_thread = threading.Thread(target=self.server.start, daemon=True)
            self.server_thread.start()
            
            logger.info("Attempting to register EdgeService")
            registrar = UDPRegistryClient(ip="255.255.255.255", port=REGISTRY_PORT)
            for alias in self.ALIASES:
                registrar.register(alias, 18812, DEFAULT_PRUNING_TIMEOUT)
                logger.info(f"Registered alias: {alias}")
            logger.info(f"EdgeService registered with aliases: {self.ALIASES}")
        except Exception as e:
            logger.error(f"Failed to start or register EdgeService: {str(e)}")

    @rpyc.exposed
    def get_node_name(self) -> str:
        return "EDGE1"