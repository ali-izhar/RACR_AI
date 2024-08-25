import logging
import rpyc
import threading
import uuid
import time
from typing import Any
from rpyc.utils.server import ThreadedServer
from rpyc.utils.registry import REGISTRY_PORT, DEFAULT_PRUNING_TIMEOUT
from rpyc.utils.registry import UDPRegistryClient

from .base import ParticipantService
from ..tasks import (
    FinishSignalTask,
    SimpleInferenceTask,
    SingleInputInferenceTask,
)

logger = logging.getLogger("tracr_logger")


class ServerService(ParticipantService):
    DOWNSTREAM_PARTNER = "PARTICIPANT"
    ALIASES = ["SERVER", "PARTICIPANT"]
    partners = ["PARTICIPANT"]

    def __init__(self):
        super().__init__()
        self.port = 18861  # Fixed port
        logger.info("ServerService initialized")

    @rpyc.exposed
    def get_node_name(self):
        return "SERVER"

    def prepare_model(self, model: Any) -> None:
        super().prepare_model(model)
        self.splittable_layer_count = getattr(
            model, "splittable_layer_count", getattr(model, "layer_count", 1)
        )
        logger.info(
            f"ServerService model prepared with {self.splittable_layer_count} splittable layers"
        )

    def inference_sequence_per_input(self, task: SingleInputInferenceTask) -> None:
        logger.info(
            f"ServerService starting inference sequence for task: {task.inference_id}"
        )
        assert self.model is not None, "Model must be initialized before inference"
        input_data = task.input

        for current_split_layer in range(self.splittable_layer_count):
            inference_id = str(uuid.uuid4())
            start, end = 0, current_split_layer

            if end == 0:
                self._perform_full_inference(input_data, inference_id)
            elif end == self.splittable_layer_count - 1:
                self._delegate_full_inference(input_data, inference_id)
            else:
                self._perform_split_inference(input_data, inference_id, start, end)

        logger.info(
            f"ServerService completed inference sequence for task: {task.inference_id}"
        )

    def _perform_full_inference(self, input_data: Any, inference_id: str) -> None:
        logger.info(f"ServerService performing full inference for {inference_id}")
        try:
            result = self.model.forward(input_data, inference_id=inference_id)
            logger.info(
                f"Full inference completed for {inference_id}. Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Error during full inference for {inference_id}: {str(e)}")
            logger.exception("Traceback:")

    def _perform_split_inference(
        self, input_data: Any, inference_id: str, start: int, end: int
    ) -> None:
        logger.info(
            f"ServerService performing split inference for {inference_id} from layers {start} to {end}"
        )
        try:
            out = self.model.forward(
                input_data, inference_id=inference_id, start=start, end=end
            )
            downstream_task = SimpleInferenceTask(
                self.node_name, out, inference_id=inference_id, start_layer=end
            )
            self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)
        except Exception as e:
            logger.error(f"Error during split inference for {inference_id}: {str(e)}")
            logger.exception("Traceback:")

    def _delegate_full_inference(self, input_data: Any, inference_id: str) -> None:
        logger.info(
            f"ServerService delegating full inference for {inference_id} to {self.DOWNSTREAM_PARTNER}"
        )
        downstream_task = SimpleInferenceTask(
            self.node_name, input_data, inference_id=inference_id, start_layer=0
        )
        self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)

    def on_finish(self, task: Any) -> None:
        logger.info("ServerService finishing tasks")
        downstream_finish_signal = FinishSignalTask(self.node_name)
        try:
            self.send_task(self.DOWNSTREAM_PARTNER, downstream_finish_signal)
            logger.info(f"Sent finish signal to {self.DOWNSTREAM_PARTNER}")
        except Exception as e:
            logger.error(
                f"Failed to send finish signal to {self.DOWNSTREAM_PARTNER}: {str(e)}"
            )

        super().on_finish(task)

    def get_participant_connection(self):
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                conn = rpyc.connect_by_service("PARTICIPANT")
                logger.info("Successfully connected to PARTICIPANT service")
                return conn
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} to connect to PARTICIPANT failed: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to connect to PARTICIPANT after {max_retries} attempts"
                    )
                    raise
                time.sleep(retry_delay)


class ParticipantService(ParticipantService):
    ALIASES: list[str] = ["PARTICIPANT", "PARTICIPANT1"]
    partners: list[str] = ["SERVER"]

    def __init__(self):
        super().__init__()
        logger.info("ParticipantService initialized")
        self._start_service()

    def _start_service(self):
        try:
            logger.info("Starting ParticipantService")
            self.server = ThreadedServer(self, port=18812, auto_register=False)
            self.server_thread = threading.Thread(target=self.server.start, daemon=True)
            self.server_thread.start()

            logger.info("Attempting to register ParticipantService")
            registrar = UDPRegistryClient(ip="255.255.255.255", port=REGISTRY_PORT)
            for alias in self.ALIASES:
                registrar.register(alias, 18812, DEFAULT_PRUNING_TIMEOUT)
                logger.info(f"Registered alias: {alias}")
            logger.info(f"ParticipantService registered with aliases: {self.ALIASES}")
        except Exception as e:
            logger.error(f"Failed to start or register ParticipantService: {str(e)}")

    @rpyc.exposed
    def get_node_name(self) -> str:
        return "PARTICIPANT"

    def _perform_full_inference(self, input_data: Any, inference_id: str) -> None:
        logger.info(f"ParticipantService performing full inference for {inference_id}")
        try:
            result = self.model.forward(input_data, inference_id=inference_id)
            logger.info(
                f"Full inference completed for {inference_id}. Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Error during full inference for {inference_id}: {str(e)}")
            logger.exception("Traceback:")

    def _perform_split_inference(
        self, input_data: Any, inference_id: str, start: int, end: int
    ) -> None:
        logger.info(
            f"ParticipantService performing split inference for {inference_id} from layers {start} to {end}"
        )
        try:
            out = self.model.forward(
                input_data, inference_id=inference_id, start=start, end=end
            )
            logger.info(
                f"Split inference completed for {inference_id}. Output shape: {out.shape if hasattr(out, 'shape') else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Error during split inference for {inference_id}: {str(e)}")
            logger.exception("Traceback:")

    def _delegate_full_inference(self, input_data: Any, inference_id: str) -> None:
        logger.error("ParticipantService should not delegate full inference")
        raise NotImplementedError("ParticipantService cannot delegate full inference")
