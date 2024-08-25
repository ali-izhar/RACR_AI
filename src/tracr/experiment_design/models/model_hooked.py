import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union
import copy
import os
from torchvision import models
from torchvision.transforms import ToTensor
from PIL import Image

from src.tracr.app_api.model_interface import ModelInterface, ModelFactoryInterface
from src.tracr.app_api.master_dict import MasterDict
from .model_utils import (
    read_model_config,
    model_selector,
    clear_cuda_cache,
    HookExitException,
    NotDict,
)

logger = logging.getLogger("tracr_logger")


class WrappedModel(ModelInterface):
    """
    Wraps a pretrained model with features necessary for edge computing tests.

    Uses PyTorch hooks to perform benchmarking, grab intermediate layers, and slice the
    Sequential to provide input to intermediate layers or exit early.
    """

    layer_template_dict = {
        "layer_id": None,
        "completed_by_node": None,
        "class": None,
        "inference_time": 0,
        "parameters": None,
        "parameter_bytes": None,
        "cpu_cycles_used": None,
        "watts_used": None,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        master_dict: Optional[MasterDict] = MasterDict(),
        flush_buffer_size: int = 100,
        **kwargs,
    ):
        try:
            self.timer = time.perf_counter_ns
            self.master_dict = master_dict
            self.io_buf_dict: Dict[str, Any] = {}
            self.inference_dict: Dict[str, Any] = {}
            self.forward_dict: Dict[int, Dict[str, Any]] = {}
            self.flush_buffer_size = flush_buffer_size

            config = read_model_config(config_path)
            self.device = config.get("device", "cpu")
            self.mode = config.get("mode", "eval")
            self.depth = config.get("depth", np.inf)
            self.base_input_size = config.get("input_size", (3, 224, 224))
            self.node_name = config.get("node_name", "unknown")
            self.model_name = config.get("model_name", "alexnet")
            self.pretrained = model_selector(self.model_name)
            self.splittable_layer_count = 0

            self.f_hooks = []
            self.f_pre_hooks = []

            self._setup_model()

            self.current_module_start_index = None
            self.current_module_stop_index = None
            self.current_module_index = None
            self.banked_input = None

            self.max_ignore_layer_index = self.splittable_layer_count - 1

            if self.mode == "eval":
                self.pretrained.eval()
            self._move_model_to_device()
            self.warmup(iterations=2)

            logger.info(
                f"WrappedModel initialized with device: {self.device}, mode: {self.mode}, model: {self.model_name}"
            )
        except Exception as e:
            logger.error(
                f"Error initializing WrappedModel: {str(e)}", exc_info=True)
            clear_cuda_cache()
            raise

    def _setup_model(self):
        try:
            from torchinfo import summary

            self.torchinfo = summary(
                self.pretrained, (1, *self.base_input_size), verbose=0
            )
            self._walk_modules(self.pretrained.children(), 1)
            del self.torchinfo
            self.empty_buffer_dict = copy.deepcopy(self.forward_dict)
            logger.info(
                f"Model setup complete. Total layers: {self.splittable_layer_count}"
            )
        except Exception as e:
            logger.error(f"Error in _setup_model: {str(e)}", exc_info=True)
            clear_cuda_cache()
            raise

    def _walk_modules(self, module_generator, depth):
        for child in module_generator:
            if len(list(child.children())) > 0 and depth < self.depth:
                logger.debug(
                    f"{'-'*depth}Module {str(child).split('(')[0]} with children found, hooking children instead of module."
                )
                self._walk_modules(child.children(), depth + 1)
                logger.debug(
                    f"{'-'*depth}End of Module {str(child).split('(')[0]}'s children."
                )
            elif isinstance(child, nn.Module):
                self._setup_layer_hooks(child, depth)

    def _setup_layer_hooks(self, child, depth):
        try:
            this_layer = next(
                (
                    layer
                    for layer in self.torchinfo.summary_list
                    if layer.layer_id == id(child)
                ),
                None,
            )
            if this_layer is None:
                raise Exception("module id not found while adding hooks.")

            this_layer_id = self.splittable_layer_count
            self.forward_dict[this_layer_id] = copy.deepcopy(
                self.layer_template_dict)
            self.forward_dict[this_layer_id].update(
                {
                    "depth": depth,
                    "layer_id": this_layer_id,
                    "class": this_layer.class_name,
                    "parameters": this_layer.num_params,
                    "parameter_bytes": this_layer.param_bytes,
                    "input_size": this_layer.input_size,
                    "output_size": this_layer.output_size,
                    "output_bytes": this_layer.output_bytes,
                }
            )

            self.f_hooks.append(
                child.register_forward_pre_hook(
                    self.forward_prehook(
                        this_layer_id, str(child).split("(")[0], (0, 0)
                    )
                )
            )
            self.f_pre_hooks.append(
                child.register_forward_hook(
                    self.forward_posthook(
                        this_layer_id, str(child).split("(")[0], (0, 0)
                    )
                )
            )
            logger.debug(
                f"{'-'*depth}Layer {this_layer_id}: {str(child).split('(')[0]} hooks applied."
            )
            self.splittable_layer_count += 1
        except Exception as e:
            logger.error(
                f"Error in _setup_layer_hooks: {str(e)}", exc_info=True)
            raise

    def forward_prehook(self, layer_index, layer_name, input_shape):
        def pre_hook(module, input):
            logger.debug(f"Prehook for layer {layer_index} ({layer_name})")
            assert (
                self.current_module_index is not None
                and self.current_module_start_index is not None
            )

            if (
                self.current_module_stop_index is not None
                and self.current_module_index >= self.current_module_stop_index
                and layer_index < self.max_ignore_layer_index
            ):
                logger.debug(f"Exiting early at layer {layer_index}")
                raise HookExitException(
                    input[0] if isinstance(input, tuple) else input)

            if self.log and (
                self.current_module_index >= self.current_module_start_index
            ):
                self.forward_dict[layer_index]["completed_by_node"] = self.node_name
                self.forward_dict[layer_index]["inference_time"] = - \
                    self.timer()

            if self.current_module_index == 0 and self.current_module_start_index > 0:
                self.banked_input = self._copy_input(
                    input[0] if isinstance(input, tuple) else input
                )
                return torch.randn(1, *self.base_input_size, device=self.device)
            elif (
                self.banked_input is not None
                and self.current_module_index == self.current_module_start_index
            ):
                input = self.banked_input
                self.banked_input = None
                return input

            input_tensor = self._unwrap_input(
                input[0] if isinstance(input, tuple) else input
            )

            if layer_name.startswith("classifier"):
                reshaped = self._reshape_for_classifier(input_tensor)
                logger.debug(
                    f"Reshaped for classifier. Input shape: {input_tensor.shape}, Output shape: {reshaped.shape}"
                )
                return reshaped

            logger.debug(f"Prehook input shape: {input_tensor.shape}")
            self.current_module_index += 1
            return input

        return pre_hook

    def forward_posthook(self, layer_index, layer_name, input_shape, **kwargs):
        def hook(module, input, output):
            logger.debug(f"Posthook for layer {layer_index} ({layer_name})")
            if (
                self.log
                and self.current_module_index >= self.current_module_start_index
            ):
                self.forward_dict[layer_index]["inference_time"] += self.timer()

            output_tensor = self._unwrap_input(output)
            logger.debug(f"Posthook output shape: {output_tensor.shape}")

            if layer_name == "features" and self.current_module_index == len(
                self.pretrained.features
            ):
                reshaped = self._reshape_for_classifier(output_tensor)
                logger.debug(
                    f"Reshaped after features. Input shape: {output_tensor.shape}, Output shape: {reshaped.shape}"
                )
                return reshaped

            return output

        return hook

    def _unwrap_input(self, x):
        """Unwrap input if it's a NotDict instance."""
        if isinstance(x, NotDict):
            return x.inner_dict
        return x

    def _copy_input(self, x):
        """Safely copy input, handling NotDict instances."""
        if isinstance(x, NotDict):
            return NotDict(
                x.inner_dict.clone()
                if isinstance(x.inner_dict, torch.Tensor)
                else x.inner_dict.copy()
            )
        return x.clone() if isinstance(x, torch.Tensor) else x.copy()

    def _reshape_for_classifier(self, x):
        """Reshape the input for the classifier, handling NotDict instances."""
        x = self._unwrap_input(x)
        logger.debug(f"Reshaping for classifier. Input shape: {x.shape}")
        if x.dim() > 2:
            x = self.pretrained.avgpool(x)
            x = torch.flatten(x, 1)
        expected_features = 256 * 6 * 6  # 9216
        if x.size(1) != expected_features:
            logger.warning(
                f"Unexpected feature size. Expected {expected_features}, got {x.size(1)}. Adjusting..."
            )
            if x.size(1) < expected_features:
                padding = torch.zeros(
                    x.size(0), expected_features - x.size(1), device=x.device
                )
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :expected_features]
        logger.debug(f"Reshaped for classifier. Output shape: {x.shape}")
        return x

    def forward(
        self,
        x: Any,
        inference_id: Optional[str] = None,
        start: int = 0,
        end: Union[int, float] = float("inf"),
        log: bool = True,
    ) -> Any:
        try:
            logger.info(f"Starting forward pass. Input shape: {x.shape}")
            end = self.splittable_layer_count if end == float("inf") else end
            self.log = log
            self.current_module_stop_index = end
            self.current_module_index = 0
            self.current_module_start_index = start

            _inference_id = "unlogged" if inference_id is None else inference_id
            self.inference_dict["inference_id"] = _inference_id
            logger.info(f"{_inference_id} id beginning.")

            if self.mode != "train":
                with torch.no_grad():
                    x = self._run_forward_pass(x, end)
            else:
                x = self._run_forward_pass(x, end)

            logger.info(
                f"Forward pass completed successfully. Output shape: {x.shape}")
        except HookExitException as e:
            logger.debug(
                f"Exit early from hook at layer {self.current_module_index}")
            x = e.result
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA out of memory error. Clearing cache and retrying.")
            clear_cuda_cache()
            raise
        except Exception as e:
            logger.error(f"Error during forward pass: {str(e)}", exc_info=True)
            raise

        self._process_inference_results(_inference_id)
        logger.info(f"{_inference_id} end.")
        return x

    def _run_forward_pass(self, x: torch.Tensor, end: int) -> torch.Tensor:
        x = self.pretrained.features(x)
        x = self._reshape_for_classifier(x)
        if end >= self.splittable_layer_count:
            x = self.pretrained.classifier(x)
        return x

    def _process_inference_results(self, _inference_id: str):
        try:
            self.inference_dict["layer_information"] = self.forward_dict
            if self.log and self.master_dict:
                self.io_buf_dict[str(_inference_id).split(".")[0]] = copy.deepcopy(
                    self.inference_dict
                )
                if len(self.io_buf_dict) >= self.flush_buffer_size:
                    self.update_master_dict()
            self.inference_dict = {}
            self.forward_dict = copy.deepcopy(self.empty_buffer_dict)
            self.current_module_stop_index = None
            self.current_module_index = None
        except Exception as e:
            logger.error(
                f"Error in _process_inference_results: {str(e)}", exc_info=True
            )
            raise

    def update_master_dict(self) -> None:
        try:
            logger.debug("WrappedModel.update_master_dict called")
            if self.master_dict is not None and self.io_buf_dict:
                logger.info(
                    f"Flushing {len(self.io_buf_dict)} items from IO buffer dict to MasterDict"
                )
                self.master_dict.update(self.io_buf_dict)
                self.io_buf_dict = {}
                self._save_master_dict_to_disk()
                return
            logger.info(
                "MasterDict not updated; either buffer is empty or MasterDict is None"
            )
        except Exception as e:
            logger.error(
                f"Error updating master dict: {str(e)}", exc_info=True)
            raise

    def _save_master_dict_to_disk(self):
        try:
            save_path = os.path.join(os.getcwd(), "master_dict_backup.pkl")
            torch.save(self.master_dict, save_path)
            logger.info(f"Master dictionary saved to {save_path}")
        except Exception as e:
            logger.error(
                f"Error saving master dictionary to disk: {str(e)}", exc_info=True
            )

    def parse_input(self, _input: Any) -> Any:
        try:
            if isinstance(_input, Image.Image):
                if _input.size != self.base_input_size[1:]:
                    _input = _input.resize(self.base_input_size[1:])
                input_tensor = ToTensor()(_input).unsqueeze(0)
            elif isinstance(_input, torch.Tensor):
                input_tensor = _input
            else:
                raise ValueError(
                    f"Bad input given to WrappedModel: type {type(_input)}"
                )
            return self._move_to_device(input_tensor)
        except Exception as e:
            logger.error(f"Error parsing input: {str(e)}", exc_info=True)
            raise

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        try:
            if (
                torch.cuda.is_available()
                and self.device == "cuda"
                and tensor.device != self.device
            ):
                return tensor.to(self.device)
            return tensor
        except Exception as e:
            logger.error(
                f"Error moving tensor to device: {str(e)}", exc_info=True)
            raise

    def _move_model_to_device(self):
        try:
            self.pretrained.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
        except Exception as e:
            logger.error(
                f"Error moving model to device: {str(e)}", exc_info=True)
            clear_cuda_cache()
            raise

    def warmup(self, iterations: int = 50, force: bool = False) -> None:
        try:
            if self.device != "cuda" and not force:
                logger.info("Warmup not required.")
                return

            logger.info(f"Starting warmup with {iterations} iterations.")
            with torch.no_grad():
                for i in range(iterations):
                    self(torch.randn(1, *self.base_input_size), log=False)
                    if (i + 1) % 10 == 0:
                        logger.debug(f"Completed {i + 1} warmup iterations")
            logger.info("Warmup complete.")
        except Exception as e:
            logger.error(f"Error during warmup: {str(e)}", exc_info=True)
            raise

    def __call__(
        self,
        x: Any,
        inference_id: Optional[str] = None,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Any:
        try:
            return self.forward(
                x,
                inference_id=inference_id,
                start=start,
                end=end if end is not None else float("inf"),
            )
        except Exception as e:
            logger.error(f"Error in __call__: {str(e)}", exc_info=True)
            raise

    def __str__(self):
        return f"WrappedModel(device={self.device}, mode={self.mode}, layers={self.splittable_layer_count})"

    def __repr__(self):
        return (
            f"WrappedModel(device={self.device}, mode={self.mode}, "
            f"depth={self.depth}, layer_count={self.splittable_layer_count})"
        )

    def __del__(self):
        try:
            logger.info("Deleting WrappedModel instance")
            self.update_master_dict()
            clear_cuda_cache()
            logger.info("WrappedModel instance deleted")
        except Exception as e:
            logger.error(
                f"Error deleting WrappedModel: {str(e)}", exc_info=True)


class WrappedModelFactory(ModelFactoryInterface):
    def create_model(
        self,
        config_path: Optional[str] = None,
        master_dict: Any = None,
        flush_buffer_size: int = 100,
    ) -> ModelInterface:
        try:
            logger.info("Creating WrappedModel instance")
            model = WrappedModel(
                config_path=config_path,
                master_dict=master_dict,
                flush_buffer_size=flush_buffer_size,
            )
            logger.info(f"WrappedModel instance created: {model}")
            return model
        except Exception as e:
            logger.error(
                f"Error creating WrappedModel: {str(e)}", exc_info=True)
            raise
