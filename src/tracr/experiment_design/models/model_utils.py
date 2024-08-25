# experiment_design/models/model_utils.py

import logging
import os
from typing import Any, Dict, Optional
import yaml
import numpy as np
import importlib
import torch

logger = logging.getLogger("tracr_logger")


class NotDict:
    """Wrapper for a dict to circumvent some of Ultralytics forward pass handling."""

    def __init__(self, passed_dict: Dict[str, Any]) -> None:
        self.inner_dict = passed_dict

    def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Any]:
        return self.inner_dict

    @property
    def shape(self):
        if isinstance(self.inner_dict, torch.Tensor):
            return self.inner_dict.shape
        return None


class HookExitException(Exception):
    """Exception to early exit from inference in naive running."""

    def __init__(self, out: Any, *args: object) -> None:
        super().__init__(*args)
        self.result = out


def read_model_config(
    path: Optional[str] = None, participant_key: str = "client"
) -> Dict[str, Any]:
    """Read and combine model configuration from YAML files."""
    try:
        config_details = _read_yaml_data(path, participant_key)
        model_fixed_details = _read_fixed_model_config(
            config_details["model_name"])
        config_details.update(model_fixed_details)
        logger.info(
            f"Model configuration read successfully for {participant_key}")
        return config_details
    except Exception as e:
        logger.error(
            f"Error reading model configuration: {str(e)}", exc_info=True)
        raise


def _read_yaml_data(path: Optional[str], participant_key: str) -> Dict[str, Any]:
    """Read YAML data from a file and extract model settings."""
    try:
        if path:
            with open(path, "r") as file:
                settings = yaml.safe_load(file)["participant_types"][participant_key][
                    "model"
                ]
            logger.debug(f"YAML data read successfully from {path}")
        else:
            logger.warning("No path provided. Using default settings.")
            settings = _get_default_settings()
        return settings
    except Exception as e:
        logger.warning(
            f"Error reading YAML data: {str(e)}. Using default settings.")
        return _get_default_settings()


def _get_default_settings() -> Dict[str, Any]:
    """Return default settings when YAML reading fails."""
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mode": "eval",
        "depth": np.inf,
        "input_size": (3, 224, 224),
        "model_name": "alexnet",
    }


def _read_fixed_model_config(model_name: str) -> Dict[str, Any]:
    """Read fixed model configuration from model_configs.yaml."""
    try:
        config_path = os.path.join(
            os.path.dirname(__file__), "model_configs.yaml")
        with open(config_path, encoding="utf8") as file:
            configs = yaml.safe_load(file)
        model_type = "yolo" if "yolo" in model_name.lower() else model_name
        config = configs.get(model_type, {})
        logger.debug(f"Fixed model configuration read for {model_name}")
        return config
    except Exception as e:
        logger.error(
            f"Error reading fixed model configuration: {str(e)}", exc_info=True
        )
        raise


def model_selector(model_name: str) -> Any:
    """Select and return a model based on the given name."""
    logger.info(f"Selecting model: {model_name}")
    try:
        if "alexnet" in model_name:
            from torchvision import models

            return models.alexnet(weights="DEFAULT")
        elif "yolo" in model_name:
            ultralytics = importlib.import_module("ultralytics")
            return ultralytics.YOLO(f"{model_name}.pt").model
        else:
            raise NotImplementedError(
                f"Model {model_name} is not implemented.")
    except ImportError as e:
        logger.error(
            f"Error importing required module: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}", exc_info=True)
        raise


def clear_cuda_cache():
    """
    Clear CUDA cache if CUDA is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")
    else:
        logger.info("CUDA not available, no cache to clear")
