# scripts/run_onion_yolo.py

import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List
import os

import torch
from PIL import Image
from tqdm import tqdm

# Add parent module (src) to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.api.master_dict import MasterDict
from src.experiment_design.models.model_hooked import WrappedModel, NotDict
from src.experiment_design.datasets.dataloader import DataManager
from src.utils.ml_utils import DetectionUtils
from src.utils.system_utils import read_yaml_file


# ----------- CONSTANTS AND CONFIGURATION -----------#
CONFIG_YAML_PATH = Path("config/model_config.yaml")
config = read_yaml_file(CONFIG_YAML_PATH)

# Update paths based on RUN_ON_EDGE
if config["default"]["run_on_edge"]:
    BASE_DIR = Path("/tmp/RACR_AI")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent


DATASET_NAME = config["default"]["default_dataset"]
CLASS_NAMES = config["dataset"][DATASET_NAME]["class_names"]

MODEL_NAME = config["default"]["default_model"]
SPLIT_LAYER = config["model"][MODEL_NAME]["split_layer"]

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
FONT_PATH = BASE_DIR / "fonts" / "DejaVuSans-Bold.ttf"

LOG_FILE = config["model"][MODEL_NAME]["log_file"]

log_dir = Path(LOG_FILE).parent
if not log_dir.exists():
    log_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(config["default"]["log_level"])
logger.addHandler(file_handler)


# Function to create an incremented directory
def create_incremented_dir(base_path: Path) -> Path:
    i = 1
    while True:
        new_path = base_path.with_name(f"{base_path.name}_{i}")
        if not new_path.exists():
            new_path.mkdir(parents=True)
            return new_path
        i += 1


# Create results directory
RESULTS_DIR = BASE_DIR / "results" / f"{MODEL_NAME}_{DATASET_NAME}"
if RESULTS_DIR.exists():
    RESULTS_DIR = create_incremented_dir(RESULTS_DIR)
else:
    RESULTS_DIR.mkdir(parents=True)

OUTPUT_DIR = RESULTS_DIR / "output_images"
OUTPUT_CSV_PATH = RESULTS_DIR / "inference_results.csv"
LOG_FILE_PATH = RESULTS_DIR / f"{MODEL_NAME}_{DATASET_NAME}.log"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------#

detection_utils = DetectionUtils(
    CLASS_NAMES, str(FONT_PATH), CONF_THRESHOLD, IOU_THRESHOLD
)


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, Image.Image, str]]
) -> Tuple[torch.Tensor, List[Image.Image], List[str]]:
    """Custom collate function to handle batches without merging PIL Images."""
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), list(original_images), list(image_files)


def process_batch(
    model1: WrappedModel,
    model2: WrappedModel,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    image_filename: str,
    master_dict: MasterDict,
    verbose: bool = False,
) -> None:
    """Processes a single batch of data."""
    logger.debug(f"Processing batch for image: {image_filename}")

    try:
        # Run forward pass on model1 up to split_layer
        res = model1(input_tensor, end=SPLIT_LAYER)
        if verbose:
            logger.debug(f"Processed split layer {SPLIT_LAYER} for {image_filename}.")

        # Handle early exit or intermediate results
        if isinstance(res, NotDict):
            inner_dict = res.inner_dict
            # Move tensors to model2's device if necessary
            for key, value in inner_dict.items():
                if isinstance(value, torch.Tensor):
                    inner_dict[key] = value.to(model2.device)
                    if verbose:
                        logger.debug(f"Moved tensor '{key}' to device {model2.device}.")
        else:
            logger.warning(
                f"Result from model1 is not an instance of NotDict for {image_filename}."
            )

        # Run forward pass on model2 starting from split_layer
        out = model2(res, start=SPLIT_LAYER)

        # Collect inference results
        inference_id = image_filename
        master_dict[inference_id] = {
            "inference_id": inference_id,
            "layer_information": model1.forward_info.copy(),
        }

        # Post-process outputs to get detections
        detections = detection_utils.postprocess(out, original_image.size)

        # Draw detections on the original image
        output_image = detection_utils.draw_detections(original_image, detections)

        # Save the output image
        output_image_path = OUTPUT_DIR / f"output_with_detections_{image_filename}"
        try:
            output_image.save(str(output_image_path))  # Convert PosixPath to string
            logger.info(f"Saved image: {output_image_path}")

            # Verify that the image was saved successfully
            if output_image_path.exists():
                logger.debug(f"Verified existence of saved image: {output_image_path}")
            else:
                logger.error(
                    f"Image file was not found after saving: {output_image_path}"
                )
        except Exception as e:
            logger.error(f"Failed to save image {image_filename}: {e}")
            raise

        logger.info(
            f"Processed {image_filename}. Detections: {len(detections)}. Output saved to {output_image_path}"
        )

    except Exception as e:
        logger.error(f"Error processing batch for {image_filename}: {e}")
        raise


def main(warmup_only=False, split_experiment=False):
    """Main function to execute the testing pipeline."""
    logger.info(
        f"Starting the {MODEL_NAME.capitalize()} model test on {DATASET_NAME.capitalize()} dataset"
    )
    logger.info(f"Warmup only: {warmup_only}, Split experiment: {split_experiment}")

    # Check if running on edge device
    run_on_edge = config["default"]["run_on_edge"]
    logger.info(f"Running on edge device: {run_on_edge}")

    # Select the model and dataset from configuration
    default_model = config.get("default", {}).get("default_model", "yolo")
    default_dataset = config.get("default", {}).get("default_dataset", "onion")
    logger.info(f"Selected model: {default_model}")
    logger.info(f"Selected dataset: {default_dataset}")

    # Extract dataset and model configurations
    dataset_config = config.get("dataset", {}).get(default_dataset)
    model_config = config.get("model", {}).get(default_model)

    if not dataset_config or not model_config:
        logger.error("Dataset or model configuration not found.")
        sys.exit(1)

    dataset_config["args"]["root"] = str(BASE_DIR / dataset_config["args"]["root"])    
    logger.info(f"Dataset root: {dataset_config['args']['root']}")

    # Initialize MasterDict
    master_dict = MasterDict()

    # Set up DataLoader
    dataloader_config = config.get("dataloader", {}).copy()
    dataloader_config["collate_fn"] = custom_collate_fn

    final_config = {"dataset": dataset_config, "dataloader": dataloader_config}

    try:
        data_loader = DataManager.get_data(final_config)
        logger.info(
            f"DataLoader created with batch size {dataloader_config.get('batch_size', 1)}."
        )
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}")
        sys.exit(1)

    # Initialize models with device types
    try:
        model1 = WrappedModel(config=config)
        model1.node_name = "CLIENT1"
        model1.eval()
        model2 = WrappedModel(config=config)
        model2.node_name = "EDGE1"
        model2.eval()
        logger.info("Models initialized and set to evaluation mode.")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        sys.exit(1)

    # Define path to the TrueType font
    font_path = Path("fonts/DejaVuSans-Bold.ttf")
    if not font_path.exists():
        logger.error(
            f"Font file not found at {font_path}. Please ensure the font file is present."
        )
        sys.exit(1)
    else:
        logger.info(f"Using font file at {font_path}")

    # Process batches
    with torch.no_grad():
        for i, (images, original_images, image_files) in enumerate(tqdm(
            data_loader, desc=f"Testing split at layer {SPLIT_LAYER}"
        )):
            input_tensor = images.to(model1.device)
            original_image = original_images[0]
            image_filename = image_files[0]

            if warmup_only and i >= config["default"]["warmup_iterations"]:
                logger.info(f"Warmup iterations completed. Processed {i} iterations.")
                break

            if split_experiment and i < config["default"]["warmup_iterations"]:
                logger.info(f"Skipping warmup iteration {i+1}")
                continue

            logger.info(f"Processing batch {i+1}, image: {image_filename}")

            try:
                process_batch(
                    model1,
                    model2,
                    input_tensor,
                    original_image,
                    image_filename,
                    master_dict,
                    verbose=True,  # Set to True for more detailed logging
                )
            except Exception as e:
                logger.error(f"Error processing batch for {image_filename}: {e}")

    # Save results
    if not warmup_only:
        try:
            df = master_dict.to_dataframe()
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            logger.info(f"Inference results saved to {OUTPUT_CSV_PATH}")
        except Exception as e:
            logger.error(f"Failed to save inference results: {e}")

    logger.info(
        f"{MODEL_NAME.capitalize()} model test on {DATASET_NAME.capitalize()} dataset completed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO model on Onion dataset")
    parser.add_argument("--warmup_only", action="store_true", help="Run only warmup iterations")
    parser.add_argument("--split_experiment", action="store_true", help="Run split experiment")
    args = parser.parse_args()

    main(warmup_only=args.warmup_only, split_experiment=args.split_experiment)
