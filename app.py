#!/usr/bin/env python3

"""
This script runs the "tracr" CLI, which is powered by the API that lives in
the "app_api" folder on this repo.

For a cleaner experience, add this directory to your PATH, which will allow
you to run the CLI from anywhere, and without preceding the command with
the word "python".
"""

import sys
import traceback
import warnings
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from time import sleep
from cryptography.utils import CryptographyDeprecationWarning

from src.tracr.app_api import utils
from src.tracr.app_api.log_handling import (
    LoggingContext,
    setup_logging,
    get_server_running_in_thread,
    shutdown_gracefully,
)
from src.tracr.app_api.device_mgmt import DeviceMgr
from src.tracr.app_api.experiment_mgmt import Experiment, ExperimentManifest
from src.tracr.app_api.model_interface import ModelFactoryInterface

# Ignore deprecation warnings from cryptography
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Constants
PROJECT_ROOT = Path(utils.get_repo_root())
CURRENT_VERSION = "0.3.2"

# Setup logging
LoggingContext.set_device("SERVER")
logger = setup_logging()


def create_model_factory() -> ModelFactoryInterface:
    """Creates a ModelFactory object for the experiment."""
    try:
        from src.tracr.experiment_design.models.model_hooked import WrappedModelFactory
        return WrappedModelFactory()
    except ImportError as e:
        logger.error(f"Failed to import WrappedModelFactory: {e}")
        sys.exit(1)


def device_ls(args: argparse.Namespace) -> None:
    """Lists available devices."""
    try:
        device_mgr = DeviceMgr()
        devices = device_mgr.get_devices()
        console = Console()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Reachable")
        table.add_column("Ready")
        table.add_column("Host")
        table.add_column("User")

        for device in devices:
            name = device._name
            device_type = device.get_type()
            can_be_reached = device.is_reachable()
            reachable = (
                "[bold green]Yes[/bold green]"
                if can_be_reached
                else "[bold red]No[/bold red]"
            )
            ready = (
                "[bold green]Yes[/bold green]"
                if can_be_reached
                else "[bold red]No[/bold red]"
            )
            host = device.get_current("host")
            user = device.get_current("user")

            table.add_row(name, device_type, reachable, ready, host, user)

        console.print(table)
    except Exception as e:
        logger.error(f"Error listing devices: {str(e)}")
        logger.error(traceback.format_exc())


def experiment_run(args: argparse.Namespace) -> None:
    """Runs the specified experiment."""
    try:
        exp_name = args.name[0]
        logger.info(f"Setting up experiment: {exp_name}")
        testcase_dir = PROJECT_ROOT / "src" / "tracr" / "app_api" / "test_cases"

        manifest_yaml_fp = next(
            testcase_dir.glob(f"**/*{exp_name}.yaml"), None)
        if not manifest_yaml_fp:
            logger.error(f"No manifest found for experiment: {exp_name}")
            sys.exit(1)
        logger.debug(f"Manifest found at: {manifest_yaml_fp}")

        rlog_server = get_server_running_in_thread()
        manifest = ExperimentManifest(manifest_yaml_fp)

        logger.debug("Initializing DeviceMgr")
        device_manager = DeviceMgr()
        available_devices = device_manager.get_devices(available_only=True)

        if not available_devices:
            logger.error("No available devices found.")
            sys.exit(1)

        # Create an instance of ModelFactoryInterface
        model_factory = create_model_factory()

        logger.debug("Initializing Experiment")
        # Pass the model_factory to the Experiment constructor
        experiment = Experiment(manifest, available_devices, model_factory)
        logger.debug(f"Running experiment: {exp_name}")
        experiment.run()

        shutdown_gracefully(rlog_server)
        sleep(2)  # Allow time for remaining logs to be displayed
        logger.info("Experiment concluded successfully")
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def setup_argparse() -> argparse.ArgumentParser:
    """Sets up the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description=r""" | |
        | |_ _ __ __ _  ___ _ __
        | __| '__/ _` |/ __| '__|
        | |_| | | (_| | (__| |
        \__|_|  \__,_|\___|_|

        A CLI for conducting collaborative AI experiments.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-v", "--version", action="version",
                        version=CURRENT_VERSION)

    subparsers = parser.add_subparsers(title="SUBMODULES")

    # Device subparser
    parser_device = subparsers.add_parser("device", help="Device management")
    device_subparsers = parser_device.add_subparsers(title="DEVICE COMMANDS")

    parser_device_ls = device_subparsers.add_parser("ls", help="List devices")
    parser_device_ls.set_defaults(func=device_ls)

    # Experiment subparser
    parser_experiment = subparsers.add_parser(
        "experiment", help="Experiment management"
    )
    exp_subparsers = parser_experiment.add_subparsers(
        title="EXPERIMENT COMMANDS")

    parser_experiment_run = exp_subparsers.add_parser(
        "run", help="Run an experiment")
    parser_experiment_run.add_argument(
        "name", nargs=1, help="Name of the experiment to run"
    )
    parser_experiment_run.add_argument(
        "-l", "--local", action="store_true", help="Run locally using simulated devices"
    )
    parser_experiment_run.add_argument(
        "-o", "--output", help="Specify performance logging output location"
    )
    parser_experiment_run.add_argument(
        "-p", "--preset", help="Use a preset runtime configuration"
    )
    parser_experiment_run.set_defaults(func=experiment_run)

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    try:
        parser = setup_argparse()
        args = parser.parse_args()

        if hasattr(args, "func"):
            args.func(args)
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
