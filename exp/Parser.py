import argparse
import yaml

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-c", "--config", required=True, help="Path to config.yaml"
        )

    def parse_args(self):
        """Parse CLI arguments."""
        return self.parser.parse_args()

    @staticmethod
    def load_config(config_path: str):
        """Load YAML config file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)