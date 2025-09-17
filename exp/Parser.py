import argparse
import yaml

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Load experiment configuration")
        self.parser.add_argument(
            "-c", "--config", required=True, help="Path to config.yaml"
        )

    def parse(self):
        """
        Parse CLI arguments and load the YAML configuration.
        Returns:
            dict: Configuration dictionary from YAML file.
        """

        args = self.parser.parse_args()
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        return config