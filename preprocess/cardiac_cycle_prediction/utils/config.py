import argparse

import yaml

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to the YAML config file', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config

