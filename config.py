import yaml
from pathlib import Path

config_dir = Path(__file__).parent.resolve() / "config"


# load yaml config
with open(config_dir / "config.yaml", "r") as f:
    config_yaml = yaml.safe_load(f)

excluded = config_yaml["EXCLUDED"]
ch_level = config_yaml["CH_LVL"]
dataset_path = config_yaml["DATASET_PATH"]
