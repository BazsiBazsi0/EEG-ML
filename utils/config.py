import yaml
from pathlib import Path

config_dir = Path(__file__).parent.parent.resolve() / "config"

# load yaml config
with open(config_dir / "config.yaml", "r") as f:
    config_yaml = yaml.safe_load(f)

channel_inclusion_lvl = config_yaml["CH_LVL"]
excluded_pat = config_yaml["EXCLUDED"]
