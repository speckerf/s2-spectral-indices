# Config file parsing
import yaml

with open("config.yaml", "r") as file:
    config_spectral = yaml.safe_load(file)


# this function is used to access the config from anywhere
def get_config() -> dict:
    return config_spectral
