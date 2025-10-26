import json

class envVars(object):
    env = None
    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.env = json.load(f)

Environment = envVars("cfg.json")