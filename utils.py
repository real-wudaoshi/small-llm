import torch

def clean_state_dict(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict