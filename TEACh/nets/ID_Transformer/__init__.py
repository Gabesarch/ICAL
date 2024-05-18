from .network_base import build

def build_model(
    args, 
    num_classes, 
    num_actions=None,
    actions2idx=None,
    ):
    return build(num_classes, num_actions, actions2idx)