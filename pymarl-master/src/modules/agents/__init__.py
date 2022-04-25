REGISTRY = {}

from .rnn_agent import RNNAgent
from .macc_agent import MACCAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["macc"] = MACCAgent
