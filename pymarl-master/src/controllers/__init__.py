REGISTRY = {}

from .basic_controller import BasicMAC
from .qsco_controller import qsco_MAC
from .macc_controller import MACCMAC
from .rnn_controller import RNNMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["qsco_mac"] = qsco_MAC
REGISTRY["macc_mac"] = MACCMAC
REGISTRY["rnn_mac"] = RNNMAC