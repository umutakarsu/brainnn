"""BCI decoders — from classical baselines to neuromorphic architectures."""

from brainnn.decoder.base import BaseDecoder
from brainnn.decoder.csp_lda import CSPLDADecoder
from brainnn.decoder.eegnet import EEGNet
from brainnn.decoder.synapseflow import SynapseFlowDecoder
from brainnn.decoder.transformer import EEGTransformer

__all__ = [
    "BaseDecoder",
    "CSPLDADecoder",
    "EEGNet",
    "EEGTransformer",
    "SynapseFlowDecoder",
]
