"""Learnable visual-to-structure reasoning interface.

The modules in this package learn latent structure tokens from visual tokens.  They
never consume an oracle graph at inference time; graph annotations are accepted only
by the training losses in :mod:`recoalign.models.recoalign.losses`.
"""

from .adapter import ReasoningInterfaceAdapter
from .alignment_module import StructureAlignmentModule
from .structure_encoder import StructureTokenEncoder
from .structure_tokens import StructureTokenBank

__all__ = [
    "ReasoningInterfaceAdapter",
    "StructureAlignmentModule",
    "StructureTokenBank",
    "StructureTokenEncoder",
]
