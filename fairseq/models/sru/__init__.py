"""
sru - Simple Recurrent Unit

sru provides a PyTorch implementation of the simple recurrent neural network cell described
in "Simple Recurrent Units for Highly Parallelizable Recurrence."
"""
from fairseq.models.sru.version import __version__
from fairseq.models.sru.modules import SRU, SRUCell
from fairseq.models.sru.modules import SRUpp, SRUppCell

__all__ = ['__version__', 'SRU', 'SRUCell', 'SRUpp', 'SRUppCell']
