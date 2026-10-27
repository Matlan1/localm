# SPDX-License-Identifier: AGPL-3.0-or-later
import os as _os

__version__ = "0.3.1"

# Number NVIDIA GPUs by PCI bus in this process and every process it starts,
# so torch, nvidia-smi and the llama.cpp runtime agree on which card is GPU N.
# A value the user set is kept. See test_cuda_device_order_is_pci_bus_id.
CUDA_ORDER_SET_BY_LOCALM = "CUDA_DEVICE_ORDER" not in _os.environ
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
