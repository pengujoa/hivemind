"""1-bit sign compression for pseudo-gradient communication in DiLoCo.

Packs sign bits (positive/negative) into 1 bit per parameter,
achieving 32x compression ratio compared to float32.
"""

import numpy as np
import torch

from hivemind.compression.base import CompressionBase, CompressionInfo
from hivemind.proto import runtime_pb2


class SignBitCompression(CompressionBase):
    """Compress tensors to 1-bit sign representation.

    compress: float tensor -> sign bits packed 8 per byte
    extract: packed bytes -> float tensor with values in {-1.0, +1.0}

    Zero values are mapped to +1.0 (positive bias) during compression.
    """

    compression_type = runtime_pb2.SIGN_1BIT

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        sign_bits = (tensor.detach().view(-1) >= 0).numpy().astype(np.uint8)
        packed = np.packbits(sign_bits)
        return runtime_pb2.Tensor(
            compression=self.compression_type,
            buffer=packed.tobytes(),
            size=tensor.shape,
            dtype=tensor.data.numpy().dtype.name if tensor.dtype != torch.bfloat16 else "bfloat16",
            requires_grad=tensor.requires_grad,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        packed = np.frombuffer(serialized_tensor.buffer, dtype=np.uint8)
        numel = 1
        for s in serialized_tensor.size:
            numel *= s
        bits = np.unpackbits(packed)[:numel]
        tensor = torch.from_numpy(bits.astype(np.float32)) * 2.0 - 1.0
        return tensor.reshape(tuple(serialized_tensor.size))

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return 1.0 / torch.finfo(info.descriptor.dtype).bits
