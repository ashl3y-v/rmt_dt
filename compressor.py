import torch as T
from torch import nn
from torch.nn import functional as F


class Compressor(nn.Module):
    def __init__(
        self, storage, block_size=11, max_length=200, dtype=T.float32, device="cpu"
    ):
        super().__init__()

        assert block_size % 2 != 0

        self.block_size = block_size
        self.max_length = max_length

        self.compressed = T.tensor([], dtype=dtype, device=device)
        self.storage = storage

        self.compressor = nn.Conv1d(
            self.block_size,
            1,
            kernel_size=self.block_size,
            padding=(self.block_size // 2),
        )
        self.decompressor = nn.ConvTranspose1d(
            1,
            self.block_size,
            kernel_size=self.block_size,
            padding=(self.block_size // 2),
        )

    def compress(self, assign=True):
        n_blocks = (self.storage.shape[-2] - self.max_length) // self.block_size
        blocks = self.storage[: self.block_size * n_blocks, :]
        if assign:
            self.storage = self.storage[self.block_size * n_blocks :, :]
        blocks = T.stack(T.split(blocks, self.block_size), dim=0)
        compressed = blocks.mean(dim=-2) + self.compressor(blocks).squeeze()

        if assign:
            self.compressed = T.cat([self.compressed, compressed], dim=0)

        return compressed

    def decompress(self, assign=True):
        compressed = self.compressed
        if assign:
            self.compressed = self.compressed.new_empty([])
        a = compressed.expand(self.block_size * compressed.shape[0], -1)
        b = self.decompressor(compressed).squeeze()

        print(a, b)

        if assign:
            self.compressed = T.cat([self.compressed, compressed], dim=0)

        return compressed


if __name__ == "__main__":
    storage = T.randn([512, 69])
    compressor = Compressor(storage, max_length=200)

    compressor.compress()
    compressor.decompress()

    print(compressor.storage.shape, compressor.compressed.shape)

    # print(compressor.compress().shape)
