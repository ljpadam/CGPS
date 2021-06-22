from torch.utils.data import Sampler, BatchSampler
from torch._six import int_classes as _int_classes

class ConstrastiveBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    indices of every n indices are same, for constrastive learning

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last, same_indices_num=2):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        if batch_size%same_indices_num != 0:
            raise ValueError("batch_size should be divided by same_indices_num")
        self.sampler = sampler
        self.real_batch_size = batch_size * same_indices_num
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.same_indices_num = same_indices_num


    def __iter__(self):
        batch = []
        for idx in self.sampler:
            for _ in range(self.same_indices_num):
                batch.append(idx)
            if len(batch) == self.real_batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
