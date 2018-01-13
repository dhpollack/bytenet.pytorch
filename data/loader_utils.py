from torch import stack, cat


def _pad_tensor(vec, pad, dim, val=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    if vec.size(dim) == pad:
        return vec
    else:
        pad_size = list(vec.size())
        pad_size[dim] = pad - vec.size(dim)
        pad_vec = vec.new(*pad_size).fill_(val)
        return cat([vec, pad_vec], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dims=(-1, -1), pad_vals=(0.,0)):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dims = dims
        self.pad_vals = pad_vals

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len_x = max([x[0].size(self.dims[0]) for x in batch])
        max_len_y = max([x[1].size(self.dims[1]) for x in batch])
        # pad according to max_len
        for i, (x, y) in enumerate(batch):
            if self.dims[0] is not None:
                x = _pad_tensor(x, pad=max_len_x, dim=self.dims[0], val=self.pad_vals[0])
            if len(self.dims) == 2:
                y = _pad_tensor(y, pad=max_len_y, dim=self.dims[1], val=self.pad_vals[1])
            batch[i] = (x, y)
        # stack all
        xs = stack([x[0] for x in batch], dim=0)
        ys = stack([x[1] for x in batch], dim=0)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
