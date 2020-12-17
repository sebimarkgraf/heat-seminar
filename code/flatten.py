from heat import factories
import torch
from heat import resplit
from heat import DNDarray


def heat_flatten(a: DNDarray, start_dim=0):
    """
    Flattens an array into one dimension.
    WARNING: if a.split > 0, then the array must be resplit.
    Parameters
    ----------
    a : DNDarray
        array to collapse
    Returns
    -------
    ret : DNDarray
        flattened copy
    Examples
    --------
    >>> a = ht.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    >>> ht.flatten(a)
    tensor([1,2,3,4,5,6,7,8])
    """
    if a.split is None:
        return factories.array(
            torch.flatten(a._DNDarray__array, start_dim=start_dim), dtype=a.dtype, is_split=None, device=a.device, comm=a.comm
        )

    if a.split > 0:
        a = resplit(a, 0)

    a = factories.array(
        torch.flatten(a._DNDarray__array, start_dim=start_dim), dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm
    )
    a.balance_()

    return a