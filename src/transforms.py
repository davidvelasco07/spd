import numpy as np

# Axis conventions (ellipsis absorbs all leading batch axes: nvar, optional
# nader, optional Nblocks, ...). Spatial element axes are z/y/x, with per-element
# point axes k/j/i (outputs) or n/m/l (inputs).
#   ndim==3:  ...zyx + nml (input) -> ...zyx + kji (output)
#   ndim==2:  ...yx  + ml  (input) -> ...yx  + ji  (output)
#   ndim==1:  ...x   + l   (input) -> ...x   + i   (output)


def compute_A_from_B(B, A_to_B, dim, ndim, ader=False) -> np.ndarray:
    """Apply per-dim transform (p+1 -> p+1 or p+2) along `dim`.

    `ader` is accepted for backward compatibility but is no longer needed —
    ellipsis absorbs the ADER time axis and any other leading batch axes.
    """
    y = ("", "y")[ndim > 1]
    j = ("", "j")[ndim > 1]
    z = ("", "z")[ndim > 2]
    k = ("", "k")[ndim > 2]
    u = f"...{z}{y}x"
    if dim == "x":
        u += f"{k}{j}"
        return np.einsum(f"fs,{u}s->{u}f", A_to_B, B)
    if dim == "y" and ndim > 1:
        u += f"{k}"
        return np.einsum(f"fs,{u}si->{u}fi", A_to_B, B)
    if dim == "z" and ndim > 2:
        return np.einsum(f"fs,{u}sji->{u}fji", A_to_B, B)
    raise ValueError(f"Wrong option for dim: {dim!r} (ndim={ndim})")


def compute_A_from_B_full(B, A_to_B, ndim) -> np.ndarray:
    """Apply per-dim transform to all ndim spatial directions at once."""
    if ndim == 3:
        return np.einsum("kn,jm,il,...zyxnml->...zyxkji",
                         A_to_B, A_to_B, A_to_B, B)
    if ndim == 2:
        return np.einsum("jm,il,...yxml->...yxji",
                         A_to_B, A_to_B, B)
    return np.einsum("il,...xl->...xi", A_to_B, B)
