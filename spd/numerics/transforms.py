import numpy as np

from spd.runtime.gpu import CUTENSOR_ENABLED, is_gpu_array

if CUTENSOR_ENABLED:
    import cupy as cp
    from cupyx import cutensor


def _contract(A, modes_A, B, modes_B, modes_C, out=None):
    """C[modes_C] = A[modes_A] * B[modes_B] via cuTENSOR (out may be strided)."""
    if out is None:
        sizes = {}
        for m, s in zip(modes_A, A.shape):
            sizes[m] = s
        for m, s in zip(modes_B, B.shape):
            sizes[m] = s
        out = cp.empty(tuple(sizes[m] for m in modes_C), dtype=B.dtype)
    return cutensor.contraction(
        1.0, A, tuple(modes_A), B, tuple(modes_B), 0.0, out, tuple(modes_C)
    )


def compute_A_from_B(B, A_to_B, dim, ndim, ader=False) -> np.ndarray:
    """Apply per-dim transform along `dim`.

    `ader` is kept for backward compatibility; the equation uses ellipsis so
    optional leading axes (e.g. ADER/time, block batches) are naturally handled.
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
    # optimize=True lets numpy/cupy contract matrices one at a time and keeps
    # CPU performance reasonable for higher-order operators.
    if ndim == 3:
        return np.einsum(
            "kn,jm,il,...zyxnml->...zyxkji",
            A_to_B,
            A_to_B,
            A_to_B,
            B,
            optimize=True,
        )
    if ndim == 2:
        return np.einsum(
            "jm,il,...yxml->...yxji",
            A_to_B,
            A_to_B,
            B,
            optimize=True,
        )
    return np.einsum("il,...xl->...xi", A_to_B, B)

def _fv_view(A_fv, B, ndim):
        """View a contiguous FV cell-based array (u, ..., N*n) with the
        element and point axes split ((u, ..., N, n)); a free reshape."""
        shape = [A_fv.shape[0]]
        for ax in range(ndim):
            N, n = B.shape[1 + ax], B.shape[1 + ndim + ax]
            shape += [N, n]
        return A_fv.reshape(shape)

def compute_A_from_B_full_fv(B,A_to_B,ndim,out=None) -> np.ndarray:
        """Same contraction as compute_A_from_B_full, but the output axes are
        interleaved as (element, point) per dimension so that a free reshape
        yields the FV cell-based layout, with no transpose+copy afterwards.

        Input : B (u, Nz, Ny, Nx, n, n, n)   [element-based/SD layout]
        Output: A (u, Nz*k, Ny*j, Nx*i)      [cell-based/FV layout]

        When ``out`` (FV cell-based layout) is given, the result is written
        into it in place -- via cuTENSOR on the GPU -- so a persistent buffer
        can be reused with a stable pointer and no per-call allocation.
        """
        u = B.shape[0]
        Ns = [B.shape[1 + ax] * B.shape[1 + ndim + ax] for ax in range(ndim)]
        out_v = None if out is None else _fv_view(out, B, ndim)
        if CUTENSOR_ENABLED and is_gpu_array(B):
            if ndim == 3:
                T1 = _contract(A_to_B, "il", B, "uzyxnml", "uzyxnmi")
                T2 = _contract(A_to_B, "jm", T1, "uzyxnmi", "uzyxnji")
                A = _contract(A_to_B, "kn", T2, "uzyxnji", "uzkyjxi", out=out_v)
            elif ndim == 2:
                T1 = _contract(A_to_B, "il", B, "uyxml", "uyxmi")
                A = _contract(A_to_B, "jm", T1, "uyxmi", "uyjxi", out=out_v)
            else:
                A = _contract(A_to_B, "il", B, "uxl", "uxi", out=out_v)
            return out if out is not None else A.reshape([u] + Ns)
        if ndim==3:
            A = np.einsum("kn,jm,il,uzyxnml->uzkyjxi",
                         A_to_B,
                         A_to_B,
                         A_to_B, B, optimize=True, out=out_v)
        elif ndim==2:
            A = np.einsum("jm,il,uyxml->uyjxi",
                         A_to_B,
                         A_to_B, B, optimize=True, out=out_v)
        else:
            A = np.einsum("il,uxl->uxi",
                         A_to_B, B, out=out_v)
        return out if out is not None else A.reshape([u] + Ns)

def compute_A_from_B_full_from_fv(B_fv,A_to_B,ndim,shape_sd,out=None) -> np.ndarray:
        """Inverse-layout counterpart of :func:`compute_A_from_B_full_fv`:
        contract an FV cell-based array directly into the SD element-based
        layout, fusing the transpose_to_sd into the einsum.

        Input : B_fv (u, Nz*n, Ny*m, Nx*l)      [cell-based/FV layout]
        Output: A    (u, Nz, Ny, Nx, k, j, i)   [element-based/SD layout]

        ``shape_sd`` is the element-based shape of the result (used to split
        the fused FV axes); ``out`` optionally receives the result in place.
        """
        B = B_fv.reshape(
            [B_fv.shape[0]]
            + [s for ax in range(ndim) for s in (shape_sd[1 + ax], shape_sd[1 + ndim + ax])]
        )
        if CUTENSOR_ENABLED and is_gpu_array(B):
            if ndim == 3:
                T1 = _contract(A_to_B, "il", B, "uznymxl", "uznymxi")
                T2 = _contract(A_to_B, "jm", T1, "uznymxi", "uznyxji")
                return _contract(A_to_B, "kn", T2, "uznyxji", "uzyxkji", out=out)
            elif ndim == 2:
                T1 = _contract(A_to_B, "il", B, "uymxl", "uymxi")
                return _contract(A_to_B, "jm", T1, "uymxi", "uyxji", out=out)
            return _contract(A_to_B, "il", B, "uxl", "uxi", out=out)
        if ndim == 3:
            return np.einsum("kn,jm,il,uznymxl->uzyxkji",
                             A_to_B, A_to_B, A_to_B, B, optimize=True, out=out)
        elif ndim == 2:
            return np.einsum("jm,il,uymxl->uyxji",
                             A_to_B, A_to_B, B, optimize=True, out=out)
        return np.einsum("il,uxl->uxi", A_to_B, B, out=out)
