import numpy as np
from . import mhd


def _hlld_fan(rho_L, u_L, pT_L, rho_R, u_R, pT_R, Bn, S_L, S_R):
    """
    Outer/entropy-wave structure of the HLLD fan (Miyoshi & Kusano 2005).

    Returns (S_M, pT_s, rho_sL, rho_sR, S_Ls, S_Rs) with the contact speed
    (eq. 38), the uniform star total pressure (eq. 41), the star densities
    (eq. 43) and the Alfven wave speeds (eq. 51).
    """
    dSu_L = S_L - u_L
    dSu_R = S_R - u_R
    denom = dSu_R * rho_R - dSu_L * rho_L
    S_M = (dSu_R * rho_R * u_R - dSu_L * rho_L * u_L - pT_R + pT_L) / denom
    pT_s = (
        dSu_R * rho_R * pT_L
        - dSu_L * rho_L * pT_R
        + rho_L * rho_R * dSu_R * dSu_L * (u_R - u_L)
    ) / denom
    rho_sL = rho_L * dSu_L / (S_L - S_M)
    rho_sR = rho_R * dSu_R / (S_R - S_M)
    S_Ls = S_M - np.abs(Bn) / np.sqrt(rho_sL)
    S_Rs = S_M + np.abs(Bn) / np.sqrt(rho_sR)
    return S_M, pT_s, rho_sL, rho_sR, S_Ls, S_Rs


def _hlld_star_factors(rho, u, S, S_M, Bn):
    """
    Common factors of the star-region transverse fields (eqs. 44-47):

        vt_s = vt - fac_v * Bt        Bt_s = fac_b * Bt

    In the degenerate case (S_M -> u with |Bn| -> sqrt(gamma p rho), where
    the denominator vanishes and Bt -> 0) the L/R state is left untouched
    (fac_v = 0, fac_b = 1), as prescribed by Miyoshi & Kusano.
    """
    dSu = S - u
    denom = rho * dSu * (S - S_M) - Bn**2
    scale = rho * dSu**2 + Bn**2
    safe = np.abs(denom) > 1e-12 * scale
    denom = np.where(safe, denom, 1.0)
    fac_v = np.where(safe, Bn * (S_M - u) / denom, 0.0)
    fac_b = np.where(safe, (rho * dSu**2 - Bn**2) / denom, 1.0)
    return fac_v, fac_b


def _select_fan(S_L, S_Ls, S_M, S_Rs, S_R, qL, qsL, qssL, qssR, qsR, qR):
    """Value at xi = 0 across the six regions of the HLLD fan."""
    left = np.where(S_L > 0, qL, np.where(S_Ls >= 0, qsL, qssL))
    right = np.where(S_R < 0, qR, np.where(S_Rs <= 0, qsR, qssR))
    return np.where(S_M >= 0, left, right)


def hlld(W_L: np.ndarray,
         W_R: np.ndarray,
         U_L: np.ndarray,
         U_R: np.ndarray,
         F,
         vels: np.array,
         _p_: int,
         gamma: float,
         min_c2: float,
         **kwargs,
    ) -> np.ndarray:
    """
    HLLD approximate Riemann solver for ideal MHD (Miyoshi & Kusano 2005),
    with the outer wave-speed estimate of their eq. (67).  The normal field
    inside the fan is the single-valued average of the two interface traces.
    Passive scalars (and the thermal-diffusion row) are advected with the
    contact discontinuity.
    """
    _d_ = 0
    na = np.newaxis
    v1, v2, v3 = vels[0], vels[1], vels[2]
    shift = _p_ + kwargs["thdiffusion"] + kwargs["npassive"]
    b1, b2, b3 = v1 + shift, v2 + shift, v3 + shift

    rho_L, rho_R = W_L[_d_], W_R[_d_]
    u_L, u_R = W_L[v1], W_R[v1]
    p_L, p_R = W_L[_p_], W_R[_p_]
    Bn = 0.5 * (W_L[b1] + W_R[b1])

    c_L = mhd.compute_fast_vel(p_L, rho_L, W_L[b1], W_L[b2], W_L[b3],
                               gamma=gamma, min_c2=min_c2)
    c_R = mhd.compute_fast_vel(p_R, rho_R, W_R[b1], W_R[b2], W_R[b3],
                               gamma=gamma, min_c2=min_c2)
    c_max = np.maximum(c_L, c_R)
    S_L = np.minimum(u_L, u_R) - c_max
    S_R = np.maximum(u_L, u_R) + c_max

    pT_L = p_L + 0.5 * (W_L[b1]**2 + W_L[b2]**2 + W_L[b3]**2)
    pT_R = p_R + 0.5 * (W_R[b1]**2 + W_R[b2]**2 + W_R[b3]**2)

    S_M, pT_s, rho_sL, rho_sR, S_Ls, S_Rs = _hlld_fan(
        rho_L, u_L, pT_L, rho_R, u_R, pT_R, Bn, S_L, S_R
    )

    F_L = mhd.compute_fluxes(W_L, vels, _p_, gamma, **kwargs)
    F_R = mhd.compute_fluxes(W_R, vels, _p_, gamma, **kwargs)

    sgn = np.sign(Bn)
    sr_L = np.sqrt(rho_sL)
    sr_R = np.sqrt(rho_sR)
    sr_den = sr_L + sr_R

    def star_state(W, U, rho, u, pT, rho_s, S):
        """U* of one side (eqs. 43-48); transverse rows via _hlld_star_factors."""
        fac_v, fac_b = _hlld_star_factors(rho, u, S, S_M, Bn)
        U_s = U.copy()
        U_s[_d_] = rho_s
        vt1_s = W[v2] - fac_v * W[b2]
        vt2_s = W[v3] - fac_v * W[b3]
        Bt1_s = fac_b * W[b2]
        Bt2_s = fac_b * W[b3]
        U_s[v1] = rho_s * S_M
        U_s[v2] = rho_s * vt1_s
        U_s[v3] = rho_s * vt2_s
        U_s[b1] = Bn
        U_s[b2] = Bt1_s
        U_s[b3] = Bt2_s
        vB = u * Bn + W[v2] * W[b2] + W[v3] * W[b3]
        vB_s = S_M * Bn + vt1_s * Bt1_s + vt2_s * Bt2_s
        U_s[_p_] = (
            (S - u) * U[_p_] - pT * u + pT_s * S_M + Bn * (vB - vB_s)
        ) / (S - S_M)
        for r in range(_p_ + 1, shift + 1):
            U_s[r] = rho_s * (U[r] / rho)
        return U_s, vt1_s, vt2_s, Bt1_s, Bt2_s, vB_s

    U_sL, vt1_sL, vt2_sL, Bt1_sL, Bt2_sL, vB_sL = star_state(
        W_L, U_L, rho_L, u_L, pT_L, rho_sL, S_L
    )
    U_sR, vt1_sR, vt2_sR, Bt1_sR, Bt2_sR, vB_sR = star_state(
        W_R, U_R, rho_R, u_R, pT_R, rho_sR, S_R
    )

    # Double-star (Alfven-averaged) region, eqs. (59)-(63).
    vt1_ss = (sr_L * vt1_sL + sr_R * vt1_sR + sgn * (Bt1_sR - Bt1_sL)) / sr_den
    vt2_ss = (sr_L * vt2_sL + sr_R * vt2_sR + sgn * (Bt2_sR - Bt2_sL)) / sr_den
    Bt1_ss = (
        sr_L * Bt1_sR + sr_R * Bt1_sL + sgn * sr_L * sr_R * (vt1_sR - vt1_sL)
    ) / sr_den
    Bt2_ss = (
        sr_L * Bt2_sR + sr_R * Bt2_sL + sgn * sr_L * sr_R * (vt2_sR - vt2_sL)
    ) / sr_den
    vB_ss = S_M * Bn + vt1_ss * Bt1_ss + vt2_ss * Bt2_ss

    def dstar_state(U_s, rho_s, vB_s, side):
        U_ss = U_s.copy()
        U_ss[v2] = rho_s * vt1_ss
        U_ss[v3] = rho_s * vt2_ss
        U_ss[b2] = Bt1_ss
        U_ss[b3] = Bt2_ss
        U_ss[_p_] = U_s[_p_] + side * np.sqrt(rho_s) * sgn * (vB_s - vB_ss)
        return U_ss

    U_ssL = dstar_state(U_sL, rho_sL, vB_sL, -1.0)
    U_ssR = dstar_state(U_sR, rho_sR, vB_sR, +1.0)

    F = _select_fan(
        S_L[na], S_Ls[na], S_M[na], S_Rs[na], S_R[na],
        F_L,
        F_L + S_L[na] * (U_sL - U_L),
        F_L + S_L[na] * (U_sL - U_L) + S_Ls[na] * (U_ssL - U_sL),
        F_R + S_R[na] * (U_sR - U_R) + S_Rs[na] * (U_ssR - U_sR),
        F_R + S_R[na] * (U_sR - U_R),
        F_R,
    )
    return F


def hlld_E(E_L, E_R, vel, gamma, min_c2):
    """
    One dimensional sweep of the edge electric-field Riemann problem using
    the HLLD fan (dimension-by-dimension UCT).

    The 8-component edge state is [E, B1, B2, v1, v2, B3, rho, p] (code
    convention E = v1*B2 - v2*B1).  ``vel`` selects the sweep:

    - vel = 3: interfaces normal to dim1; the normal velocity is v1, the CT
      normal field is B1 (continuous) and the discontinuous transverse field
      is B2, whose 1D induction flux is exactly +E.
    - vel = 4: interfaces normal to dim2; normal velocity v2, normal field
      B2, transverse field B1 with induction flux -E.

    Returns the resolved 8-component state at xi = 0: slot 0 holds the
    upwind E (the fan's transverse-field flux, sign adjusted) and the other
    slots hold the fan state, consumed by the second sweep.  The velocity
    parallel to the edge is not part of the state, so the B3 double-star
    value drops the (v3_R* - v3_L*) term; B3 only enters the next sweep's
    wave-speed estimate.
    """
    if vel == 3:
        _u_, _vt_, _bn_, _bt_ = 3, 4, 1, 2
        e_sign = 1.0
    elif vel == 4:
        _u_, _vt_, _bn_, _bt_ = 4, 3, 2, 1
        e_sign = -1.0
    else:
        raise ValueError(f"Invalid sweep velocity slot: {vel}")

    rho_L, rho_R = E_L[6], E_R[6]
    p_L, p_R = E_L[7], E_R[7]
    u_L, u_R = E_L[_u_], E_R[_u_]
    vt_L, vt_R = E_L[_vt_], E_R[_vt_]
    Bt_L, Bt_R = E_L[_bt_], E_R[_bt_]
    B3_L, B3_R = E_L[5], E_R[5]
    Bn = 0.5 * (E_L[_bn_] + E_R[_bn_])

    c_L = mhd.compute_fast_vel(p_L, rho_L, E_L[_bn_], Bt_L, B3_L,
                               gamma=gamma, min_c2=min_c2)
    c_R = mhd.compute_fast_vel(p_R, rho_R, E_R[_bn_], Bt_R, B3_R,
                               gamma=gamma, min_c2=min_c2)
    c_max = np.maximum(c_L, c_R)
    S_L = np.minimum(u_L, u_R) - c_max
    S_R = np.maximum(u_L, u_R) + c_max

    pT_L = p_L + 0.5 * (E_L[_bn_]**2 + Bt_L**2 + B3_L**2)
    pT_R = p_R + 0.5 * (E_R[_bn_]**2 + Bt_R**2 + B3_R**2)

    S_M, pT_s, rho_sL, rho_sR, S_Ls, S_Rs = _hlld_fan(
        rho_L, u_L, pT_L, rho_R, u_R, pT_R, Bn, S_L, S_R
    )

    fac_vL, fac_bL = _hlld_star_factors(rho_L, u_L, S_L, S_M, Bn)
    fac_vR, fac_bR = _hlld_star_factors(rho_R, u_R, S_R, S_M, Bn)
    vt_sL = vt_L - fac_vL * Bt_L
    vt_sR = vt_R - fac_vR * Bt_R
    Bt_sL = fac_bL * Bt_L
    Bt_sR = fac_bR * Bt_R
    B3_sL = fac_bL * B3_L
    B3_sR = fac_bR * B3_R

    sgn = np.sign(Bn)
    sr_L = np.sqrt(rho_sL)
    sr_R = np.sqrt(rho_sR)
    sr_den = sr_L + sr_R
    vt_ss = (sr_L * vt_sL + sr_R * vt_sR + sgn * (Bt_sR - Bt_sL)) / sr_den
    Bt_ss = (
        sr_L * Bt_sR + sr_R * Bt_sL + sgn * sr_L * sr_R * (vt_sR - vt_sL)
    ) / sr_den
    B3_ss = (sr_L * B3_sR + sr_R * B3_sL) / sr_den

    def sel(qL, qsL, qssL, qssR, qsR, qR):
        return _select_fan(S_L, S_Ls, S_M, S_Rs, S_R,
                           qL, qsL, qssL, qssR, qsR, qR)

    # Induction flux of the transverse field: F(Bt) = e_sign * E.
    G_L = e_sign * E_L[0]
    G_R = e_sign * E_R[0]
    G_sL = G_L + S_L * (Bt_sL - Bt_L)
    G_sR = G_R + S_R * (Bt_sR - Bt_R)
    G_ssL = G_sL + S_Ls * (Bt_ss - Bt_sL)
    G_ssR = G_sR + S_Rs * (Bt_ss - Bt_sR)
    E_res = e_sign * sel(G_L, G_sL, G_ssL, G_ssR, G_sR, G_R)

    Es = np.empty_like(E_L)
    Es[0] = E_res
    Es[_bn_] = Bn
    Es[_bt_] = sel(Bt_L, Bt_sL, Bt_ss, Bt_ss, Bt_sR, Bt_R)
    Es[_u_] = sel(u_L, S_M, S_M, S_M, S_M, u_R)
    Es[_vt_] = sel(vt_L, vt_sL, vt_ss, vt_ss, vt_sR, vt_R)
    Es[5] = sel(B3_L, B3_sL, B3_ss, B3_ss, B3_sR, B3_R)
    Es[6] = sel(rho_L, rho_sL, rho_sL, rho_sR, rho_sR, rho_R)
    # The gas pressure in the star region is not defined by HLLD; the
    # contact-upwinded value is used (it only feeds wave-speed estimates).
    Es[7] = np.where(S_M >= 0, p_L, p_R)
    return Es


def llf(W_L: np.ndarray,
        W_R: np.ndarray,
        U_L: np.ndarray,
        U_R: np.ndarray,
        F,
        vels: np.array,
        _p_: int,
        gamma: float,
        min_c2: float,
        **kwargs,
    ) -> np.ndarray:
        #Density index
        _d_=0 
        v_1 = vels[0]
        _b_ = _p_ + kwargs["thdiffusion"] + kwargs["npassive"]
        
        B1_L = W_L[v_1+_b_] 
        B1_R = W_R[v_1+_b_]
        B2_L = W_L[vels[1]+_b_] 
        B2_R = W_R[vels[1]+_b_]
        B3_L = W_L[vels[2]+_b_] 
        B3_R = W_R[vels[2]+_b_]

        c_L = mhd.compute_fast_vel(W_L[_p_],W_L[0], B1_L, B2_L, B3_L,gamma=gamma, min_c2=min_c2)
        c_R = mhd.compute_fast_vel(W_R[_p_],W_R[0], B1_R, B2_R, B3_R,gamma=gamma, min_c2=min_c2)
        c_max = np.maximum(abs(W_L[v_1]),abs(W_R[v_1])) + np.maximum(c_L,c_R)
    
        F_L = mhd.compute_fluxes(W_L,vels,_p_,gamma,**kwargs)
        F_R = mhd.compute_fluxes(W_R,vels,_p_,gamma,**kwargs)

        F = 0.5*(F_R+F_L)-0.5*c_max[np.newaxis]*(U_R-U_L)
        return F