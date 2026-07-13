import numpy as np

# ----------------------------------------------------------------------
# Double Mach Reflection (Woodward & Colella 1984)
# ----------------------------------------------------------------------
# A Mach-10 shock in gamma=1.4 air meets a reflecting wall.  The shock
# starts at x = DMR_XC on the bottom boundary, inclined at DMR_ANGLE to
# the wall.  These constants are shared by the initial condition and the
# "doublemach" boundary conditions.
DMR_XC = 1.0 / 6.0           # shock foot on the bottom wall
DMR_ANGLE = np.pi / 3.0      # 60 degrees from the x-axis
DMR_SHOCK_SPEED = 10.0       # shock speed factor used for the moving trace


def dmr_post_shock(gamma=1.4):
    """Post-shock primitive state behind the Mach-10 shock."""
    return dict(
        rho=8.0,
        vx=8.25 * np.cos(np.pi / 6),   # 8.25 * cos(30 deg) ~ 7.1447
        vy=-8.25 * np.sin(np.pi / 6),  # -8.25 * sin(30 deg) = -4.125
        P=116.5,
    )


def dmr_ambient(gamma=1.4):
    """Undisturbed (ahead-of-shock) primitive state."""
    return dict(rho=1.4, vx=0.0, vy=0.0, P=1.0)


def dmr_shock_x(t, y, xc=DMR_XC, angle=DMR_ANGLE, speed=DMR_SHOCK_SPEED):
    """x-position of the (tilted, moving) shock front at height ``y``/time ``t``."""
    return speed * t / np.sin(angle) + xc + y / np.tan(angle)


def double_mach_reflection(
    xy: np.ndarray, case: int, gamma=1.4
) -> np.ndarray:
    """Initial condition for the Double Mach Reflection problem.

    Behind the initial shock line ``x < xc + y/tan(angle)`` the flow is in
    the post-shock state; ahead of it the flow is at rest (ambient).
    Recommended domain: ``[0, 4] x [0, 1]``.
    """
    x = xy[0]
    y = xy[1]
    behind = x < (DMR_XC + y / np.tan(DMR_ANGLE))
    ps = dmr_post_shock(gamma)
    am = dmr_ambient(gamma)
    if case == 0:
        return np.where(behind, ps["rho"], am["rho"])
    elif case == 1:
        return np.where(behind, ps["vx"], am["vx"])
    elif case == 2:
        return np.where(behind, ps["vy"], am["vy"])
    elif case == 4:
        return np.where(behind, ps["P"], am["P"])
    else:
        return np.zeros(x.shape)


def step_function(xy: np.ndarray,case: int, vx=1, vy=1, P=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,
                        np.where(np.fabs(y-0.5)<0.25,2,1),1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)
    
def sine_wave(xy: np.ndarray,case: int, A=0.125, vx=1, vy=1, P=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return 1.0+A*(np.sin(2*np.pi*(x+y)))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)

def orszag_tang(xy: np.ndarray, case: int, gamma=5.0 / 3.0):
    """Orszag-Tang vortex (ideal MHD), periodic BCs, domain [-1/2, 1/2]^2.

    Variable order: rho, vx, vy, vz, P, Bx, By, Bz.  The cell-centered B
    returned here is only an initial guess; the divergence-free staggered
    field must come from :func:`orszag_tang_Az` via the vector potential.

    The canonical setup is defined on [0, 1]^2; the half-box coordinate
    shift below maps it onto the centered domain so the central current
    sheet (and the plasmoid it may spawn) sits at the origin, matching the
    standard figures in the literature.
    """
    x = xy[0] + 0.5
    y = xy[1] + 0.5
    # Code units (magnetic pressure B^2/2): the canonical pairing is
    # rho = gamma^2, p = gamma, B0 = 1 (equivalent to the Gaussian-units
    # setup rho = 25/36pi, p = 5/12pi, B0 = 1/sqrt(4pi)).
    B0 = 1.0
    if case == 0:
        return gamma**2 * np.ones(x.shape)
    elif case == 1:
        return -np.sin(2 * np.pi * y)
    elif case == 2:
        return np.sin(2 * np.pi * x)
    elif case == 4:
        return gamma * np.ones(x.shape)
    elif case == 5:
        return -B0 * np.sin(2 * np.pi * y)
    elif case == 6:
        return B0 * np.sin(4 * np.pi * x)
    else:
        return np.zeros(x.shape)


def orszag_tang_Az(mesh: np.ndarray, j: int):
    """z-component of the vector potential for the Orszag-Tang field:
    Bx = dAz/dy, By = -dAz/dx.  Same half-box shift as :func:`orszag_tang`
    (centered domain [-1/2, 1/2]^2)."""
    x = mesh[0] + 0.5
    y = mesh[1] + 0.5
    B0 = 1.0
    if j == 2:
        return B0 * (
            np.cos(4 * np.pi * x) / (4 * np.pi) + np.cos(2 * np.pi * y) / (2 * np.pi)
        )
    return np.zeros(x.shape)


def RTI(
    xy: np.ndarray,
    case: int,
    P0=1.0,
    gamma=5 / 3,
    g=-1.0,
    rho1=2.0,
    rho2=1.0,
    yc=0.5,
) -> np.ndarray:
    """Rayleigh-Taylor instability (single-mode, hydrostatic background).

    A heavy fluid ``rho1`` sits on top of a light fluid ``rho2`` at the
    interface ``y = yc`` in a constant downward gravitational field, with a
    small single-mode vertical velocity perturbation seeded at the interface.

    The gravitational potential is returned for ``case == -1`` so the same
    function can be passed as ``init_fct`` with ``potential=True``.
    """
    x = xy[0]
    y = xy[1]
    if case == 0:
        # density
        return np.where(y > yc, rho2, rho1)
    elif case == 1:
        # vx
        return np.zeros(x.shape)
    elif case == 2:
        # vy: single-mode perturbation scaled by the local sound speed
        dv = np.sqrt(gamma * (P0 + rho1 * yc - P0 + 1) / rho1)
        return -0.025 * dv * np.cos(8 * np.pi * x)
    elif case == 4:
        # Pressure (hydrostatic equilibrium: dP/dy = rho * g, with g = -1)
        return np.where(
            y > yc, P0 + rho2 * y + (rho1 - rho2) * yc, P0 + rho1 * y
        )
    elif case == -1:
        # Gravitational potential phi (acceleration g_y = -dphi/dy = g)
        return g * y
    else:
        return np.zeros(x.shape)


def implosion(
    xy: np.ndarray,
    case: int,
    rho_in=0.125,
    P_in=0.14,
    rho_out=1.0,
    P_out=1.0,
    diag=0.15,
) -> np.ndarray:
    """Implosion problem (Liska & Wendroff 2003, sec. 4.7).

    A low-density, low-pressure triangle in the corner (below the diagonal
    ``x + y = diag``) of an otherwise uniform gas at rest.  The inward-moving
    shock reflects off the origin and a jet forms along the diagonal.

    Recommended setup: domain ``[0, 0.3] x [0, 0.3]``, reflective walls on
    all four sides, ``gamma = 1.4``.  Preserving the x <-> y symmetry of the
    jet to late times (t = 2.5) is the hard part of this test.
    """
    x = xy[0]
    y = xy[1]
    inside = x + y < diag
    if case == 0:
        return np.where(inside, rho_in, rho_out)
    elif case == 4:
        return np.where(inside, P_in, P_out)
    else:
        return np.zeros(x.shape)


def KH_instability(xy: np.ndarray, case: int) -> np.ndarray:
    y=xy[1]
    w0=0.1
    sigma = 0.05/np.sqrt(2)
    if case==0:
        return np.where(y<0.25,1,np.where(y<0.75,2,1))
    elif case==1:
        return np.where(y<0.25,-0.5,np.where(y<0.75,0.5,-0.5))
    elif case==2:
        return w0*np.sin(4*np.pi*xy[0])*(np.exp(-(y-0.25)**2/(2*sigma**2))+np.exp(-(y-0.75)**2/(2*sigma**2)))
    elif case==4:
        #Pressure
        return 2.5*np.ones(y.shape)
    else:
        return np.zeros(xy[0].shape)
