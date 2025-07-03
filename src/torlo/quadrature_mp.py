from mpmath import mp
from  precision_mp import precision
from  lobatto_quadrature import get_quadrature as lobatto_quadrature
from  legendre_quadrature import get_quadrature as legendre_quadrature
import numpy as np

mp.dps=precision

def orthogonal_legendre_polynomials_sym(x,k, mp_flag=True):
    """ Orthogonal Legendre polynomials in arbitrary precision defined on [-1,1] 
    up to degree 10
    """
    if mp_flag:
        if k==0:
            p= mp.mpf('1')
        elif k==1:
            p= x
        elif k==2:
            p= mp.mpf('0.5') * (mp.mpf(3) * x**2 - mp.mpf('1'))
        elif k==3:
            p= mp.mpf('0.5') * (mp.mpf('5') * x**3 - mp.mpf('3') * x)
        elif k==4:
            p= mp.mpf('0.125') * (mp.mpf('35') * x**4 - mp.mpf('30') * x**2 + mp.mpf('3'))
        elif k==5:
            p= mp.mpf('0.125') * (mp.mpf('63') * x**5 - mp.mpf('70') * x**3 + mp.mpf('15') * x)
        elif k==6:
            p= mp.mpf('0.0625') * (mp.mpf('231') * x**6 - mp.mpf('315') * x**4 + mp.mpf('105') * x**2 - mp.mpf('5'))
        elif k==7:
            p= mp.mpf('0.0625') * (mp.mpf('429') * x**7 - mp.mpf('693') * x**5 + mp.mpf('315') * x**3 - mp.mpf('35') * x)
        elif k==8:
            p= mp.mpf('0.0078125') * (mp.mpf('6435') * x**8 - mp.mpf('12012') * x**6 + mp.mpf('6930') * x**4 - mp.mpf('1260') * x**2 + mp.mpf('35'))
        elif k==9:
            return mp.mpf('0.0078125') * (mp.mpf('12155') * x**9 - mp.mpf('25740') * x**7 + mp.mpf('18018') * x**5 - mp.mpf('4620 ')* x**3 + mp.mpf('315 ')* x)
        elif k==10:
            p= mp.mpf('0.00390625') * (mp.mpf('46189') * x**10 - mp.mpf('109395') * x**8 + mp.mpf('90090')* x**6 - mp.mpf('30030') * x**4 + mp.mpf('3465') * x**2 - mp.mpf('63'))
        else:
            raise ValueError(f"Legendre polynomial not implemented for k={k}")
    else:
        if k == 0:
            p = 1.0
        elif k == 1:
            p = x
        elif k == 2:
            p = 0.5 * (3 * x**2 - 1)
        elif k == 3:
            p = 0.5 * (5 * x**3 - 3 * x)
        elif k == 4:
            p = 0.125 * (35 * x**4 - 30 * x**2 + 3)
        elif k == 5:
            p = 0.125 * (63 * x**5 - 70 * x**3 + 15 * x)
        elif k == 6:
            p = 0.0625 * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5)
        elif k == 7:
            p = 0.0625 * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x)
        elif k == 8:
            p = 0.0078125 * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35)
        elif k == 9:
            p = 0.0078125 * (12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x)
        elif k == 10:
            p = 0.00390625 * (46189 * x**10 - 109395 * x**8 + 90090 * x**6 - 30030 * x**4 + 3465 * x**2 - 63)
        else:
            raise ValueError(f"Legendre polynomial not implemented for k={k}")
    return p

def orthogonal_legendre_polynomials(x,k,mp_flag=True):
    """ Orthogonal Legendre polynomials in arbitrary precision defined on [0,1] 
    up to degree 10
    """
    if mp_flag:
        y=x*mp.mpf('2')-mp.mpf('1')
        return orthogonal_legendre_polynomials_sym(y,k, mp_flag=mp_flag)
    else:
        y=2*x-1.
        return orthogonal_legendre_polynomials_sym(y,k, mp_flag=mp_flag)
    

def derivative_orthogonal_legendre_polynomials_sym(x,k, mp_flag=True):
    """ Derivative of Orthogonal Legendre polynomials in arbitrary precision defined on [-1,1] 
    up to degree 10
    """
    if mp_flag:
        if k==0:
            return mp.mpf('0')
        elif k==1:
            return mp.mpf('1')
        elif k==2:
            return mp.mpf('0.5') * (mp.mpf('6') * x )
        elif k==3:
            return mp.mpf('0.5') * (mp.mpf('15') * x**2 - mp.mpf('3') )
        elif k==4:
            return mp.mpf('0.125') * (mp.mpf('35')*4 * x**3 - 2*mp.mpf('30') * x )
        elif k==5:
            return mp.mpf('0.125') * (mp.mpf('63')*5 * x**4 - 3*mp.mpf('70') * x**2 + mp.mpf('15') )
        elif k==6:
            return mp.mpf('0.0625') * (6*mp.mpf('231') * x**5 - 4*mp.mpf('315') * x**3 + 2*mp.mpf('105') * x)
        elif k==7:
            return mp.mpf('0.0625') * (7*mp.mpf('429') * x**6 - 5*mp.mpf('693') * x**4 + 3*mp.mpf('315') * x**2 - mp.mpf('35') )
        elif k==8:
            return mp.mpf('0.0078125') * (8*mp.mpf('6435') * x**7 - mp.mpf('12012')*6 * x**5 + 4*mp.mpf('6930') * x**3 - 2*mp.mpf('1260') * x )
        elif k==9:
            return mp.mpf('0.0078125') * (mp.mpf('12155') *9* x**8 - 7*mp.mpf('25740') * x**6 +5* mp.mpf('18018') * x**4 - 3*mp.mpf('4620 ')* x**2 + mp.mpf('315 '))
        elif k==10:
            return mp.mpf('0.00390625') * (10*mp.mpf('46189') * x**9 - 8*mp.mpf('109395') * x**7 + 6*mp.mpf('90090')* x**5 - 4*mp.mpf('30030') * x**3 + 2*mp.mpf('3465') * x )
        else:
            raise ValueError(f"Legendre polynomial not implemented for k={k}")
    else:
        if k==0:
            return 0
        elif k==1:
            return 1
        elif k==2:
            return 0.5 * (6 * x)
        elif k==3:
            return 0.5 * (15 * x**2 - 3)
        elif k==4:
            return 0.125 * (35 * x**3 - 30 * x)
        elif k==5:
            return 0.125 * (63 * x**4 - 70 * x**2 + 15)
        elif k==6:
            return 0.0625 * (231 * x**5 - 315 * x**3 + 105 * x)
        elif k==7:
            return 0.0625 * (429 * x**6 - 693 * x**4 + 315 * x**2 - 35)
        elif k==8:
            return 0.0078125 * (6435 * x**7 - 12012 * x**5 + 6930 * x**3 - 1260 * x)
        elif k==9:
            return 0.0078125 * (12155 * x**8 - 25740 * x**6 + 18018 * x**4 - 4620 * x**2 + 315)
        elif k==10:
            return 0.00390625 * (46189 * x**9 - 109395 * x**7 + 90090 * x**5 - 30030 * x**3 + 3465 * x)
        else:
            raise ValueError(f"Legendre polynomial not implemented for k={k}")


def derivative_orthogonal_legendre_polynomials(x,k, mp_flag=True):
    """ Derivative of Orthogonal Legendre polynomials in arbitrary precision defined on [0,1] 
    up to degree 10
    """
    if mp_flag:
        y=x*mp.mpf('2')-mp.mpf('1')
        return mp.mpf('2')*derivative_orthogonal_legendre_polynomials_sym(y,k, mp_flag=mp_flag)
    else:
        y=2*x-1
        return 2*derivative_orthogonal_legendre_polynomials_sym(y,k,mp_flag=mp_flag)


def lagrange_basis(nodes,x,k):
    lagr = mp.mpf(1)
    nodek=nodes[k]
    for j, xj in enumerate(nodes):
        if j!=k:
            lagr = lagr * (x- nodes[j] )/(nodek-nodes[j])
    return lagr

def lagrange_deriv(nodes,x,k):
    f=mp.mpf(0)
    for j in range(len(nodes)):
        p=mp.mpf(1)
        if k!=j:
            for l in range(len(nodes)):
                if l!=k and l!=j: 
                    p=p*(x-nodes[l])/(nodes[k]-nodes[l])
            f = f + p/(nodes[k]-nodes[j])
    return f

def sort_quad(nodes,weights):
    idxs = np.argsort(nodes)
    nodes_new = mp.zeros(nodes.rows,nodes.cols)
    weights_new = mp.zeros(weights.rows,weights.cols)
    for i in range(len(nodes)):
        nodes_new[i] = nodes[idxs[i],0]
        weights_new[i] = weights[idxs[i],0]
    return nodes_new, weights_new

def lobatto(n):
    nodes, weights = lobatto_quadrature(n)
    nodes, weights = sort_quad(nodes, weights)
    return nodes, weights

def legendre(n):
    nodes, weights = legendre_quadrature(n)
    nodes, weights = sort_quad(nodes, weights)
    return nodes, weights

def equispaced(n):
    n=int(n)
    nodes = mp.matrix(mp.linspace(-1,1,n))
    quad_n, quad_w = legendre_quadrature(n)
    weights = mp.zeros(n,1)
    for i in range(n):
        for j in range(len(quad_n)):
            weights[i] = weights[i]+lagrange_basis(nodes,quad_n[j],i)*quad_w[j]
    return nodes, weights


def convert_mp_2_np(M):
    Mnp = np.zeros((M.rows,M.cols))
    for i in range(M.rows):
        for j in range(M.cols):
            Mnp[i,j]=float(M[i,j])
    return Mnp