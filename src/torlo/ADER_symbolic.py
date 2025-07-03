import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from src.quadr import lglnodes,equispaced
from scipy.interpolate import lagrange
import sympy

tol = 1e-15

def lagrange_basis(X,t,i):
    if t in X:
        if t == X[i]:
            return 1
        else:
            return 0
    if len(X)==1:
        return 1.0+0.0*t
    else:
        return sympy.prod((t-X[j])/(X[i]-X[j]) for j in range(len(X)) if j != i)


            
def equispaced(order):
    nodes=sympy.zeros(order,1)
    for k in range(0,order):
        nodes[k]=sympy.Integer(-1)+sympy.Integer(k)*sympy.Rational(2,order-1)
    w=sympy.zeros(1,order)
    x=sympy.Symbol("x")
    for k in range(order):
        zz=sympy.Integer(1)
        for s in range(order):
            if s == k:
                continue
            else:
                zz=zz*(x-nodes[s])# Polynomial([-nodes[s],1])
        zz=zz/zz.subs(x,nodes[k])
        w[k]=sympy.integrate(zz,(x,-1,1))
    return nodes,w


def my_gausslobatto(order):
    nodes=sympy.zeros(order,1)
    w=sympy.zeros(1,order)
    if order==2:
        nodes[0] = sympy.Integer(-1)
        nodes[1] = sympy.Integer(1)
        w[0] = sympy.Rational(1,2)
        w[1] = sympy.Rational(1,2)
    elif order==3:
        nodes[0] = sympy.Integer(-1)
        nodes[1] = sympy.Integer(0)
        nodes[2] = sympy.Integer(1)
        w[0] = sympy.Rational(1,3)
        w[1] = sympy.Rational(4,3)
        w[2] = sympy.Rational(1,3)
    elif order==4:
        nodes[0] = sympy.Integer(-1)
        nodes[1] = sympy.sympify("-sqrt(1/5)")
        nodes[2] = sympy.sympify("sqrt(1/5)")
        nodes[3] = sympy.Integer(1)
        w[0] = sympy.Rational(1,6)
        w[1] = sympy.Rational(5,6)
        w[2] = sympy.Rational(5,6)
        w[3] = sympy.Rational(1,6)
    elif order==5:
        nodes[0] = sympy.Integer(-1)
        nodes[1] = sympy.sympify("-sqrt(3/7)")
        nodes[2] = sympy.sympify("0")
        nodes[3] = sympy.sympify("sqrt(3/7)")
        nodes[4] = sympy.Integer(1)
        w[0] = sympy.Rational(1,10)
        w[1] = sympy.Rational(49,90)
        w[2] = sympy.Rational(32,45)
        w[3] = sympy.Rational(49,90)
        w[4] = sympy.Rational(1,10)
    elif order==6:
        nodes[0] = sympy.Integer(-1)
        nodes[1] = sympy.sympify("-sqrt(1/3+2*sqrt(7)/21)")
        nodes[2] = sympy.sympify("-sqrt(1/3-2*sqrt(7)/21)")
        nodes[3] = sympy.sympify(" sqrt(1/3-2*sqrt(7)/21)")
        nodes[4] = sympy.sympify(" sqrt(1/3+2*sqrt(7)/21)")
        nodes[5] = sympy.Integer(1)
        w[0] = sympy.Rational(1,15)
        w[1] = sympy.sympify("(14-sqrt(7))/30")
        w[2] = sympy.sympify("(14+sqrt(7))/30")
        w[3] = sympy.sympify("(14+sqrt(7))/30")
        w[4] = sympy.sympify("(14-sqrt(7))/30")
        w[5] = sympy.Rational(1,15)
    elif order==7:
        nodes[0] = sympy.Integer(-1)
        nodes[1] = sympy.sympify("-sqrt(5/11+2*sqrt(5/3)/11)")
        nodes[2] = sympy.sympify("-sqrt(5/11-2*sqrt(5/3)/11)")
        nodes[3] = sympy.sympify("0")
        nodes[4] = sympy.sympify(" sqrt(5/11-2*sqrt(5/3)/11)")
        nodes[5] = sympy.sympify(" sqrt(5/11+2*sqrt(5/3)/11)")
        nodes[6] = sympy.Integer(1)
        w[0] = sympy.Rational(1,15)
        w[1] = sympy.sympify("(124-7*sqrt(15))/350")
        w[2] = sympy.sympify("(124+7*sqrt(15))/350")
        w[3] = sympy.sympify("256/525")
        w[4] = sympy.sympify("(124+7*sqrt(15))/350")
        w[5] = sympy.sympify("(124-7*sqrt(15))/350")
        w[6] = sympy.Rational(1,21)
    else:
        nodes,w = lglnodes(order-1,tol)
    return nodes,w


def gauss_legendre(n):
    nodes=sympy.zeros(n,1)
    wi=sympy.zeros(1,n)
    x = sympy.Symbol("x")
    Pnx = sympy.legendre(n,x)
    Pp = sympy.diff(Pnx,x)
    xi = sympy.solve( Pnx, x )
    xi.sort()
    for k in range(n):
        nodes[k] = xi[k]
    for j, xj in enumerate(xi):
        wi[j] = sympy.simplify(2/(1 - xj**2)/(Pp.subs(x,xj))**2)
    return nodes, wi

def get_nodes(order,nodes_type):
    if nodes_type=="equispaced":
        nodes,w = equispaced(order)
    elif nodes_type == "gaussLegendre":
        nodes,w = gauss_legendre(order)
    elif nodes_type == "gaussLobatto":
        nodes, w = my_gausslobatto(order)
    nodes=nodes*sympy.Rational(1,2)+sympy.Rational(1,2)*sympy.ones(order,1)
    w = w*sympy.Rational(1,2)
    return nodes, w
        
def getADER_matrix(order, nodes_type):
    nodes_poly, w_poly = get_nodes(order,nodes_type)
    if nodes_type=="equispaced":
        quad_order=order
        nodes_quad, w = get_nodes(quad_order,"gaussLegendre")
    else:
        quad_order=order
        nodes_quad, w = get_nodes(quad_order,nodes_type)

    x=sympy.Symbol("x")               
    # generate mass matrix
    M = sympy.zeros(order,order)
    RHSmat = sympy.zeros(order,order)
    bADER = sympy.zeros(1,order)
    for i in range(order):
        phii=lagrange_basis(nodes_poly,x,i)
        dphii=sympy.diff(phii,x)#lagrange_deriv(nodes_poly,x,i)
        for j in range(order):
            phij = lagrange_basis(nodes_poly,x,j)
            integr1= sum([dphii.subs(x,nodes_quad[q])*phij.subs(x,nodes_quad[q])*w[q] for q in range(quad_order)])
#        dphii*phij,(x,0,1))
            integr1=integr1.simplify()
            integr2=sum([phii.subs(x,nodes_quad[q])*phij.subs(x,nodes_quad[q])*w[q] for q in range(quad_order)])
        #sympy.integrate(phii*phij,(x,0,1))
            integr2=integr2.simplify()
            M[i,j] = phii.subs(x,1)*phij.subs(x,1)-integr1
            M[i,j] = M[i,j].simplify()
            RHSmat[i,j] = integr2

        bADER[i] = sympy.integrate(phii,(x,0,1)).simplify()
    return nodes_poly, w_poly, M, RHSmat, bADER

def ader(func, tspan, y_0, M_sub, K_corr, distribution):
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    u_p=np.zeros((dim, M_sub))
    u_a=np.zeros((dim, M_sub))
    u_tn=np.zeros((dim, M_sub))
    rhs= np.zeros((dim,M_sub))
    
    x_poly, w_poly, ADER, RHS_mat = getADER_matrix(M_sub, distribution)
    invader = np.linalg.inv(ADER)
    evolMatrix=np.matmul(invader,RHS_mat)
    
    U[:,0]=y_0
    
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        for m in range(M_sub):
            u_a[:,m]=U[:,it-1]
            u_p[:,m]=U[:,it-1]
            u_tn[:,m]=U[:,it-1]
        for k in range(1,K_corr+1):
            u_p=np.copy(u_a)
            for r in range(M_sub):
                rhs[:,r]=func(u_p[:,r])
            for d in range(dim):
                u_a[d,:] = u_tn[d,:] + delta_t*np.matmul(evolMatrix,rhs[d,:])
        for d in range(dim):
            U[d,it]=sum(u_a[d,:]*[lagrange_basis(x_poly,1.0,i) for i in range(M_sub)])
    return tspan, U



def compute_RK_ADER(order,nodes_type,M=0):
    if M==0:
        M = order
    if M==1 and nodes_type in ["gaussLobatto","equispaced"]:
        print(f"M=1 is not enough to run with {nodes_type}, I run with M=2")
        M = 2
    K_corr = order
    NRK = K_corr*(M)+1
    ARK = sympy.zeros(NRK,NRK)
    bRK = sympy.zeros(1,NRK)
    cRK = sympy.zeros(NRK,1)
    
    x_poly, w_poly, ADER, RHS_mat,bADER = getADER_matrix(M, nodes_type)
    Qmat= ADER.inv()@RHS_mat
    Pvec=sympy.zeros(M,1)
    for z in range(M):
        for j in range(M):
            Qmat[z,j]=Qmat[z,j].simplify()
            Pvec[z]=Pvec[z]+Qmat[z,j]
        Pvec[z]=Pvec[z].simplify()
        cRK[z+1]=Pvec[z]
        ARK[z+1,0]=Pvec[z]

    for k in range(1,K_corr):
        rowShift = 1+(k)*(M)
        columnShift= 1+(k-1)*(M)
        for z in range(M): 
            cRK[z+rowShift]=Pvec[z]
            for j in range(M):
                ARK[z+rowShift,j+columnShift]=Qmat[z,j]
    for k in range(M):
        bRK[NRK-M+k] = bADER[k]

    return ARK,bRK,cRK


