import numpy as np
from simulator import Simulator
from slicing import cut, crop_fv

def detect_troubles(self: Simulator):
    if self.godunov:
        return

    # Reset to check troubled control volumes
    ngh=self.Nghc
    crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,ngh)
    self.dm.troubles[...] = 0
    W_old = self.compute_primitives_cv(self.dm.U_cv)
    # W_old -> s.dm.M_fv
    self.fill_active_region(W_old)
    W_new = self.compute_primitives_cv(self.dm.U_new)    
    ##############################################
    # NAD Check for numerically adimissible values
    ##############################################
    # First check if DMP criteria is met, if it is we can avoid computing alpha
    self.fv_Boundaries(self.dm.M_fv)
    W_max = self.dm.M_fv.copy()
    W_min = self.dm.M_fv.copy()
    for idim in self.idims:
        W_max = np.maximum(compute_W_max(self.dm.M_fv,idim),W_max)
        W_min = np.minimum(compute_W_min(self.dm.M_fv,idim),W_min)
            
    W_max = self.crop(W_max,ngh=ngh)
    W_min = self.crop(W_min,ngh=ngh)
    
    if self.p > 0:
        if self.NAD == "delta":
            epsilon = self.tolerance*(W_max-W_min)
            W_min -= epsilon 
            W_max += epsilon
        else:
            W_min -= np.abs(W_min) * self.tolerance
            W_max += np.abs(W_max) * self.tolerance

    possible_trouble = np.where(W_new >= W_min, 0, 1)
    possible_trouble = np.where(W_new <= W_max, possible_trouble, 1)
       
    # Now check for smooth extrema and relax the criteria for such cases
    if self.p > 1 and self.SED:
        self.fill_active_region(W_new)
        self.fv_Boundaries(self.dm.M_fv)
        alpha = W_new*0 + 1
        for dim in self.dims:
            idim = self.dims[dim]
            alpha_new = compute_smooth_extrema(self, self.dm.M_fv, dim)[crop(None,None,idim)]
            alpha = np.where(alpha_new < alpha, alpha_new, alpha)

        possible_trouble *= np.where(alpha<1, 1, 0)

    self.dm.troubles[...] = np.amax(possible_trouble[self.limiting_variables],axis=0)
    
    ###########################
    # PAD Check for physically admissible values
    ###########################
    if self.PAD:
        if self.WB:
            W_new += self.compute_primitives(self.dm.U_eq_cv)
        # For the density
        self.dm.troubles = np.where(
            W_new[self._d_, ...] >= self.min_rho, self.dm.troubles, 1
        )
        self.dm.troubles = np.where(
            W_new[self._d_, ...] <= self.max_rho, self.dm.troubles, 1
        )
        # For the pressure
        self.dm.troubles = np.where(
            W_new[self._p_, ...] >= self.min_P, self.dm.troubles, 1)

    #self.n_troubles += self.dm.troubles.sum()
    self.dm.M_fv[...] = 0
    self.fill_active_region(self.dm.troubles)
    self.fv_Boundaries(self.dm.M_fv,all=False)
    trouble = self.dm.M_fv[0]
    self.dm.theta[0][...] = trouble
    theta = self.dm.theta[0]

    if self.blending:
        apply_blending(self,trouble,theta)

    for dim in self.dims:
        idim = self.dims[dim]
        affected_faces = self.dm.__getattribute__(f"affected_faces_{dim}")
        affected_faces[...] = 0
        affected_faces[...] = np.maximum(theta[crop(ngh-1,-ngh,idim)],theta[crop(ngh,-(ngh-1),idim)])

def compute_W_ex(W, idim, f):
    W_f = W.copy()
    # W_f(i) = f(W(i-1),W(i),W(i+1))
    # First comparing W(i) and W(i+1)
    W_f[cut(None,-1,idim)] = f(  W[cut(None,-1,idim)],W[cut(1, None,idim)])
    # Now comparing W_f(i) and W_(i-1)
    W_f[cut( 1,None,idim)] = f(W_f[cut( 1,None,idim)],W_f[cut(None,-1,idim)])
    return W_f

def compute_W_max(W, idim):
    return compute_W_ex(W, idim, np.maximum)

def compute_W_min(W, idim):
    return compute_W_ex(W, idim, np.minimum)

def first_order_derivative(U, h, idim):
    dU = (U[cut(2,None,idim)] - U[cut(None,-2,idim)])/(h[cut(2,None,idim)] - h[cut(None,-2,idim)])
    return dU

def compute_min(A, Amin, idim):
    Amin[cut(None,-1,idim)] = np.minimum(A[cut(None,-1,idim)],   A[cut(1,None,idim)])
    Amin[cut( 1,None,idim)] = np.minimum(A[cut(None,-1,idim)],Amin[cut(1,None,idim)])

def compute_smooth_extrema(self, U, dim):
    eps = 0
    idim = self.dims[dim]
    centers = self.centers[dim][self.shape(idim)]
    # First derivative dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
    dU  = first_order_derivative( U, centers, idim)
    # Second derivative d2Udx2(i) = [dU(i+1)-dU(i-1)]/[x_cv(i+1)-x_cv(i-1)]
    d2U = first_order_derivative(dU, centers[cut(1,-1,idim)], idim)
    dv = 0.5 * self.h_fp[dim][cut(2,-2,idim)] * d2U
    # vL = dU(i-1)-dU(i)
    vL = dU[cut(None,-2,idim)] - dU[cut(1,-1,idim)]
    # alphaL = min(1,max(vL,0)/(-dv)),1,min(1,min(vL,0)/(-dv)) for dv<0,dv=0,dv>0
    alphaL = (
        -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0)) / dv
    )
    alphaL = np.where(np.abs(dv) <= eps, 1, alphaL)
    alphaL = np.where(alphaL < 1, alphaL, 1)
    # vR = dU(i+1)-dU(i)
    vR = dU[cut( 2,None,idim)] - dU[cut(1,-1,idim)]
    # alphaR = min(1,max(vR,0)/(dv)),1,min(1,min(vR,0)/(dv)) for dv>0,dv=0,dv<0
    alphaR = np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0)) / dv
    alphaR = np.where(np.abs(dv) <= eps, 1, alphaR)
    alphaR = np.where(alphaR < 1, alphaR, 1)
    alphaR = np.where(alphaR < alphaL, alphaR, alphaL)
    compute_min(alphaR, alphaL, idim)
    return alphaL

def apply_blending(self,trouble,theta):
    a = slice(None,-1)
    b = slice( 1,None)
    cuts = [(a,a),(a,b),(b,a),(b,b)]
    #First neighbors
    for idim in self.idims:
        theta[cut(None,-1,idim)] = np.maximum(.75*trouble[cut( 1,None,idim)],theta[cut(None,-1,idim)])
        theta[cut( 1,None,idim)] = np.maximum(.75*trouble[cut(None,-1,idim)],theta[cut( 1,None,idim)])
          
    if self.ndim==2:
        #Second neighbors
        for i in range(len(cuts)):
            theta[cuts[i]] = np.maximum(.5*trouble[cuts[::-1][i]],theta[cuts[i]])
                
    elif self.ndim==3:
        #Second neighbors
        for i in range(len(cuts)):
            for idim in self.idims:
                shape1 = tuple(np.roll(np.array((slice(None),)+cuts[ i]),-idim))
                shape2 = tuple(np.roll(np.array((slice(None),)+cuts[::-1][-i]),-idim))
                theta[shape1] = np.maximum(.5*trouble[shape2],theta[shape1])
        #Third neighbors
        cuts1 = [(x,y,z) for x in (a,b) for y in (a,b) for z in (a,b)]
        cuts2 = [(x,y,z) for x in (b,a) for y in (b,a) for z in (b,a)]
        for i in range(len(cuts)):
            theta[cuts1[i]] = np.maximum(.375*trouble[cuts2[i]],theta[cuts1[i]])
    
    #Last layer
    for idim in self.idims:
        theta[cut(None,-1,idim)] = np.maximum(.25*(theta[cut( 1,None,idim)]>0),theta[cut(None,-1,idim)])
        theta[cut( 1,None,idim)] = np.maximum(.25*(theta[cut(None,-1,idim)]>0),theta[cut( 1,None,idim)])
     
        
