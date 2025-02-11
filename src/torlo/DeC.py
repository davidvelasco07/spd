import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from quadr import lglnodes,equispaced

class DeC:
    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.compute_theta_DeC()
        self.name = f"DeC_{self.nodes_type}"
    
    def compute_theta_DeC(self):
        nodes, w = get_nodes(self.n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(self.n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta = np.zeros((self.n_subNodes,self.n_subNodes))
        self.beta = np.zeros(self.n_subNodes)
        for m in range(self.n_subNodes):
            self.beta[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(self.n_subNodes):
                self.theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return self.theta, self.beta
    
    
    def compute_RK_from_DeC(self):
        bar_beta=self.beta[1:]  # M_sub
        bar_theta=self.theta[:,1:].transpose() # M_sub x (M_sub +1)
        theta0= bar_theta[:,0]  # M_sub x 1
        bar_theta= bar_theta[:,1:] #M_sub x M_sub
        self.ARK=np.zeros((self.M_sub*(self.n_iter-1)+1,self.M_sub*(self.n_iter-1)+1))  # (M_sub x K_corr +1)^2
        self.bRK=np.zeros(self.M_sub*(self.n_iter-1)+1)
        self.cRK=np.zeros(self.M_sub*(self.n_iter-1)+1)

        self.cRK[1:self.M_sub+1]=bar_beta
        self.ARK[1:self.M_sub+1,0]=bar_beta
        for k in range(1,self.n_iter-1):
            r0=1+self.M_sub*k
            r1=1+self.M_sub*(k+1)
            c0=1+self.M_sub*(k-1)
            c1=1+self.M_sub*(k)
            self.cRK[r0:r1]=bar_beta
            self.ARK[r0:r1,0]=theta0
            self.ARK[r0:r1,c0:c1]=bar_theta
        self.bRK[0]=theta0[-1]
        self.bRK[-self.M_sub:]=bar_theta[self.M_sub-1,:]
        return self.ARK,self.bRK,self.cRK
    
    def compute_RK_from_DeCImplicit(self):
        self.compute_RK_from_DeC()
        self.ARKIM=np.zeros((self.M_sub*(self.n_iter-1)+2,self.M_sub*(self.n_iter-1)+2))  # (M_sub x K_corr +1)^2
        self.bRKIM=np.zeros(self.M_sub*(self.n_iter-1)+2)
        self.cRKIM=np.zeros(self.M_sub*(self.n_iter-1)+2)
        bar_beta=self.beta[1:]  # M_sub
        bar_theta=self.theta[:,1:].transpose() # M_sub x (M_sub +1)
        bar_theta= bar_theta[:,1:] #M_sub x M_sub
        theta0= bar_theta[:,0]  # M_sub x 1
        self.ARKIM[:-1,:-1]=self.ARK  # (M_sub x K_corr +1)^2
        self.bRKIM[:-1]=self.bRK
        self.cRKIM[:-1]=self.cRK

        self.cRK[1:self.M_sub+1]=bar_beta
        self.ARK[1:self.M_sub+1,0]=bar_beta
        k=0
        r0=1+self.M_sub*k
        r1=1+self.M_sub*(k+1)
        c0=0
        c1=1+self.M_sub*(k)
        c2=1+self.M_sub*(k+1)
        self.ARKIM[r0:r1,c1:c2] = np.diag(bar_beta)
        self.ARKIM[r0:r1,c0:c1] = self.ARKIM[r0:r1,c0:c1] - bar_beta.reshape((r1-r0,c1-c0))
        for k in range(1,self.n_iter-1):
            r0=1+self.M_sub*k
            r1=1+self.M_sub*(k+1)
            c0=1+self.M_sub*(k-1)
            c1=1+self.M_sub*(k)
            c2=1+self.M_sub*(k+1)
            self.ARKIM[r0:r1,c1:c2] = np.diag(bar_beta) 
            self.ARKIM[r0:r1,c0:c1] = self.ARKIM[r0:r1,c0:c1] - np.diag(bar_beta) 
        self.ARKIM[-1,:-1] = self.bRK
        self.ARKIM[-1,-2] = self.ARKIM[-1,-2] - bar_beta[-1]
        self.ARKIM[-1,-1] = bar_beta[-1]
        self.cRKIM[-1] = bar_beta[-1]
        self.bRKIM = self.ARKIM[-1,:]
        return self.ARKIM,self.bRKIM,self.cRKIM
    
    def compute_RK_from_DeCImplicit2(self):
        self.compute_RK_from_DeC()
        # very lazy way
        NRK = (self.n_subNodes)*(self.n_iter+1)
        self.ARKIM=np.zeros((NRK,NRK))  
        self.bRKIM=np.zeros(NRK)
        self.cRKIM=np.zeros(NRK)
        B = np.diag(self.beta)
        theta = self.theta.transpose()

        TDB = theta - B
        k=0
        self.ARKIM[:self.n_subNodes,:self.n_subNodes] = 0. 
        for k in range(1,self.n_iter+1):
            idx0=self.n_subNodes*(k-1)
            idx1=self.n_subNodes*(k)
            idx2=self.n_subNodes*(k+1)
            self.ARKIM[idx1:idx2,idx0:idx1] = TDB 
            self.ARKIM[idx1:idx2,idx1:idx2] = B 
        self.bRKIM = self.ARKIM[-1,:]
        self.cRKIM = self.ARKIM.sum(axis=1)
        
        self.NRK = NRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARKIM[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARKIM[:,0] = self.ARKIM[:,0] + self.ARKIM[:,i]
                self.bRKIM[0] = self.bRKIM[0] + self.bRKIM[i]
                # delete i-th row and column
                self.ARKIM = np.delete(self.ARKIM, i,0)
                self.ARKIM = np.delete(self.ARKIM, i,1)
                self.cRKIM = np.delete(self.cRKIM,i,0)
                self.bRKIM = np.delete(self.bRKIM,i,0)
            else:
                i=i+1


        return self.ARKIM,self.bRKIM,self.cRKIM

    def dec(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        rhs= np.zeros((dim,self.M_sub+1))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            t_sub = tspan[it-1]+ delta_t*self.beta
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            rhs[:,0] = func(U[:,it-1],t_sub[0])
            for r in range(1,self.M_sub+1):
                rhs[:,r] = rhs[:,0]
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                if k>1:
                    for r in range(1,self.M_sub+1):
                        rhs[:,r]=func(u_p[:,r],t_sub[r])
                if k < self.n_iter:
                    for m in range(1,self.M_sub+1):
                        u_a[:,m]= U[:,it-1]+delta_t*sum([self.theta[r,m]*rhs[:,r] for r in range(self.M_sub+1)])
                else:
                    u_a[:,self.M_sub]= U[:,it-1]+delta_t*sum([self.theta[r,self.M_sub]*rhs[:,r] for r in range(self.M_sub+1)])
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U

    def decImplicit(self, func,jac_stiff, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        u_help= np.zeros(dim)
        rhs= np.zeros((dim,self.M_sub+1))
        invJac=np.zeros((self.M_sub+1,dim,dim))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])            
            t_sub = tspan[it-1]+ delta_t*self.beta
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            SS=jac_stiff(u_p[:,0])
            for m in range(1,self.M_sub+1):
                invJac[m,:,:]=np.linalg.inv(np.eye(dim) - delta_t*self.beta[m]*SS)
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                for r in range(self.M_sub+1):
                    rhs[:,r]=func(u_p[:,r],t_sub[r])
                for m in range(1,self.M_sub+1):
                    u_a[:,m]= u_p[:,m]+delta_t*np.matmul(invJac[m,:,:],\
                    (-(u_p[:,m]-u_p[:,0])/delta_t\
                     +sum([self.theta[r,m]*rhs[:,r] for r in range(self.M_sub+1)])))
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U



    def decMPatankar(self, prod_dest, rhs, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        prod_p = np.zeros((dim,dim,self.M_sub+1))
        dest_p = np.zeros((dim,dim,self.M_sub+1))
        rhs_p= np.zeros((dim,self.M_sub+1))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                for r in range(self.M_sub+1):
                    prod_p[:,:,r], dest_p[:,:,r]=prod_dest(u_p[:,r])
                    rhs_p[:,r]=rhs(u_p[:,r])
                for m in range(1,self.M_sub+1):
                    u_a[:,m]= patankar_type_dec(prod_p,dest_p,rhs_p,delta_t,m,self.M_sub,self.theta,u_p,dim)
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U

 
class DeC_small_sub:
    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.compute_theta_DeC()
        self.name = f"DeC_small_{self.nodes_type}"
    
    def compute_theta_DeC(self):
        nodes, w = get_nodes(self.n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(self.n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta = np.zeros((self.n_subNodes,self.n_subNodes))
        self.beta = np.zeros(self.n_subNodes)
        for m in range(self.n_subNodes):
            self.beta[m] = nodes[m]-nodes[max(m-1,0)]
            nodes_m = int_nodes*(nodes[m]-nodes[max(m-1,0)])+nodes[max(m-1,0)]
            w_m = int_w*(nodes[m]-nodes[max(m-1,0)])
            for r in range(self.n_subNodes):
                self.theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        # Original DeC coefficients
        nodes, w = get_nodes(self.n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(self.n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta_big = np.zeros((self.n_subNodes,self.n_subNodes))
        self.beta_big = np.zeros(self.n_subNodes)
        for m in range(self.n_subNodes):
            self.beta_big[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(self.n_subNodes):
                self.theta_big[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return self.theta, self.beta
    
    
    def compute_RK_from_DeC(self):        
        bar_theta=self.theta_big[:,1:].transpose() # M_sub x (M_sub +1)
        theta0= bar_theta[:,0]  # M_sub x 1
        bar_theta= bar_theta[:,1:] #M_sub x M_sub
        bar_beta= np.zeros(np.shape(bar_theta))
        beta0 = np.zeros(np.shape(theta0))
        beta0[:]=self.beta[1]
        for m in range(1,self.M_sub):
            bar_beta[m:,m-1] = self.beta[m+1]

        self.ARK=np.zeros((self.M_sub*(self.n_iter)+1,self.M_sub*(self.n_iter)))  # (M_sub x K_corr +1)^2
        self.bRK=np.zeros(self.M_sub*(self.n_iter))
        self.cRK=np.zeros(self.M_sub*(self.n_iter)+1)

        self.cRK[1:self.M_sub+1]=self.beta[1:]
        self.ARK[1:self.M_sub+1,0]=beta0
        self.ARK[1:self.M_sub+1,1:self.M_sub+1]=bar_beta
        for k in range(1,self.n_iter):
            r0=1+self.M_sub*k
            r1=1+self.M_sub*(k+1)
            c0=1+self.M_sub*(k-1)
            c1=1+self.M_sub*(k)
            self.cRK[r0:r1]=self.beta[1:]
            self.ARK[r0:r1,0]=theta0
            self.ARK[r0:r1,c0:c1]=bar_theta-bar_beta
            self.ARK[r0:r1,r0:r1-1]=bar_beta[:,:-1]
        self.bRK=self.ARK[-1,:]
        self.ARK=self.ARK[:-1,:]
        self.cRK=self.cRK[:-1]
        return self.ARK,self.bRK,self.cRK
    
    def compute_RK_from_DeCImplicit(self):
        self.compute_RK_from_DeC()
        # very lazy way
        NRK = (self.n_subNodes)*(self.n_iter+1)
        self.ARKIM=np.zeros((NRK,NRK))  
        self.bRKIM=np.zeros(NRK)
        self.cRKIM=np.zeros(NRK)
        delta_beta = np.zeros((self.n_subNodes,self.n_subNodes))
        for m in range(self.n_subNodes):
            for r in range(1,m+1):
                delta_beta[m,r] = self.beta_big[r]-self.beta_big[r-1]
        theta_big = self.theta_big.transpose()

        TDB = theta_big - delta_beta
        k=0
        self.ARKIM[:self.n_subNodes,:self.n_subNodes] = 0. 
        for k in range(1,self.n_iter+1):
            idx0=self.n_subNodes*(k-1)
            idx1=self.n_subNodes*(k)
            idx2=self.n_subNodes*(k+1)
            self.ARKIM[idx1:idx2,idx0:idx1] = TDB 
            self.ARKIM[idx1:idx2,idx1:idx2] = delta_beta 
        self.bRKIM = self.ARKIM[-1,:]
        self.cRKIM = self.ARKIM.sum(axis=1)
        
        self.NRK = NRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARKIM[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARKIM[:,0] = self.ARKIM[:,0] + self.ARKIM[:,i]
                self.bRKIM[0] = self.bRKIM[0] + self.bRKIM[i]
                # delete i-th row and column
                self.ARKIM = np.delete(self.ARKIM, i,0)
                self.ARKIM = np.delete(self.ARKIM, i,1)
                self.cRKIM = np.delete(self.cRKIM,i,0)
                self.bRKIM = np.delete(self.bRKIM,i,0)
            else:
                i=i+1


        return self.ARKIM,self.bRKIM,self.cRKIM
    
    def dec(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        rhs_p= np.zeros((dim,self.M_sub+1))
        rhs_a= np.zeros((dim,self.M_sub+1))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            rhs_0 = func(U[:,it-1],tspan[it-1])
            t_sub = tspan[it-1] + delta_t*self.beta_big
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
                rhs_p[:,m] = rhs_0
                rhs_a[:,m] = rhs_0
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                rhs_p=np.copy(rhs_a)
                for m in range(1,self.M_sub+1):
                    u_a[:,m]= u_a[:,m-1]+ delta_t*self.beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])\
                        +delta_t*sum([self.theta[r,m]*rhs_p[:,r] for r in range(self.M_sub+1)])
                    rhs_a[:,m] =func(u_a[:,m],t_sub[m])
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U
    
    
    def dec_v2(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        rhs_p= np.zeros((dim,self.M_sub+1))
        rhs_a= np.zeros((dim,self.M_sub+1))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            t_sub = tspan[it-1] + delta_t*self.beta
            rhs_0 = func(U[:,it-1],tspan[it-1])
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
                rhs_p[:,m] = rhs_0
                rhs_a[:,m] = rhs_0
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                rhs_p=np.copy(rhs_a)
                incr = np.zeros(np.shape(rhs_a[:,0]))
                for m in range(1,self.M_sub+1):
                    incr=incr+self.beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])
                    u_a[:,m]= U[:,it-1]+ delta_t*incr\
                        +delta_t*sum([self.theta_big[r,m]*rhs_p[:,r] for r in range(self.M_sub+1)])
                    rhs_a[:,m] =func(u_a[:,m],t_sub[m])
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U

    
class DeC_staggered:
    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.thetas=[]
        self.betas=[]
        self.interpolation_matrices = []
        theta, beta = self.compute_theta_DeC(2)
        self.thetas.append(theta)
        self.betas.append(beta)
        self.interpolation_matrices.append(None)
        self.name = f"DeC_stag_{self.nodes_type}"
        
        for p in range(1,self.M_sub+1):
            theta, beta = self.compute_theta_DeC(max(p+1,2))
            self.thetas.append(theta)
            self.betas.append(beta)
            self.interpolation_matrices.append(\
                    self.compute_interpolation_matrix(p+1))
        

    def compute_theta_DeC(self, n_subNodes=None):
        if n_subNodes is None:
            n_subNodes = self.n_subNodes
        nodes, w = get_nodes(n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        theta = np.zeros((n_subNodes,n_subNodes))
        beta = np.zeros(n_subNodes)
        for m in range(n_subNodes):
            beta[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(n_subNodes):
                theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return theta, beta
    
    def compute_interpolation_matrix(self,N):
        nodes_interpolation, _ = get_nodes(max(N,  2),self.nodes_type)
        nodes_basis, _         = get_nodes(max(N-1,2),self.nodes_type)
        A = np.zeros((max(N,2),max(N-1,2)))
        for m in range(len(nodes_basis)):
            A[:,m] = lagrange_basis(nodes_basis,nodes_interpolation,m)
        return A
        
    
   
    def compute_RK_from_DeC(self):
        NRK = (self.M_sub*(self.n_iter-1)+1) - int((self.M_sub-2)*(self.M_sub-1)/2)
        self.ARK=np.zeros((NRK,NRK))  # (M_sub x K_corr +1)^2
        self.bRK=np.zeros(NRK)
        self.cRK=np.zeros(NRK)

        idx =1

        p = 1
        theta = self.thetas[p].transpose()
        interp_mat = self.interpolation_matrices[p+1]
        II=np.array([1.,1.])
        zeta = interp_mat@theta@II
        zeta_bar = zeta[1:]
        idx_new = idx + len(zeta_bar)
        self.ARK[idx:idx_new,0] = zeta_bar
        for p in range(2,self.n_iter):
            idx_old=idx
            idx=idx_new
            idx_new = idx + min(p+1,self.M_sub)
            theta = self.thetas[min(p,self.M_sub)].transpose()
            if p<self.M_sub:                            
                interp_mat = self.interpolation_matrices[min(p+1,self.M_sub)]
            else:
                interp_mat = np.eye(theta.shape[0])
            zeta = interp_mat@theta
            zeta_bar = zeta[1:,1:]
            zeta_zero = zeta[1:,0]
            self.ARK[idx:idx_new,idx_old:idx] = zeta_bar
            self.ARK[idx:idx_new,0] = zeta_zero

        theta_M=theta[-1,1:] # M_sub x (M_sub +1)
        theta0= theta[-1,0]  # M_sub x 1
        self.bRK[0]=theta0
        self.bRK[-self.M_sub:]=theta_M
        self.cRK = np.sum(self.ARK,1)
        return self.ARK,self.bRK,self.cRK
    
    def dec(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        U[:,0]=y_0
        u_p = np.zeros((dim,self.M_sub+1))
        u_a = np.zeros((dim,self.M_sub+1))
        rhs = np.zeros((dim,self.M_sub+1))
        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            feval=0
            delta_t=(tspan[it]-tspan[it-1])
            u_p[:,:] = 0. #np.zeros((dim, 2))
            u_a[:,:] = 0. #np.zeros((dim, 2))
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            rhs[:,0]=func(u_p[:,0],tspan[it-1])
            for p in range(1,self.n_iter):
#                 print(p)
                M_old = M_olds[p]#max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                # u_p=np.zeros((dim, M_new))
                u_p[:,:]= 0. #=np.zeros((dim, M_new+1))
                if p <= self.M_sub and p>1:
                    for idim in range(dim):
                        u_p[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@u_a[idim,:M_old+1]
                        u_p[idim,M_new] = u_a[idim,M_old]
                        #u_p[idim,:M_new+1] = self.interpolation_matrices[p]@u_a[idim,:M_old+1]
                    theta = self.thetas[M_new]
                    beta = self.betas[M_new]
                else:
                    u_p = np.copy(u_a)
                    theta = self.thetas[M_new]
                    beta = self.betas[M_new]

                t_sub = tspan[it-1] + delta_t*beta
#                 print(theta)
                # u_a[:,:]= 0. #= np.zeros((dim,M_new+1))
#                 print(u_a.shape)
                # rhs[:,:]= 0. #= np.zeros((dim,M_new+1))
                if p==1:
                    for r in range(1,M_new+1):
                        rhs[:,r]=rhs[:,0]
                else:
                    for r in range(1,M_new+1):
                        feval+=1
                        rhs[:,r]=func(u_p[:,r],t_sub[r])
                for m in range(1,M_new+1):
                    u_a[:,m]= U[:,it-1]+delta_t*sum([theta[r,m]*rhs[:,r] for r in range(M_new+1)])
            # Extra iteration with no interpolation to restore the original accuracy
            u_p = np.copy(u_a)
            theta = self.thetas[-1]
            beta = self.betas[-1]
            t_sub = tspan[it-1] + delta_t*beta
#             print(theta)
#             print(u_a.shape)
            # u_a=np.zeros((dim, M_new+1))
            # rhs= np.zeros((dim,M_new+1))
            for r in range(1,M_new+1):
                rhs[:,r]=func(u_p[:,r],t_sub[r])
            u_a[:,M_new]= U[:,it-1]+delta_t*sum([theta[r,M_new]*rhs[:,r] for r in range(M_new+1)])         
            U[:,it]=u_a[:,-1]
        return tspan, U

    
    def dec_order_control(self, func, tspan, y_0, err_tol):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        p_times=np.zeros(N_time)
        U[:,0]=y_0
        u_p = np.zeros((dim,self.M_sub+1))
        u_a = np.zeros((dim,self.M_sub+1))
        rhs = np.zeros((dim,self.M_sub+1))
        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            u_p[:,:] = 0. #np.zeros((dim, 2))
            u_a[:,:] = 0. #np.zeros((dim, 2))
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            rhs[:,0]=func(u_p[:,0],tspan[it-1])
            err_local = err_tol + 1.
            p=0
            while( p < self.n_iter-1): # and err_local > err_tol 
                p=p+1
#            for p in range(1,self.n_iter):
#                 print(p)
                M_old = M_olds[p]#max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                # u_p=np.zeros((dim, M_new))
                u_p[:,:]= 0. #=np.zeros((dim, M_new+1))
                if p <= self.M_sub and p>1:
                    for idim in range(dim):
                        u_p[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@u_a[idim,:M_old+1]
                        u_p[idim,M_new] = u_a[idim,M_old]
                        #u_p[idim,:M_new+1] = self.interpolation_matrices[p]@u_a[idim,:M_old+1]
                    theta = self.thetas[M_new]
                    beta = self.betas[M_new]
                else:
                    u_p = np.copy(u_a)
                    theta = self.thetas[M_new]
                    beta = self.betas[M_new]

                t_sub = tspan[it-1] + delta_t*beta
#                 print(theta)
                # u_a[:,:]= 0. #= np.zeros((dim,M_new+1))
#                 print(u_a.shape)
                # rhs[:,:]= 0. #= np.zeros((dim,M_new+1))
                if p==1:
                    for r in range(1,M_new+1):
                        rhs[:,r]=rhs[:,0]
                else:
                    for r in range(1,M_new+1):
                        rhs[:,r]=func(u_p[:,r],t_sub[r])
                u_a[:,M_new]= U[:,it-1]+delta_t*sum([theta[r,M_new]*rhs[:,r] for r in range(M_new+1)])

                err_local = np.linalg.norm(u_a[:,M_new]-u_p[:,M_new]) /np.linalg.norm(u_a[:,M_new])

                if err_local <= err_tol:
                    break
                for m in range(1,M_new):
                    u_a[:,m]= U[:,it-1]+delta_t*sum([theta[r,m]*rhs[:,r] for r in range(M_new+1)])
            p_times[it] = p
            U[:,it]=u_a[:,M_new]
        return tspan, U, p_times

class DeC_small_sub_staggered:
    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.thetas=[]
        self.betas=[]
        self.thetas_big=[]
        self.betas_big=[]
        self.interpolation_matrices = []
        theta, beta, theta_big, beta_big = self.compute_theta_DeC(2)
        self.thetas.append(theta)
        self.betas.append(beta)
        self.thetas_big.append(theta_big)
        self.betas_big.append(beta_big)
        self.interpolation_matrices.append(None)
        self.name = f"DeC_small_stag_{self.nodes_type}"
        
        for p in range(1,self.M_sub+1):
            theta, beta, theta_big, beta_big  = self.compute_theta_DeC(max(p+1,2))
            self.thetas.append(theta)
            self.betas.append(beta)
            self.thetas_big.append(theta_big)
            self.betas_big.append(beta_big)
            self.interpolation_matrices.append(\
                    self.compute_interpolation_matrix(p+1))
        

    def compute_interpolation_matrix(self,N):
        nodes_interpolation, _ = get_nodes(max(N,  2),self.nodes_type)
        nodes_basis, _         = get_nodes(max(N-1,2),self.nodes_type)
        A = np.zeros((max(N,2),max(N-1,2)))
        for m in range(len(nodes_basis)):
            A[:,m] = lagrange_basis(nodes_basis,nodes_interpolation,m)
        return A
        
    def compute_theta_DeC(self,n_subNodes=None):        
        if n_subNodes is None:
            n_subNodes = self.n_subNodes
        nodes, w = get_nodes(n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta = np.zeros((n_subNodes,n_subNodes))
        self.beta = np.zeros(n_subNodes)
        for m in range(n_subNodes):
            self.beta[m] = nodes[m]-nodes[max(m-1,0)]
            nodes_m = int_nodes*(nodes[m]-nodes[max(m-1,0)])+nodes[max(m-1,0)]
            w_m = int_w*(nodes[m]-nodes[max(m-1,0)])
            for r in range(n_subNodes):
                self.theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        # Original DeC coefficients
        nodes, w = get_nodes(n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta_big = np.zeros((n_subNodes,n_subNodes))
        self.beta_big = np.zeros(n_subNodes)
        for m in range(n_subNodes):
            self.beta_big[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(n_subNodes):
                self.theta_big[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return self.theta, self.beta, self.theta_big, self.beta_big
    
    
    def compute_RK_from_DeC(self):
        NRK = 2*self.M_sub*(self.n_iter-1) -(self.M_sub-1)**2 
        
        self.ARK=np.zeros((NRK+1,NRK+1)) # one extra row that becomes b
        self.bRK=np.zeros(NRK)
        self.cRK=np.zeros(NRK+1)


        idx =1

        p = 1
        self.ARK[1:4,0] = [1.,1./2.,1.]
        beta=self.betas[p]
        gamma_mat_new = np.zeros((min(p+1,self.M_sub),min(p+1,self.M_sub)))
        for m in range(1,min(p+1,self.M_sub)):
            gamma_mat_new[m:,m-1] = self.beta[m+1]

        idx_new = idx + 3
        idx_p_s = 2
        idx_pp1 = 4

        self.cRK[idx:idx_new]=self.ARK[1:4,0]
        for p in range(2,self.n_iter-1):
            p_loc = min(p,self.M_sub)
            p1_loc = min(p+1,self.M_sub)

            idx_pm1_s = idx_p_s
            idx_p     = idx_pp1
            idx_p_s   = idx_p   + p_loc
            idx_pp1   = idx_p_s + p1_loc


            theta     = self.thetas_big[p_loc].transpose()
            beta_big  = self.betas_big[p_loc]
            beta      = self.betas[p_loc]
            if p<self.M_sub:                            
                interp_mat = self.interpolation_matrices[p1_loc]
            else:
                interp_mat = np.eye(theta.shape[0])
            
            emme_theta = interp_mat@theta
            emme_theta_bar = emme_theta[1:,1:]
            emme_theta_zero = emme_theta[1:,0]

            theta_zero = theta[1:,0]
            theta_bar  = theta[1:,1:]

            gamma_mat = np.zeros((min(p+1,self.M_sub+1),min(p+1,self.M_sub+1)))
            for m in range(1,min(p+1,self.M_sub+1)):
                gamma_mat[m:,m-1] = beta[m]
            
            gamma_bar = gamma_mat[1:,1:]
            emme_gamma = interp_mat@gamma_mat
            emme_gamma_bar = emme_gamma[1:,1:]
            gamma_mat_bar = gamma_mat[1:,1:]

            # y^(p)
            self.ARK[idx_p:idx_p_s,0] = theta_zero
            self.ARK[idx_p:idx_p_s,idx_pm1_s:idx_p] = theta_bar-gamma_bar
            self.ARK[idx_p:idx_p_s,idx_p:idx_p_s] = gamma_bar
            self.cRK[idx_p:idx_p_s]=beta_big[1:]

            # y^*(p)
            emme_beta = interp_mat@beta_big
            self.ARK[idx_p_s:idx_pp1,0] = emme_theta_zero
            self.ARK[idx_p_s:idx_pp1,idx_pm1_s:idx_p] = emme_theta_bar-emme_gamma_bar
            self.ARK[idx_p_s:idx_pp1,idx_p:idx_p_s]   = emme_gamma_bar
            self.cRK[idx_p_s:idx_pp1] =  emme_beta[1:]

        p=p+1
        idx_pm1_s = idx_p_s
        idx_p     = idx_pp1
        idx_p_s   = idx_p   + self.M_sub
        idx_pp1   = idx_p_s + self.M_sub

        theta     = self.thetas_big[-1].transpose()
        beta_big  = self.betas_big[-1]
        beta      = self.betas[-1]

        theta_zero = theta[1:,0]
        theta_bar  = theta[1:,1:]

        gamma_mat = np.zeros((min(p+1,self.M_sub+1),min(p+1,self.M_sub+1)))
        for m in range(1,min(p+1,self.M_sub+1)):
            gamma_mat[m:,m-1] = beta[m]
        
        gamma_bar = gamma_mat[1:,1:]

        # y^(p)
        self.ARK[idx_p:idx_p_s,0] = theta_zero
        self.ARK[idx_p:idx_p_s,idx_pm1_s:idx_p] = theta_bar-gamma_bar
        self.ARK[idx_p:idx_p_s,idx_p:idx_p_s] = gamma_bar
        self.cRK[idx_p:idx_p_s]=beta_big[1:]

        # y^(p+1)
        self.ARK[idx_p_s:idx_pp1,0] = theta_zero
        self.ARK[idx_p_s:idx_pp1,idx_p:idx_p_s]   = theta_bar-gamma_bar
        self.ARK[idx_p_s:idx_pp1,idx_p_s:idx_pp1] = gamma_bar
        self.cRK[idx_p_s:idx_pp1] =  beta_big[1:]


            
        # Last iteration
        self.bRK=self.ARK[-1,:-1]
        self.ARK=self.ARK[:-1,:-1]
        self.cRK=self.cRK[:-1]
        return self.ARK,self.bRK,self.cRK

    
    def dec(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        U[:,0]=y_0
        u_p = np.zeros((dim,self.M_sub+1))
        u_a = np.zeros((dim,self.M_sub+1))
        rhs_a = np.zeros((dim,self.M_sub+1))
        rhs_p = np.zeros((dim,self.M_sub+1))
        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            rhs_0 = func(U[:,it-1],tspan[it-1])
            u_p[:,:] = 0. #np.zeros((dim, 2))
            u_a[:,:] = 0. #np.zeros((dim, 2))
            rhs_p[:,:]=0.
            rhs_a[:,:]=0.
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
                #rhs_p[:,m] = rhs_0
                rhs_a[:,m] = rhs_0
            for p in range(1,self.n_iter):
                # New subtimesteps
                M_old = M_olds[p]#max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                
                #u_p[:,:]=0.
                rhs_p[:,:]=0.
                if p <= self.M_sub:
                    for idim in range(dim):
                        u_p[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@u_a[idim,:M_old+1]
                        u_p[idim,M_new] = u_a[idim,M_old]
#                        rhs_p[idim,:] = self.interpolation_matrices[p]@rhs_a[idim,:]
                    rhs_p[:,0] = rhs_0

                    theta     = self.thetas[p]
                    beta      = self.betas[p]
                    beta_big  = self.betas_big[p]
                    t_sub = tspan[it-1] + delta_t*beta_big
                    if p==1:
                        for m in range(1,M_new+1):
                            rhs_p[:,m] = rhs_0
                    else:
                        for m in range(1,M_new+1):
                            rhs_p[:,m] = func(u_p[:,m],t_sub[m])
                else:
                    u_p = np.copy(u_a)
                    rhs_p = np.copy(rhs_a)
                    theta = self.thetas[-1]
                    beta  = self.betas[-1]
                    beta_big  = self.betas_big[-1]
                    t_sub = tspan[it-1] + delta_t*beta_big
                
                #u_a=np.zeros((dim, M_new+1))
                #rhs_a= np.zeros((dim,M_new+1))
                #u_a[:,0]=U[:,it-1]
                #rhs_a[:,0] = rhs_0
                
                for m in range(1,M_new+1):
                    u_a[:,m]= u_a[:,m-1]+ delta_t*beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])\
                        +delta_t*sum([theta[r,m]*rhs_p[:,r] for r in range(M_new+1)])
                    rhs_a[:,m] =func(u_a[:,m],t_sub[m])
            # Extra iteration with no interpolation to restore the original accuracy
            u_p = np.copy(u_a)
            rhs_p = np.copy(rhs_a)
            theta = self.thetas[-1]
            beta  = self.betas[-1]
            beta_big  = self.betas_big[-1]
            t_sub = tspan[it-1] + delta_t*beta_big

            # u_a=np.zeros((dim, M_new+1))
            # rhs_a= np.zeros((dim,M_new+1))
            # u_a[:,0]=U[:,it-1]
            # rhs_a[:,0] = rhs_0
            for m in range(1,M_new+1):
                u_a[:,m]= u_a[:,m-1]+ delta_t*beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])\
                    +delta_t*sum([theta[r,m]*rhs_p[:,r] for r in range(M_new+1)])
                rhs_a[:,m] =func(u_a[:,m],t_sub[m])
            # update last value
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U


    def dec_order_control(self, func, tspan, y_0, err_tol):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        p_times=np.zeros(N_time)
        U[:,0]=y_0
        u_p = np.zeros((dim,self.M_sub+1))
        u_a = np.zeros((dim,self.M_sub+1))
        rhs_a = np.zeros((dim,self.M_sub+1))
        rhs_p = np.zeros((dim,self.M_sub+1))
        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            rhs_0 = func(U[:,it-1],tspan[it-1])
            u_p[:,:] = 0. #np.zeros((dim, 2))
            u_a[:,:] = 0. #np.zeros((dim, 2))
            rhs_p[:,:]=0.
            rhs_a[:,:]=0.
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
                #rhs_p[:,m] = rhs_0
                rhs_a[:,m] = rhs_0
            err_local = err_tol + 1.
            p=0
            while( p < self.n_iter-1 and err_local > err_tol ):
                p=p+1
                # New subtimesteps
                M_old = M_olds[p]#max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                
                #u_p[:,:]=0.
                rhs_p[:,:]=0.
                if p <= self.M_sub:
                    for idim in range(dim):
                        u_p[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@u_a[idim,:M_old+1]
                        u_p[idim,M_new] = u_a[idim,M_old]
#                        rhs_p[idim,:] = self.interpolation_matrices[p]@rhs_a[idim,:]
                    rhs_p[:,0] = rhs_0

                    theta     = self.thetas[p]
                    beta      = self.betas[p]
                    beta_big  = self.betas_big[p]
                    t_sub = tspan[it-1] + delta_t*beta_big
                    if p==1:
                        for m in range(1,M_new+1):
                            rhs_p[:,m] = rhs_0
                    else:
                        for m in range(1,M_new+1):
                            rhs_p[:,m] = func(u_p[:,m],t_sub[m])
                else:
                    u_p = np.copy(u_a)
                    rhs_p = np.copy(rhs_a)
                    theta = self.thetas[-1]
                    beta  = self.betas[-1]
                    beta_big  = self.betas_big[-1]
                    t_sub = tspan[it-1] + delta_t*beta_big

            
                for m in range(1,M_new+1):
                    u_a[:,m]= u_a[:,m-1]+ delta_t*beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])\
                        +delta_t*sum([theta[r,m]*rhs_p[:,r] for r in range(M_new+1)])
                    rhs_a[:,m] =func(u_a[:,m],t_sub[m])

                err_local = np.linalg.norm(u_a[:,M_new]-u_p[:,M_new]) /np.linalg.norm(u_a[:,M_new])
            p_times[it] = p


            U[:,it]=u_a[:,M_new]
        return tspan, U, p_times



class DeC_staggered_f:
    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.thetas=[]
        self.betas=[]
        self.interpolation_matrices = []
        theta, beta = self.compute_theta_DeC(2)
        self.thetas.append(theta)
        self.betas.append(beta)
        self.interpolation_matrices.append(None)
        self.name = f"DeC_stag_f_{self.nodes_type}"
        
        for p in range(1,self.M_sub+1):
            theta, beta = self.compute_theta_DeC(max(p+1,2))
            self.thetas.append(theta)
            self.betas.append(beta)
            self.interpolation_matrices.append(\
                    self.compute_interpolation_matrix(p+1))
        

    def compute_theta_DeC(self, n_subNodes=None):
        if n_subNodes is None:
            n_subNodes = self.n_subNodes
        nodes, w = get_nodes(n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        theta = np.zeros((n_subNodes,n_subNodes))
        beta = np.zeros(n_subNodes)
        for m in range(n_subNodes):
            beta[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(n_subNodes):
                theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return theta, beta
    
    def compute_interpolation_matrix(self,N):
        nodes_interpolation, _ = get_nodes(max(N,  2),self.nodes_type)
        nodes_basis, _         = get_nodes(max(N-1,2),self.nodes_type)
        A = np.zeros((max(N,2),max(N-1,2)))
        for m in range(len(nodes_basis)):
            A[:,m] = lagrange_basis(nodes_basis,nodes_interpolation,m)
        return A
        
    
    def compute_RK_from_DeC(self):
        #NRK = (self.n_iter)*(self.n_iter-1)//2+1
        NRK = (self.M_sub*(self.n_iter-1)+1) - int(self.M_sub*(self.M_sub-1)/2)
        self.ARK=np.zeros((NRK,NRK))  # (M_sub x K_corr +1)^2
        self.bRK=np.zeros(NRK)
        self.cRK=np.zeros(NRK)

        idx =1

        p = 1
        theta = self.thetas[p].transpose()
        interp_mat = self.interpolation_matrices[p+1]
        II=np.array([1.,1.])
        zeta = theta@II
        zeta_bar = zeta[1:]
        idx_new = idx + len(zeta_bar)
        self.ARK[idx:idx_new,0] = zeta_bar
        for p in range(2,self.n_iter):
            idx_old=idx
            idx=idx_new
            idx_new = idx + min(p,self.M_sub)
            theta = self.thetas[min(p,self.M_sub)].transpose()
            if p<self.M_sub+1:
                interp_mat = self.interpolation_matrices[p]
            else:
                interp_mat = np.eye(theta.shape[0])
            zeta = theta@interp_mat
            zeta_bar = zeta[1:,1:]
            zeta_zero = zeta[1:,0]
            self.ARK[idx:idx_new,idx_old:idx] = zeta_bar
            self.ARK[idx:idx_new,0] = zeta_zero

        bar_theta= theta[:,1:].transpose() # M_sub x (M_sub +1)
        theta0= bar_theta[:,0]  # M_sub x 1
        bar_theta= bar_theta[:,1:] #M_sub x M_sub
        self.bRK[0]=theta[-1, 0]
        self.bRK[-self.M_sub:]=theta[-1,1:]# bar_theta[self.M_sub-1,:]
        self.cRK = np.sum(self.ARK,1)
        return self.ARK,self.bRK,self.cRK
    
    def dec(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        U[:,0]=y_0
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        rhs = np.zeros((dim, self.M_sub+1))
        rhs_new = np.zeros((dim, self.M_sub+1))

        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            u_p[:,:]=0.
            u_a[:,:]=0.
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            rhs[:,0] = func(u_a[:,0],tspan[it-1])
            rhs_new[:,0] = rhs[:,0]
            for p in range(1,self.n_iter):
#                 print(p)
                M_old = M_olds[p] #max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                # u_p=np.zeros((dim, M_new))
                beta = self.betas[M_old]
                t_sub = tspan[it-1] + delta_t*beta
                # rhs= np.zeros((dim,M_old+1))
                if p==1:
                    for r in range(1,M_old+1):
                        rhs[:,r]=rhs[:,0]
                else:
                    for r in range(1,M_old+1):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                    
                # rhs_new=np.zeros((dim, M_new+1))
                if p <= self.M_sub:
                    for idim in range(dim):
                        rhs_new[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@rhs[idim,:M_old+1]
                        rhs_new[idim,M_new] = rhs[idim,M_old]
                    theta = self.thetas[p]
                else:
                    rhs_new = np.copy(rhs)
                    theta = self.thetas[-1]
#                 print(theta)
#                u_a=np.zeros((dim, M_new+1))
#                 print(u_a.shape)
                for m in range(1,M_new+1):
                    u_a[:,m]= U[:,it-1]+delta_t*sum([theta[r,m]*rhs_new[:,r] for r in range(M_new+1)])
            # Extra iteration with no interpolation to restore the original accuracy
            u_p = np.copy(u_a)
            theta = self.thetas[-1]
            beta = self.betas[-1]
            t_sub = tspan[it-1] + delta_t*beta
#             print(theta)
#             print(u_a.shape)
            # u_a=np.zeros((dim, M_new+1))
            # rhs= np.zeros((dim,M_new+1))
            for r in range(1,M_new+1):
                rhs[:,r]=func(u_p[:,r],t_sub[r])
            u_a[:,-1]= U[:,it-1]+delta_t*sum([theta[r,-1]*rhs[:,r] for r in range(M_new+1)])         
            U[:,it]=u_a[:,-1]
        return tspan, U
        
    def dec_order_control(self, func, tspan, y_0, err_tol):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        p_times=np.zeros(N_time)
        U[:,0]=y_0
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        rhs = np.zeros((dim, self.M_sub+1))
        rhs_new = np.zeros((dim, self.M_sub+1))

        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            u_p[:,:]=0.
            u_a[:,:]=0.
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            rhs[:,0] = func(u_a[:,0],tspan[it-1])
            rhs_new[:,0] = rhs[:,0]
            err_local = err_tol + 1.
            p=0
            while( p < self.n_iter-1):# and err_local > err_tol ):
                p=p+1
            # for p in range(1,self.n_iter):
#                 print(p)
                M_old = M_olds[p] #max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                u_p=np.copy(u_a)
                beta = self.betas[M_old]
                t_sub = tspan[it-1] + delta_t*beta
                # rhs= np.zeros((dim,M_old+1))
                if p==1:
                    for r in range(1,M_old+1):
                        rhs[:,r]=rhs[:,0]
                else:
                    for r in range(1,M_old+1):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                    
                # rhs_new=np.zeros((dim, M_new+1))
                if p <= self.M_sub:
                    for idim in range(dim):
                        rhs_new[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@rhs[idim,:M_old+1]
                        rhs_new[idim,M_new] = rhs[idim,M_old]
                    theta = self.thetas[p]
                else:
                    rhs_new = np.copy(rhs)
                    theta = self.thetas[-1]
#                 print(theta)
#                u_a=np.zeros((dim, M_new+1))
#                 print(u_a.shape)
                u_a[:,M_new]= U[:,it-1]+delta_t*sum([theta[r,M_new]*rhs_new[:,r] for r in range(M_new+1)])
                err_local = np.linalg.norm(u_a[:,M_new]-u_p[:,M_old]) /np.linalg.norm(u_a[:,M_new])
                if err_local<= err_tol:
                    break

                for m in range(1,M_new):
                    u_a[:,m]= U[:,it-1]+delta_t*sum([theta[r,m]*rhs_new[:,r] for r in range(M_new+1)])

            p_times[it] = p

            U[:,it]=u_a[:,M_new]
        return tspan, U, p_times

    
    
class DeC_small_sub_staggered_f:
    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.thetas=[]
        self.betas=[]
        self.thetas_big=[]
        self.betas_big=[]
        self.interpolation_matrices = []
        theta, beta, theta_big, beta_big = self.compute_theta_DeC(2)
        self.thetas.append(theta)
        self.betas.append(beta)
        self.thetas_big.append(theta_big)
        self.betas_big.append(beta_big)
        self.interpolation_matrices.append(None)
        self.name = f"DeC_small_stag_f_{self.nodes_type}"
        
        for p in range(1,self.M_sub+1):
            theta, beta, theta_big, beta_big  = self.compute_theta_DeC(max(p+1,2))
            self.thetas.append(theta)
            self.betas.append(beta)
            self.thetas_big.append(theta_big)
            self.betas_big.append(beta_big)
            self.interpolation_matrices.append(\
                    self.compute_interpolation_matrix(p+1))
        

    def compute_interpolation_matrix(self,N):
        nodes_interpolation, _ = get_nodes(max(N,  2),self.nodes_type)
        nodes_basis, _         = get_nodes(max(N-1,2),self.nodes_type)
        A = np.zeros((max(N,2),max(N-1,2)))
        for m in range(len(nodes_basis)):
            A[:,m] = lagrange_basis(nodes_basis,nodes_interpolation,m)
        return A
        
    def compute_theta_DeC(self,n_subNodes=None):        
        if n_subNodes is None:
            n_subNodes = self.n_subNodes
        nodes, w = get_nodes(n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta = np.zeros((n_subNodes,n_subNodes))
        self.beta = np.zeros(n_subNodes)
        for m in range(n_subNodes):
            self.beta[m] = nodes[m]-nodes[max(m-1,0)]
            nodes_m = int_nodes*(nodes[m]-nodes[max(m-1,0)])+nodes[max(m-1,0)]
            w_m = int_w*(nodes[m]-nodes[max(m-1,0)])
            for r in range(n_subNodes):
                self.theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        # Original DeC coefficients
        nodes, w = get_nodes(n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta_big = np.zeros((n_subNodes,n_subNodes))
        self.beta_big = np.zeros(n_subNodes)
        for m in range(n_subNodes):
            self.beta_big[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(n_subNodes):
                self.theta_big[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return self.theta, self.beta, self.theta_big, self.beta_big
    
    
    def compute_RK_from_DeC(self):
        NRK = self.M_sub*self.n_iter - int(self.M_sub*(self.M_sub-1)//2)
        #self.M_sub*(self.M_sub+3)//2 
        
        self.ARK=np.zeros((NRK+1,NRK+1)) # one extra row that becomes b
        self.bRK=np.zeros(NRK)
        self.cRK=np.zeros(NRK+1)


        idx =1

        p = 1
        self.ARK[1,0] = 1.
        beta=self.betas[p]

        idx_old = 0
        idx_p   = 1
        idx_pp1 = 2

        self.cRK[idx_p:idx_pp1]=self.ARK[idx_p:idx_pp1,0]
        for p in range(2,self.n_iter+1):
            Mloc = min(p,self.M_sub)
            idx_old   = idx_p
            idx_p     = idx_pp1
            idx_pp1   = idx_pp1 + Mloc
                       

            theta     = self.thetas_big[Mloc].transpose()
            beta_big  = self.betas_big[Mloc]
            beta      = self.betas[Mloc]
            if p<self.M_sub+1:                            
                interp_mat = self.interpolation_matrices[min(p,self.M_sub)]
            else:
                interp_mat = np.eye(theta.shape[0])
            
            emme_theta = theta@interp_mat
            emme_theta_bar = emme_theta[1:,1:]
            emme_theta_zero = emme_theta[1:,0]

            gamma_mat = np.zeros((Mloc+1,Mloc+1))
            for m in range(1,min(p+1,self.M_sub+1)):
                gamma_mat[m:,m-1] = beta[m]
            
            gamma_bar = gamma_mat[1:,1:]
            emme_gamma = gamma_mat@interp_mat
            emme_gamma_bar = emme_gamma[1:,1:]
            emme_gamma_zero = emme_gamma[1:,0]
            gamma_mat_bar = gamma_mat[1:,1:]
            gamma_mat_zero = gamma_mat[1:,0]

            # y^(p)
            self.ARK[idx_p:idx_pp1,0] = emme_theta_zero+gamma_mat_zero-emme_gamma_zero
            self.ARK[idx_p:idx_pp1,idx_old:idx_p] = emme_theta_bar-emme_gamma_bar
            self.ARK[idx_p:idx_pp1,idx_p:idx_pp1] = gamma_bar
            self.cRK[idx_p:idx_pp1]=beta_big[1:]

            
        # Last iteration
        self.bRK=self.ARK[-1,:-1]
        self.ARK=self.ARK[:-1,:-1]
        self.cRK=self.cRK[:-1]
        return self.ARK,self.bRK,self.cRK

    
    def dec(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        U[:,0]=y_0
        u_p = np.zeros((dim,self.M_sub+1))
        u_a = np.zeros((dim,self.M_sub+1))
        rhs_a = np.zeros((dim,self.M_sub+1))
        rhs_p = np.zeros((dim,self.M_sub+1))
        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            rhs_0 = func(U[:,it-1],tspan[it-1])
            # u_p=np.zeros((dim,2))
            # u_a=np.zeros((dim,2))
            # rhs_p=np.zeros((dim,2))
            # rhs_a=np.zeros((dim,2))
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
                rhs_p[:,m] = rhs_0
                rhs_a[:,m] = rhs_0
            for p in range(1,self.n_iter):
                # New subtimesteps
                M_old = M_olds[p]#max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                
                # u_p=np.zeros((dim, M_new+1))
                # rhs_p= np.zeros((dim,M_new+1))
                if p <= self.M_sub:
                    for idim in range(dim):
                        rhs_p[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@rhs_a[idim,:M_old+1]
                        rhs_p[idim,M_new] = rhs_a[idim,M_old]
                    theta = self.thetas[p]
                    beta  = self.betas[p]
                    beta_big = self.betas_big[p]
                else:
                    rhs_p = np.copy(rhs_a)
                    theta = self.thetas[-1]
                    beta  = self.betas[-1]
                    beta_big = self.betas_big[-1]

                t_sub = tspan[it-1]+delta_t*beta_big
                
                # u_a=np.zeros((dim, M_new+1))
                # rhs_a= np.zeros((dim,M_new+1))
                # u_a[:,0]=U[:,it-1]
                # rhs_a[:,0] = rhs_0
                
                for m in range(1,M_new+1):
                    u_a[:,m]= u_a[:,m-1]+ delta_t*beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])\
                        +delta_t*sum([theta[r,m]*rhs_p[:,r] for r in range(M_new+1)])
                    rhs_a[:,m] =func(u_a[:,m],t_sub[m])
            # Extra iteration with no interpolation to restore the original accuracy
            u_p = np.copy(u_a)
            rhs_p = np.copy(rhs_a)
            theta = self.thetas[-1]
            beta  = self.betas[-1]
            beta_big  = self.betas_big[-1]
            t_sub = tspan[it-1]+delta_t*beta_big
            # u_a=np.zeros((dim, M_new+1))
            # rhs_a= np.zeros((dim,M_new+1))
            # u_a[:,0]=U[:,it-1]
            # rhs_a[:,0] = rhs_0
            for m in range(1,M_new+1):
                u_a[:,m]= u_a[:,m-1]+ delta_t*beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])\
                    +delta_t*sum([theta[r,m]*rhs_p[:,r] for r in range(M_new+1)])
                rhs_a[:,m] =func(u_a[:,m],t_sub[m])
            # update last value
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U


    def dec_order_control(self, func, tspan, y_0, err_tol):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        p_times=np.zeros(N_time)
        U[:,0]=y_0
        u_p = np.zeros((dim,self.M_sub+1))
        u_a = np.zeros((dim,self.M_sub+1))
        rhs_a = np.zeros((dim,self.M_sub+1))
        rhs_p = np.zeros((dim,self.M_sub+1))
        M_olds = [0]
        M_news = [0]
        for p in range(1,self.n_iter):
            M_olds.append( max(min(p-1,self.M_sub),1))
            M_news.append( max(min(p,  self.M_sub),1))

        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            rhs_0 = func(U[:,it-1],tspan[it-1])
            # u_p=np.zeros((dim,2))
            # u_a=np.zeros((dim,2))
            # rhs_p=np.zeros((dim,2))
            # rhs_a=np.zeros((dim,2))
            for m in range(2):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
                rhs_p[:,m] = rhs_0
                rhs_a[:,m] = rhs_0

            err_local = err_tol + 1.
            p=0
            while( p < self.n_iter-1 and err_local > err_tol ):
                p=p+1
                # New subtimesteps
                M_old = M_olds[p]#max(min(p-1,self.M_sub),1)
                M_new = M_news[p] #max(min(p,  self.M_sub),1)
                
                u_p=np.copy(u_a)
                # rhs_p= np.zeros((dim,M_new+1))
                if p <= self.M_sub:
                    for idim in range(dim):
                        rhs_p[idim,1:M_new] = self.interpolation_matrices[p][1:M_new,:]@rhs_a[idim,:M_old+1]
                        rhs_p[idim,M_new] = rhs_a[idim,M_old]
                    theta = self.thetas[p]
                    beta  = self.betas[p]
                    beta_big = self.betas_big[p]
                else:
                    rhs_p = np.copy(rhs_a)
                    theta = self.thetas[-1]
                    beta  = self.betas[-1]
                    beta_big = self.betas_big[-1]

                t_sub = tspan[it-1]+delta_t*beta_big
                
                # u_a=np.zeros((dim, M_new+1))
                # rhs_a= np.zeros((dim,M_new+1))
                # u_a[:,0]=U[:,it-1]
                # rhs_a[:,0] = rhs_0
                
                for m in range(1,M_new+1):
                    u_a[:,m]= u_a[:,m-1]+ delta_t*beta[m]*(rhs_a[:,m-1]-rhs_p[:,m-1])\
                        +delta_t*sum([theta[r,m]*rhs_p[:,r] for r in range(M_new+1)])
                    rhs_a[:,m] =func(u_a[:,m],t_sub[m])
                err_local = np.linalg.norm(u_a[:,M_new]-u_p[:,M_old]) /np.linalg.norm(u_a[:,M_new])
            p_times[it] = p
            U[:,it]=u_a[:,M_new]
        return tspan, U, p_times




def lagrange_basis(nodes,x,k):
    y=np.zeros(x.size)
    for ix, xi in enumerate(x):
        tmp=[(xi-nodes[j])/(nodes[k]-nodes[j])  for j in range(len(nodes)) if j!=k]
        y[ix]=np.prod(tmp)
    return y

def get_nodes(order,nodes_type):
    if nodes_type=="equispaced":
        nodes,w = equispaced(order)
    elif nodes_type == "gaussLegendre":
        nodes,w = leggauss(order)
    elif nodes_type == "gaussLobatto":
        nodes, w = lglnodes(order-1,10**-15)
    nodes=nodes*0.5+0.5
    w = w*0.5
    return nodes, w
        




def patankar_type_dec(prod_p,dest_p,rhs_p,delta_t,m,M_sub,Theta,u_p,dim):
    mass= np.eye(dim)
    RHS= u_p[:,0]
    for i in range(dim):
        for r in range(M_sub+1):
            RHS[i]=RHS[i]+delta_t*Theta[r,m]*rhs_p[i,r]
            if Theta[r,m]>0:
                for j in range(dim):
                    mass[i,j]=mass[i,j]-delta_t*Theta[r,m]*(prod_p[i,j,r]/u_p[j,m])
                    mass[i,i]=mass[i,i]+ delta_t*Theta[r,m]*(dest_p[i,j,r]/u_p[i,m])
            elif Theta[r,m]<0:
                for j in range(dim):
                    mass[i,i]=mass[i,i]- delta_t*Theta[r,m]*(prod_p[i,j,r]/u_p[i,m])
                    mass[i,j]=mass[i,j]+ delta_t*Theta[r,m]*(dest_p[i,j,r]/u_p[j,m])
    return np.linalg.solve(mass,RHS)
