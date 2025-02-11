import numpy as np
from mpmath import mp
from precision_mp import precision
from quadrature_mp import equispaced, lobatto, legendre
from quadrature_mp import orthogonal_legendre_polynomials, derivative_orthogonal_legendre_polynomials
from quadrature_mp import lagrange_basis, lagrange_deriv
from quadrature_mp import convert_mp_2_np
mp.dps=precision

class ADER:
    def __init__(self, M_sub, n_iter, nodes_type):
        """ 
        ADER class with input:
         * M_sub number of subtimesteps (i.e. M_sub +1 nodes), if M_sub=-1 then optimal one
         * n_iter number of iterations and expected order of accuracy
         * nodes_type among "equispaced" "gaussLobatto" "gaussLegendre"
         """
        M_sub = check_M(M_sub, n_iter, nodes_type)
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter        
        self.nodes_type = nodes_type
        self.compute_ADER_matrices()
        self.name = f"ADER_{self.nodes_type}"  

    def compute_ADER_matrices(self):
        """
        Compute ADER matrices: 
        ADER = phi_i(1)phi_j(1) -int_0^1 d(phi_i)/dt phi_j dt 
        RHS  = int_0^1 phi_i phi_j dt
        """
        self.x_poly, self.w_poly, self.ADERmat, self.RHSmat, self.bADER, self.evolMat, \
            self.recon_ADER,  self.nodes_quad, self.w_quad, self.basis_quad, self.evol_un = \
            getADER_matrix(self.n_subNodes, self.nodes_type)

        
        self.skip_first_rhs = np.zeros(self.n_iter,dtype=bool)

        # Iterative procedure
        for k in range(1,self.n_iter):
            if sum(abs(self.evolMat[0,:]))<1e-15 or (sum(self.evolMat[0,:])<1e-15 and k==1):
                self.skip_first_rhs[k] = True
            else:
                self.skip_first_rhs[k] = False

        return self.ADERmat, self.RHSmat, self.bADER, self.evolMat
    
    def compute_RK(self):
        "Compute the RK  matrices of the ADER scheme (to be checked for GLB and equi)"
        Mnodes = self.n_subNodes
        K_corr = self.n_iter-1
        NRK = K_corr*(Mnodes)+1
        self.ARK = np.zeros((NRK,NRK))
        self.bRK = np.zeros((NRK,1))
        self.cRK = np.zeros((NRK,1))
        
        bADER = self.bADER
        Qmat = self.evolMat
        if K_corr>0:
            Pvec=Qmat@np.ones((Mnodes,1))
            for z in range(Mnodes):
                self.cRK[z+1]=Pvec[z]
                self.ARK[z+1,0]=Pvec[z]
        else:
            self.ARK[:,:]=0.
            self.bRK[0]=1.
            self.cRK[0]=0.
            self.NRK = 1
            return self.ARK,self.bRK,self.cRK

        for k in range(1,K_corr):
            rowShift = 1+k*(Mnodes)
            columnShift= 1+(k-1)*(Mnodes)
            for z in range(Mnodes):
                self.cRK[z+rowShift]=Pvec[z]
                for j in range(Mnodes):
                    self.ARK[z+rowShift,j+columnShift]=Qmat[z,j]

        for k in range(Mnodes):
            self.bRK[NRK-Mnodes+k] = bADER[k]
        
        self.NRK = NRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARK[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARK[:,0] = self.ARK[:,0] + self.ARK[:,i]
                self.bRK[0] = self.bRK[0] + self.bRK[i]
                # delete i-th row and column
                self.ARK = np.delete(self.ARK, i,0)
                self.ARK = np.delete(self.ARK, i,1)
                self.cRK = np.delete(self.cRK,i,0)
                self.bRK = np.delete(self.bRK,i,0)
            else:
                i=i+1


        return self.ARK,self.bRK,self.cRK

    def compute_IMEX_RK(self):
        "Compute the IMEX RK  matrices of the ADER scheme (to be checked for GLB and equi)"
        Mnodes = self.n_subNodes
        K_corr = self.n_iter
        NRK = K_corr*(Mnodes)+1
        self.ARKEX = np.zeros((NRK,NRK))
        self.bRKEX = np.zeros((NRK,1))
        self.ARKIM = np.zeros((NRK,NRK))
        self.bRKIM = np.zeros((NRK,1))
        self.cRK = np.zeros((NRK,1))
        
        bADER = self.bADER
        Qmat = self.evolMat
        if K_corr==0:
            self.NRK = 2
            self.ARKEX[1,0]=1.
            self.bRKEX[0]=1.
            self.ARKIM[1,1]=1.
            self.bRKIM[1]=1.
            self.cRK[1]=1.
            return self.ARKEX,self.bRKEX, self.ARKIM, self.bRKIM ,self.cRK

        Pvec=Qmat@np.ones((Mnodes,1))
        for z in range(Mnodes):
            self.cRK[z+1]=Pvec[z]
            self.ARKEX[z+1,0]=Pvec[z]
            for j in range(Mnodes):
                self.ARKIM[z+1,j+1]=Qmat[z,j]        

        for k in range(1,K_corr):
            rowShift = 1+k*(Mnodes)
            columnShift= 1+(k-1)*(Mnodes)
            for z in range(Mnodes):
                self.cRK[z+rowShift]=Pvec[z]
                for j in range(Mnodes):
                    self.ARKEX[z+rowShift,j+columnShift]=Qmat[z,j]
                    self.ARKIM[z+rowShift,j+rowShift]   =Qmat[z,j]

        for k in range(Mnodes):
            self.bRKIM[NRK-Mnodes+k] = bADER[k]
            self.bRKEX[NRK-2*Mnodes+k] = bADER[k]
        
        self.NRK = NRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARKEX[i,:]))<1e-17 and sum(abs(self.ARKIM[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARKEX[:,0] = self.ARKEX[:,0] + self.ARKEX[:,i]
                self.bRKEX[0] = self.bRKEX[0] + self.bRKEX[i]
                self.ARKIM[:,0] = self.ARKIM[:,0] + self.ARKIM[:,i]
                self.bRKIM[0] = self.bRKIM[0] + self.bRKIM[i]
                # delete i-th row and column
                self.ARKEX = np.delete(self.ARKEX, i,0)
                self.ARKEX = np.delete(self.ARKEX, i,1)
                self.ARKIM = np.delete(self.ARKIM, i,1)
                self.ARKIM = np.delete(self.ARKIM, i,0)
                self.cRK = np.delete(self.cRK,i,0)
                self.bRKEX = np.delete(self.bRKEX,i,0)
                self.bRKIM = np.delete(self.bRKIM,i,0)
            else:
                i=i+1


        return self.ARKEX,self.bRKEX, self.ARKIM, self.bRKIM,self.cRK


    def ader(self, func, tspan, y_0):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        if self.nodes_type=="orthogonal":
            return self.ader_orthogonal(func,tspan, y_0)
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_a=np.zeros((dim, M_nodes))
        rhs= np.zeros((dim,M_nodes))
        U[:,0]=y_0


        
        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])
            t_sub = tspan[it-1]+ delta_t*self.x_poly

            # Initialize strucutres
            for m in range(M_nodes):
                u_a[:,m]=U[:,it-1]
            # Compute the flux of the initial state and copy to other stages
            rhs0=func(u_a[:,0],t_sub[0])
            for m in range(M_nodes):
                rhs[:,m] = rhs0
            
            # Iterative procedure
            for k in range(1,K_corr):
                # update value of u
                for d in range(dim):
                    u_a[d,:] = U[d,it-1] + delta_t*np.matmul(self.evolMat,rhs[d,:])
                
                # update value of rhs
                if self.skip_first_rhs[k]:
                    rhs[:,0]=rhs0
                    for r in range(1,M_nodes):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                else:
                    for r in range(M_nodes):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
            # final update 
            for d in range(dim):
                U[d,it]=U[d,it-1] + delta_t*np.dot(bADER, rhs[d,:])
        return tspan, U




    def ader_orthogonal(self, func, tspan, y_0):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_a=np.zeros((dim, M_nodes))
        U[:,0]=y_0

        N_quad = len(self.w_quad)
        Fu_p_quad = np.zeros((dim, N_quad))
        tmp = np.zeros(M_nodes)

        invMass = np.linalg.inv(self.ADERmat)
        
        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])
            t_sub = tspan[it-1]+ delta_t*self.x_poly

            # Initialize strucutres
            u_a[:,:] = 0. 
            u_a[:,0] = U[:,it-1]
            
            # Iterative procedure
            for k in range(1,K_corr+1):
                # Evaluating F( sum phi_j(x_q) * u^j)
                u_rec = u_a @ (self.basis_quad.transpose())
                for jq, xq in enumerate(self.nodes_quad):
                    Fu_p_quad[:,jq] = func(u_rec[:,jq], tspan[it-1]+delta_t*xq)

                for d in range(dim):
                    # Computing the integral int_0^1 phi_j F(u) dt
                    for j in range(M_nodes):
                        tmp[j] = sum([ Fu_p_quad[d,jq]*self.basis_quad[jq,j]*self.w_quad[jq] for jq in range(N_quad) ])
                    # updating ua as ua=M^{-1}*1*u_n + d *M^{-1} RHS(u) 
                    u_a[d,:] = self.evol_un.flatten()*U[d,it-1] + delta_t*np.matmul(invMass,tmp).flatten()

            # final update 
            for d in range(dim):
                U[d,it]= u_a[d,:]@self.recon_ADER
        return tspan, U




class ClassicalADER:
    def __init__(self, M_sub, n_iter, nodes_type):
        """ 
        ADER class with input:
         * M_sub number of subtimesteps (i.e. M_sub +1 nodes), if M_sub=-1 then optimal one
         * n_iter number of iterations and expected order of accuracy
         * nodes_type among "equispaced" "gaussLobatto" "gaussLegendre"
         """
        if M_sub ==-1:
            if nodes_type != "gaussLegendre":
                M_sub = max(n_iter- 1,1)
            else:
                M_sub = max(n_iter-1,0)
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter        
        self.nodes_type = nodes_type
        self.compute_ADER_matrices()
        self.name = f"ClassicalADER_{self.nodes_type}"  

    def compute_ADER_matrices(self):
        """
        Compute ADER matrices: 
        ADER = phi_i(1)phi_j(1) -int_0^1 d(phi_i)/dt phi_j dt 
        RHS  = int_0^1 phi_i phi_j dt
        """
        self.x_poly, self.w_poly, self.ADERmat, self.RHSmat, self.bADER, self.evolMat, \
            self.recon_ADER,  self.nodes_quad, self.w_quad, self.basis_quad, self.evol_un = \
            getADER_matrix(self.n_subNodes, self.nodes_type)

        
        self.skip_first_rhs = np.zeros(self.n_iter,dtype=bool)

        # Iterative procedure
        for k in range(1,self.n_iter):
            if sum(abs(self.evolMat[0,:]))<1e-15 or (sum(self.evolMat[0,:])<1e-15 and k==1):
                self.skip_first_rhs[k] = True
            else:
                self.skip_first_rhs[k] = False

        return self.ADERmat, self.RHSmat, self.bADER, self.evolMat
    
    def compute_RK(self):
        "Compute the RK  matrices of the ADER scheme (to be checked for GLB and equi)"
        Mnodes = self.n_subNodes
        K_corr = self.n_iter-1
        NRK = K_corr*(Mnodes)+1
        self.ARK = np.zeros((NRK,NRK))
        self.bRK = np.zeros((NRK,1))
        self.cRK = np.zeros((NRK,1))
        
        bADER = self.bADER
        Qmat = self.evolMat
        if K_corr>0:
            Pvec=Qmat@np.ones((Mnodes,1))
            for z in range(Mnodes):
                self.cRK[z+1]=Pvec[z]
                self.ARK[z+1,0]=Pvec[z]
        else:
            self.ARK[:,:]=0.
            self.bRK[0]=1.
            self.cRK[0]=0.
            self.NRK = 1
            return self.ARK,self.bRK,self.cRK

        for k in range(1,K_corr):
            rowShift = 1+k*(Mnodes)
            columnShift= 1+(k-1)*(Mnodes)
            for z in range(Mnodes):
                self.cRK[z+rowShift]=Pvec[z]
                for j in range(Mnodes):
                    self.ARK[z+rowShift,j+columnShift]=Qmat[z,j]

        for k in range(Mnodes):
            self.bRK[NRK-Mnodes+k] = bADER[k]
        
        self.NRK = NRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARK[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARK[:,0] = self.ARK[:,0] + self.ARK[:,i]
                self.bRK[0] = self.bRK[0] + self.bRK[i]
                # delete i-th row and column
                self.ARK = np.delete(self.ARK, i,0)
                self.ARK = np.delete(self.ARK, i,1)
                self.cRK = np.delete(self.cRK,i,0)
                self.bRK = np.delete(self.bRK,i,0)
            else:
                i=i+1


        return self.ARK,self.bRK,self.cRK

    def compute_IMEX_RK(self):
        "Compute the IMEX RK  matrices of the ADER scheme (to be checked for GLB and equi)"
        Mnodes = self.n_subNodes
        K_corr = self.n_iter
        NRK = K_corr*(Mnodes)+1
        self.ARKEX = np.zeros((NRK,NRK))
        self.bRKEX = np.zeros((NRK,1))
        self.ARKIM = np.zeros((NRK,NRK))
        self.bRKIM = np.zeros((NRK,1))
        self.cRK = np.zeros((NRK,1))
        
        bADER = self.bADER
        Qmat = self.evolMat
        if K_corr==0:
            self.NRK = 2
            self.ARKEX[1,0]=1.
            self.bRKEX[0]=1.
            self.ARKIM[1,1]=1.
            self.bRKIM[1]=1.
            self.cRK[1]=1.
            return self.ARKEX,self.bRKEX, self.ARKIM, self.bRKIM ,self.cRK

        Pvec=Qmat@np.ones((Mnodes,1))
        for z in range(Mnodes):
            self.cRK[z+1]=Pvec[z]
            self.ARKEX[z+1,0]=Pvec[z]
            for j in range(Mnodes):
                self.ARKIM[z+1,j+1]=Qmat[z,j]        

        for k in range(1,K_corr):
            rowShift = 1+k*(Mnodes)
            columnShift= 1+(k-1)*(Mnodes)
            for z in range(Mnodes):
                self.cRK[z+rowShift]=Pvec[z]
                for j in range(Mnodes):
                    self.ARKEX[z+rowShift,j+columnShift]=Qmat[z,j]
                    self.ARKIM[z+rowShift,j+rowShift]   =Qmat[z,j]

        for k in range(Mnodes):
            self.bRKIM[NRK-Mnodes+k] = bADER[k]
            self.bRKEX[NRK-2*Mnodes+k] = bADER[k]
        
        self.NRK = NRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARKEX[i,:]))<1e-17 and sum(abs(self.ARKIM[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARKEX[:,0] = self.ARKEX[:,0] + self.ARKEX[:,i]
                self.bRKEX[0] = self.bRKEX[0] + self.bRKEX[i]
                self.ARKIM[:,0] = self.ARKIM[:,0] + self.ARKIM[:,i]
                self.bRKIM[0] = self.bRKIM[0] + self.bRKIM[i]
                # delete i-th row and column
                self.ARKEX = np.delete(self.ARKEX, i,0)
                self.ARKEX = np.delete(self.ARKEX, i,1)
                self.ARKIM = np.delete(self.ARKIM, i,1)
                self.ARKIM = np.delete(self.ARKIM, i,0)
                self.cRK = np.delete(self.cRK,i,0)
                self.bRKEX = np.delete(self.bRKEX,i,0)
                self.bRKIM = np.delete(self.bRKIM,i,0)
            else:
                i=i+1


        return self.ARKEX,self.bRKEX, self.ARKIM, self.bRKIM,self.cRK


    def ader(self, func, tspan, y_0):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        if self.nodes_type=="orthogonal":
            return self.ader_orthogonal(func,tspan, y_0)
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_a=np.zeros((dim, M_nodes))
        rhs= np.zeros((dim,M_nodes))
        U[:,0]=y_0


        
        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])
            t_sub = tspan[it-1]+ delta_t*self.x_poly

            # Initialize strucutres
            for m in range(M_nodes):
                u_a[:,m]=U[:,it-1]
            # Compute the flux of the initial state and copy to other stages
            rhs0=func(u_a[:,0],t_sub[0])
            for m in range(M_nodes):
                rhs[:,m] = rhs0
            
            # Iterative procedure
            for k in range(1,K_corr):
                # update value of u
                for d in range(dim):
                    u_a[d,:] = U[d,it-1] + delta_t*np.matmul(self.evolMat,rhs[d,:])
                
                # update value of rhs
                if self.skip_first_rhs[k]:
                    rhs[:,0]=rhs0
                    for r in range(1,M_nodes):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                else:
                    for r in range(M_nodes):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
            # final update 
            for d in range(dim):
                U[d,it]=U[d,it-1] + delta_t*np.dot(bADER, rhs[d,:])
        return tspan, U




    def ader_orthogonal(self, func, tspan, y_0):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_a=np.zeros((dim, M_nodes))
        U[:,0]=y_0

        N_quad = len(self.w_quad)
        Fu_p_quad = np.zeros((dim, N_quad))
        tmp = np.zeros(M_nodes)

        invMass = np.linalg.inv(self.ADERmat)
        
        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])
            t_sub = tspan[it-1]+ delta_t*self.x_poly

            # Initialize strucutres
            u_a[:,:] = 0. 
            u_a[:,0] = U[:,it-1]
            
            # Iterative procedure
            for k in range(1,K_corr+1):
                # Evaluating F( sum phi_j(x_q) * u^j)
                u_rec = u_a @ (self.basis_quad.transpose())
                for jq, xq in enumerate(self.nodes_quad):
                    Fu_p_quad[:,jq] = func(u_rec[:,jq], tspan[it-1]+delta_t*xq)

                for d in range(dim):
                    # Computing the integral int_0^1 phi_j F(u) dt
                    for j in range(M_nodes):
                        tmp[j] = sum([ Fu_p_quad[d,jq]*self.basis_quad[jq,j]*self.w_quad[jq] for jq in range(N_quad) ])
                    # updating ua as ua=M^{-1}*1*u_n + d *M^{-1} RHS(u) 
                    u_a[d,:] = self.evol_un.flatten()*U[d,it-1] + delta_t*np.matmul(invMass,tmp).flatten()

            # final update 
            for d in range(dim):
                U[d,it]= u_a[d,:]@self.recon_ADER
        return tspan, U



class ADER_L2:
    """ ADER class with efficient procedure with L2 projection between different iterations """
    def __init__(self, M_sub, n_iter, nodes_type):
        """ 
        ADER class with input:
         * M_sub number of subtimesteps (i.e. M_sub +1 nodes), if M_sub=-1 then optimal one
         * n_iter number of iterations and expected order of accuracy
         * nodes_type among "equispaced" "gaussLobatto" "gaussLegendre"
         """
        M_sub = check_M(M_sub, n_iter, nodes_type)
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter        
        self.nodes_type = nodes_type

        self.set_increasing_orders()

        self.compute_ADER_matrices()
        self.name = f"ADER_L2_{self.nodes_type}"  

    def set_increasing_orders(self):
        self.M_subs = np.zeros(self.n_iter,dtype=np.int32)
        self.n_subNodess = np.zeros(self.n_iter,dtype=np.int32)

        for i in range(self.n_iter):
            if i <1:
                if self.nodes_type in ["gaussLobatto","equispaced","equispaced_lumped"]:
                    self.n_subNodess[i] = 2
                elif self.nodes_type in ["gaussLegendre", "orthogonal"]:
                    self.n_subNodess[i] = 1
            else:
                self.n_subNodess[i] = min(i+1,self.n_subNodes)

    def compute_ADER_matrices(self):
        """
        Compute ADER matrices for all basis functions up to the final order 
        ADER = phi_i(1)phi_j(1) -int_0^1 d(phi_i)/dt phi_j dt 
        RHS  = int_0^1 phi_i phi_j dt
        """
        self.x_polys=[np.empty(0)];
        self.evolMats=[np.empty(0)]
        self.recon_ADERs=[np.empty(0)]
        self.evol_uns=[np.empty(0)]
        self.basis_quads=[np.empty(0)]
        self.nodes_quads=[np.empty(0)]
        self.w_quads=[np.empty(0)]
        self.invMasss=[np.empty(0)]
        for i in range(self.n_iter-1):
            n_subNodes_old = self.n_subNodess[i]
            n_subNodes_new = self.n_subNodess[i+1]
            x_poly, _, ADERmat, _, _, _, recon_ADER,\
                nodes_quad, w_quad, basis_quad, evol_un = \
            getADER_matrix(n_subNodes_new, self.nodes_type)
            self.x_polys.append(x_poly)
            self.recon_ADERs.append(recon_ADER)
            self.evol_uns.append(evol_un)
            self.nodes_quads.append(nodes_quad)
            self.w_quads.append(w_quad)
            self.basis_quads.append(basis_quad)
            self.invMasss.append(np.linalg.inv(ADERmat))
            evolMat = getADER_evol_matr_L2_proj(n_subNodes_old,n_subNodes_new, self.nodes_type)
            self.evolMats.append(evolMat)
        
        n_subNodes_old = self.n_subNodess[self.n_iter-1]
        x_poly, _, ADERmat, _, self.bADER, evolMat, self.recon_ADER,\
                self.nodes_quad, self.w_quad, self.basis_quad, self.evol_un = \
            getADER_matrix(n_subNodes_old, self.nodes_type)
        self.x_polys.append(x_poly)
        self.evolMats.append(evolMat)
        self.evol_uns.append(self.evol_un)
        self.nodes_quads.append(self.nodes_quad)
        self.w_quads.append(self.w_quad)
        self.basis_quads.append(self.basis_quad)
        self.invMasss.append(np.linalg.inv(ADERmat))
        self.recon_ADERs.append(self.recon_ADER)

        self.skip_first_rhs = np.zeros(self.n_iter,dtype=bool)
        # Iterative procedure
        for k in range(1,self.n_iter):
            if sum(abs(self.evolMats[k][0,:]))<1e-15 or (sum(self.evolMats[k][0,:])<1e-15 and k==1):
                self.skip_first_rhs[k] = True
            else:
                self.skip_first_rhs[k] = False


        #return self.ADERmat, self.RHSmat, self.bADER, self.evolMat
    
    def compute_RK(self):
        "Compute the RK  matrices of the ADER scheme"
        Mnodes = self.n_subNodes
        K_corr = self.n_iter
        #Could be optimized in GLB and equi (extra first step)
        NRK = sum(self.n_subNodess)
        ARK = np.zeros((NRK,NRK))
        bRK = np.zeros((NRK,1))
        cRK = np.zeros((NRK,1))
        
        bADER = self.bADER

        rowShift = 0
        columnShift = 0
        Mnodes_new = self.n_subNodess[0]        
        # actung, it works only for first step with 1 node
        for k in range(1,K_corr):
            Mnodes_old = Mnodes_new
            Mnodes_new = self.n_subNodess[k]
            rowShift = rowShift + Mnodes_old
            Qmat = self.evolMats[k]
            Pvec=Qmat@np.ones((Qmat.shape[1],1))
            for z in range(Mnodes_new):
                cRK[z+rowShift]=Pvec[z]
                for j in range(Mnodes_old):
                    ARK[z+rowShift,j+columnShift]=Qmat[z,j]
            
            columnShift = columnShift + Mnodes_old

        for k in range(self.n_subNodess[-1]):
            bRK[NRK-self.n_subNodess[-1]+k] = bADER[k]

        self.NRK = NRK
        self.ARK = ARK
        self.bRK = bRK
        self.cRK = cRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARK[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARK[:,0] = self.ARK[:,0] + self.ARK[:,i]
                self.bRK[0] = self.bRK[0] + self.bRK[i]
                # delete i-th row and column
                self.ARK = np.delete(self.ARK, i,0)
                self.ARK = np.delete(self.ARK, i,1)
                self.cRK = np.delete(self.cRK,i,0)
                self.bRK = np.delete(self.bRK,i,0)
            else:
                i=i+1
        return self.ARK,self.bRK,self.cRK

    def ader(self, func, tspan, y_0):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        if self.nodes_type=="orthogonal":
            return self.ader_orthogonal(func,tspan, y_0)
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_a=np.zeros((dim, M_nodes))
        rhs= np.zeros((dim,M_nodes))

        U[:,0]=y_0


        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])

            M_nodes = self.n_subNodess[0]

            # Initialize strucutres
            for m in range(M_nodes):
                u_a[:,m]=U[:,it-1]
            # Compute the flux of the initial state and copy to other stages
            rhs0 = func(u_a[:,0],tspan[it-1])
            for r in range(M_nodes):
                rhs[:,r]=rhs0
            
            # Iterative procedure
            for k in range(1,K_corr):
                # update value of u
                M_nodes_old = M_nodes
                M_nodes = self.n_subNodess[k]
                for d in range(dim):
                    u_a[d,:M_nodes] = U[d,it-1] + delta_t*np.matmul(self.evolMats[k],rhs[d,:M_nodes_old])
                
                t_sub = tspan[it-1]+ delta_t*self.x_polys[k]
                    
                # update value of rhs
                if self.skip_first_rhs[k]:
                    rhs[:,0]=rhs0
                    for r in range(1,M_nodes):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                else:
                    for r in range(M_nodes):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
            # final update 
            for d in range(dim):
                U[d,it]=U[d,it-1] + delta_t*np.dot(bADER, rhs[d,:])
        return tspan, U


    def ader_orthogonal(self, func, tspan, y_0):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_a=np.zeros((dim, M_nodes))
        rhs= np.zeros((dim,M_nodes))

        N_quad = len(self.w_quad)
        Fu_p_quad = np.zeros((dim, N_quad))
        tmp = np.zeros(M_nodes)

        U[:,0]=y_0


        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])

            M_nodes = self.n_subNodess[0]

            # Initialize strucutres
            u_a[:,:] = 0. 
            u_a[:,0] = U[:,it-1]

            # Compute the flux of the initial state and copy to other stages
            rhs0 = func(u_a[:,0],tspan[it-1])
            
            # Iterative procedure
            for k in range(1,K_corr+1):
                k_eff = min(k,K_corr-1)
                # update value of u
                M_nodes_old = M_nodes
                M_nodes = self.n_subNodess[k_eff]
                nodes_quad = self.nodes_quads[k_eff]
                w_quad = self.w_quads[k_eff]
                N_quad = len(nodes_quad)
                invMass = self.invMasss[k_eff]
                basis_quad = self.basis_quads[k_eff]

                # Evaluating F( sum phi_j(x_q) * u^j)
                u_rec = u_a[:,:M_nodes_old] @ (basis_quad[:N_quad,:M_nodes_old].transpose())
                for jq, xq in enumerate(nodes_quad):
                    Fu_p_quad[:,jq] = func(u_rec[:,jq], tspan[it-1]+delta_t*xq)

                for d in range(dim):
                    # Computing the integral int_0^1 phi_j F(u) dt
                    for j in range(M_nodes):
                        tmp[j] = sum([ Fu_p_quad[d,jq]*self.basis_quad[jq,j]*w_quad[jq] for jq in range(N_quad) ])
                
                    # updating ua as ua=M^{-1}*1*u_n + d *M^{-1} RHS(u) 
                    u_a[d,:M_nodes] = self.evol_uns[k_eff].flatten()*U[d,it-1] + delta_t*np.matmul(invMass,tmp[:M_nodes]).flatten()
   

            # final update 
            for d in range(dim):
                U[d,it]= u_a[d,:]@self.recon_ADER

        return tspan, U

    def ader_order_control(self, func, tspan, y_0, err_tol):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        p_times=np.zeros(N_time,dtype=np.int32)
        dim=len(y_0)
        U=np.zeros((dim, N_time))

        u_a=np.zeros((dim, M_nodes))
        rhs= np.zeros((dim,M_nodes))
        u_a_recon = np.zeros(dim)
        u_p_recon = np.zeros(dim)

        U[:,0]=y_0
        
        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])

            M_nodes = self.n_subNodess[0]

            # Initialize strucutres
            for m in range(M_nodes):
                u_a[:,m]=U[:,it-1]

            t_sub = tspan[it-1]*np.ones(M_nodes)
            
            u_a_recon[:] = U[:,it-1]

            rhs0 = func(u_a_recon,tspan[it-1])

            # Iterative procedure
            err_local = err_tol +1.0
            k=0
            while (k < K_corr-1 and err_local>err_tol):
                k=k+1
                # update value of u
                M_nodes_old = M_nodes
                M_nodes = self.n_subNodess[k]
      
                u_p_recon[:] = u_a_recon[:]

                # update value of rhs
                if k==1:
                    for r in range(M_nodes_old):
                        rhs[:,r]=rhs0
                elif self.skip_first_rhs[k-1]:
                    rhs[:,0]=rhs0
                    for r in range(1,M_nodes_old):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                else:
                    for r in range(M_nodes_old):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])


                for d in range(dim):
                    u_a[d,:M_nodes] = U[d,it-1] + delta_t*np.matmul(self.evolMats[k],rhs[d,:M_nodes_old])
                
                t_sub = tspan[it-1]+ delta_t*self.x_polys[k]

                u_a_recon[:] = np.matmul(u_a[:,:M_nodes],self.recon_ADERs[k]).flatten()
                err_local = np.linalg.norm(u_a_recon-u_p_recon) /np.linalg.norm(u_a_recon)

            p_times[it] = k
            U[:,it]=u_a_recon
        return tspan, U, p_times



class ADER_u:
    """ ADER class with efficient procedure with u interpolation between different iterations """
    def __init__(self, M_sub, n_iter, nodes_type):
        """ 
        ADER class with input:
         * M_sub number of subtimesteps (i.e. M_sub +1 nodes), if M_sub=-1 then optimal one
         * n_iter number of iterations and expected order of accuracy
         * nodes_type among "equispaced" "gaussLobatto" "gaussLegendre"
         """
        M_sub = check_M(M_sub, n_iter, nodes_type)
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter        
        self.nodes_type = nodes_type

        self.set_increasing_orders()

        self.compute_ADER_matrices()
        self.name = f"ADER_u_{self.nodes_type}"  

    def set_increasing_orders(self):
        #self.M_subs           = np.zeros(self.n_iter,dtype=np.int32)
        # the last number of subtimenodes is equal to the second last number of subtimenodes
        # hence I do not include this value in the follwing arrays
        self.n_subNodess      = np.zeros(self.n_iter,dtype=np.int32)
        self.n_subNodess_star = np.zeros(self.n_iter,dtype=np.int32)

        for i in range(self.n_iter):
            if i <1:
                if self.nodes_type in ["gaussLobatto","equispaced","equispaced_lumped"]:
                    self.n_subNodess[i] = 2
                    self.n_subNodess_star[i] = 2
                elif self.nodes_type in ["gaussLegendre"]:
                    self.n_subNodess[i] = 2
                    self.n_subNodess_star[i] = 2
            else:
                if self.nodes_type in ["gaussLobatto","equispaced","equispaced_lumped"]:
                    self.n_subNodess[i] = min(i+1,self.n_subNodes)
                    self.n_subNodess_star[i] = min(i+2,self.n_subNodes)
                elif self.nodes_type in ["gaussLegendre"]:
                    self.n_subNodess[i] = min(i+1,self.n_subNodes)
                    self.n_subNodess_star[i] = min(i+2,self.n_subNodes)

    def compute_ADER_matrices(self):
        """
        Compute ADER matrices for all basis functions up to the final order 
        ADER = phi_i(1)phi_j(1) -int_0^1 d(phi_i)/dt phi_j dt 
        RHS  = int_0^1 phi_i phi_j dt
        """
        # iteration zero
        self.x_polys=[np.empty(0)]
        self.evolMats=[np.empty(0)]
        self.interpMatrices=[np.empty(0)]
        self.evolMats_no_interp=[np.empty(0)]
        self.recon_ADERs=[np.empty(0)]
        
        self.skip_first_rhs = np.zeros(self.n_iter,dtype=bool)

        # iter 1 ... n_iter-1
        for i in range(1,self.n_iter):
            n_subNodes_old = self.n_subNodess[i]     
            n_subNodes_new = self.n_subNodess_star[i]
            
            # For ader scheme
            _, _, _, _, _, evolMat_no_int, _ ,\
                nodes_quad, w_quad, basis_quad, evol_un= getADER_matrix(n_subNodes_old, self.nodes_type)
            self.evolMats_no_interp.append(evolMat_no_int)
            
            x_poly,_,_,_,_,_, recon_ADER,\
                nodes_quad, w_quad, basis_quad, evol_un = getADER_matrix(n_subNodes_new, self.nodes_type)            
            self.x_polys.append(x_poly)
            self.recon_ADERs.append(recon_ADER)

            int_mat = getADER_interpolation_matrix(n_subNodes_old,n_subNodes_new, self.nodes_type)
            self.interpMatrices.append(int_mat)
                        
            # For RK
            evolMat = getADER_evol_matr_u_interp(n_subNodes_old,n_subNodes_new, self.nodes_type)
            self.evolMats.append(evolMat)

            if sum(abs(self.evolMats_no_interp[i][0,:]))<1e-15 or (sum(self.evolMats_no_interp[i][0,:])<1e-15 and i==1):
                self.skip_first_rhs[i] = True
            else:
                self.skip_first_rhs[i] = False

        n_subNodes_old = self.n_subNodess_star[self.n_iter-1]
        x_poly, _, _, _, self.bADER, evolMat, self.recon_ADER,\
                nodes_quad, w_quad, basis_quad, evol_un = \
            getADER_matrix(n_subNodes_old, self.nodes_type)
        self.x_polys.append(x_poly)
        self.evolMats.append(evolMat)
        self.evolMats_no_interp.append(evolMat)
        self.interpMatrices.append(np.eye(n_subNodes_old))
        self.recon_ADERs.append(self.recon_ADER)
        
        #return self.ADERmat, self.RHSmat, self.bADER, self.evolMat
    
    def compute_RK(self):
        "Compute the RK  matrices of the ADER scheme"
        Mnodes = self.n_subNodes
        K_corr = self.n_iter
        #Could be optimized in GLB and equi (extra first step)
        NRK = sum(self.n_subNodess_star)
        ARK = np.zeros((NRK,NRK))
        bRK = np.zeros((NRK,1))
        cRK = np.zeros((NRK,1))
        
        bADER = self.bADER

        rowShift = 0
        columnShift = 0
        Mnodes_new = self.n_subNodess_star[0]


        for k in range(1,K_corr):
            Mnodes_old = self.n_subNodess_star[k-1]
            Mnodes_new = self.n_subNodess_star[k]
            rowShift = rowShift + Mnodes_old
            Qmat = self.evolMats[k]
            Pvec=Qmat@np.ones((Qmat.shape[1],1))
            for z in range(Mnodes_new):
                cRK[z+rowShift]=Pvec[z]
                for j in range(Mnodes_old):
                    ARK[z+rowShift,j+columnShift]=Qmat[z,j]
            
            columnShift = columnShift + Mnodes_old

        for k in range(self.n_subNodess_star[-1]):
            bRK[NRK-self.n_subNodess_star[-1]+k] = bADER[k]

        self.NRK = NRK
        self.ARK = ARK
        self.bRK = bRK
        self.cRK = cRK

        # removing "0" lines
        i=1
        while (i<self.NRK):
            if sum(abs(self.ARK[i,:]))<1e-17:
                self.NRK=self.NRK-1
                self.ARK[:,0] = self.ARK[:,0] + self.ARK[:,i]
                self.bRK[0] = self.bRK[0] + self.bRK[i]
                # delete i-th row and column
                self.ARK = np.delete(self.ARK, i,0)
                self.ARK = np.delete(self.ARK, i,1)
                self.cRK = np.delete(self.cRK,i,0)
                self.bRK = np.delete(self.bRK,i,0)
            else:
                i=i+1
        return self.ARK,self.bRK,self.cRK

    def ader(self, func, tspan, y_0):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))

        u_ps = np.zeros((dim, M_nodes))
        u_a=np.zeros((dim, M_nodes))
        rhs= np.zeros((dim,M_nodes))

        U[:,0]=y_0


        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])
            
            M_nodes = self.n_subNodess[0]

            # Initialize strucutres
            for m in range(M_nodes):
                u_a[:,m]=U[:,it-1]
            
            # k=1
            k=0
            
            M_nodes_old = self.n_subNodess[k]
            M_nodes_star = self.n_subNodess_star[k]

            rhs0 = func(u_a[:,0],tspan[it-1])
            for r in range(M_nodes_star):
                rhs[:,r]=rhs0

            # Iterative procedure
            for k in range(1,K_corr):

                M_nodes_old = self.n_subNodess[k]
                M_nodes_star = self.n_subNodess_star[k]

                # update value of u
                for d in range(dim):
                    u_a[d,:M_nodes_star] = U[d,it-1] + delta_t*np.matmul(self.evolMats[k],rhs[d,:M_nodes_old])


                # subtimenodes on new
                t_sub =  tspan[it-1]+self.x_polys[k]*delta_t

                if self.skip_first_rhs[k]:
                    rhs[:,0]=rhs0
                    for r in range(1,M_nodes_star):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                else:
                    for r in range(M_nodes_star):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                

            # final update 
            for d in range(dim):
                U[d,it]=U[d,it-1] + delta_t*np.dot(bADER, rhs[d,:M_nodes_star])
        return tspan, U


    def ader_order_control(self, func, tspan, y_0, err_tol):
        "Apply the ADER scheme to the ODE du/dt=func(u,t) with u0 =y_0 and timesteps tspan "
        M_nodes = self.n_subNodes
        K_corr = self.n_iter
        bADER=self.bADER.flatten()
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        p_times=np.zeros(N_time,dtype=np.int32)

        u_a=np.zeros((dim, M_nodes))
        rhs= np.zeros((dim,M_nodes))
        u_a_recon = np.zeros(dim)
        u_p_recon = np.zeros(dim)

        U[:,0]=y_0
        
        for it in range(1, N_time):
            # Time variables
            delta_t=(tspan[it]-tspan[it-1])
            
            M_nodes = self.n_subNodess_star[1]

            # Initialize strucutres
            for m in range(M_nodes):
                u_a[:,m]=U[:,it-1]


            t_sub = tspan[it-1]*np.ones(M_nodes)
            
            u_a_recon[:] = U[:,it-1]

            rhs0 = func(u_a_recon, tspan[it-1])

            # Iterative procedure
            err_local = err_tol +1.0
            k=0
        
            while (k < K_corr-2 and err_local>err_tol):
                k=k+1
                M_nodes_old = self.n_subNodess_star[k]
                M_nodes_star = self.n_subNodess_star[k+1]
      
                u_p_recon[:] = u_a_recon[:]

                # subtimenodes on new
                t_sub =  tspan[it-1]+self.x_polys[k]*delta_t

                if self.skip_first_rhs[k]:
                    rhs[:,0]=rhs0
                    for r in range(1,M_nodes_old):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                else:
                    for r in range(M_nodes_old):
                        rhs[:,r]=func(u_a[:,r],t_sub[r])
                
                # update value of u
                for d in range(dim):
                    u_a[d,:M_nodes_star] = U[d,it-1] + delta_t*np.matmul(self.evolMats[k+1],rhs[d,:M_nodes_old])

                u_a_recon[:] = np.matmul(u_a[:,:M_nodes_star],self.recon_ADERs[k+1]).flatten()
                err_local = np.linalg.norm(u_a_recon-u_p_recon) /np.linalg.norm(u_a_recon)

            p_times[it] = k
            U[:,it]=u_a_recon

        return tspan, U, p_times



def get_nodes(order,nodes_type, mp_flag=False):
    if "equispaced" in nodes_type:
        nodes,w = equispaced(int(order))
    elif nodes_type == "gaussLegendre":
        nodes,w = legendre(int(order))
    elif nodes_type == "gaussLobatto":
        nodes, w = lobatto(int(order))
    nodes=nodes*mp.mpf('0.5')+mp.mpf('0.5')
    w = w*mp.mpf('0.5')
    if mp_flag==False:
        nodes = convert_mp_2_np(nodes)
        w = convert_mp_2_np(w)
    return nodes, w

def getADER_matrix(order, nodes_type, mp_flag=False):
    """
    input M, nodes_type, mp_flag (opt), compute ADER matrices
    for M+1 nodes of type nodes_type, in multiple_precision if mp_flag==True
    returns nodes_poly, w_poly, M, RHSmat, bADER, evolMat
    nodes of lagr polynomials, their weights for a quadrature,
    ADER matrix M to be inverted (phi_i(1)phi_j(1)-int_0^1 \dphi_i/dt phi_j)
    RHS matrix int_0^1 phi_i phi_j
    evolMat = ADER^-1*RHS
    """

    if nodes_type=="orthogonal":
        basis = orthogonal_legendre_polynomials
        deriv_basis = derivative_orthogonal_legendre_polynomials
    else:
        nodes_poly, w_poly = get_nodes(order,nodes_type,mp_flag=True)
        basis = lambda x, k : lagrange_basis(nodes_poly,x,k)
        deriv_basis = lambda x, k : lagrange_deriv(nodes_poly,x,k)
    
    if nodes_type=="equispaced" or nodes_type=="orthogonal":
        quad_order=order
        nodes_quad, w = get_nodes(quad_order,"gaussLegendre",mp_flag=True)
    else:
        quad_order=order
        nodes_quad, w = get_nodes(quad_order,nodes_type,mp_flag=True)
                    
    # generate mass matrix
    M = mp.zeros(int(order),int(order))
    for i in range(order):
        for j in range(order):
            M[i,j] = basis(mp.mpf(1),i) *basis(mp.mpf(1),j)\
                  -sum([deriv_basis(nodes_quad[q],i)\
                  *basis(nodes_quad[q],j)\
                  *w[q] for q in range(quad_order)])
    # generate mass matrix
    RHSmat = mp.zeros(int(order),int(order))
    for i in range(order):
        for j in range(order):
            for q in range(quad_order):
                RHSmat[i,j] = RHSmat[i,j] +w[q]*\
                    basis(nodes_quad[q],i)*\
                    basis(nodes_quad[q],j)
    # generate RHS matrix
    bADER = mp.zeros(int(order),1)
    if nodes_type=="orthogonal":
        # first basis function is constantly 1 and orthogonal to others
        bADER[0,0]= mp.mpf('1.0')
    else:
        # sum on basis functions is constantly 1
        for i in range(order):
            bADER[i,0] = sum([basis(nodes_quad[q],i)*w[q] \
                for q in range(quad_order)])

    # generate Reconstruction coefficients matrix
    recon_ADER = mp.zeros(int(order),1)
    for i in range(order):
        recon_ADER[i,0] = basis(1.,i)

    # generate reconstruction coefficients for u(0)
    recon_0 = mp.zeros(int(order),1)
    for i in range(order):
        recon_0[i,0] = basis(0.,i)


    M1=mp.inverse(M)
    evolMat=M1@RHSmat
    evol_un = M1@recon_0


    N_quad = int(len(w))
    basis_quad = mp.zeros(N_quad, int(order))
    for j in range(order):
        for jq in range(N_quad):
            basis_quad[jq,j]= basis(nodes_quad[jq],j) 

    if nodes_type=="orthogonal":
        nodes_poly = evolMat@mp.ones(int(order),1)
        w_poly = mp.ones(int(order),1)

    if mp_flag==False:
        nodes_poly = convert_mp_2_np(nodes_poly)
        w_poly = convert_mp_2_np(w_poly)
        M = convert_mp_2_np(M)
        RHSmat = convert_mp_2_np(RHSmat)
        bADER = convert_mp_2_np(bADER)
        evolMat = convert_mp_2_np(evolMat)
        recon_ADER = convert_mp_2_np(recon_ADER)
        evol_un = convert_mp_2_np(evol_un)
        nodes_quad = convert_mp_2_np(nodes_quad)
        w = convert_mp_2_np(w)
        basis_quad = convert_mp_2_np(basis_quad)

    return nodes_poly, w_poly, M, RHSmat, bADER, evolMat, recon_ADER, nodes_quad, w, basis_quad, evol_un


### Efficient ADER

def getADER_RHS_L2_proj(order_old, order_new, nodes_type, mp_flag=False):
    if nodes_type=="orthogonal":
        basis_old=orthogonal_legendre_polynomials
        basis_new=orthogonal_legendre_polynomials
    else:
        nodes_poly_old, w_poly_old = get_nodes(int(order_old),nodes_type, mp_flag=True)
        nodes_poly_new, w_poly_new = get_nodes(int(order_new),nodes_type, mp_flag=True)
        basis_old = lambda x,k:lagrange_basis(nodes_poly_old,x,k)
        basis_new = lambda x,k:lagrange_basis(nodes_poly_new,x,k)
    
    if nodes_type=="equispaced" or nodes_type=="orthogonal":
        quad_order=order_new
        nodes_quad, w = get_nodes(int(quad_order),"gaussLegendre", mp_flag=True)
    else:
        quad_order=order_new
        nodes_quad, w = get_nodes(int(quad_order),nodes_type, mp_flag=True)
                    
    # generate  RHS matrix RHS_ij =int phi_i^new phi_j^old
    RHSmat = mp.zeros(int(order_new),int(order_old))
    for i in range(order_new):
        for j in range(order_old):
            for q in range(quad_order):
                RHSmat[i,j] = RHSmat[i,j] +w[q]*\
                    basis_new(nodes_quad[q],i)*\
                    basis_old(nodes_quad[q],j)

    if mp_flag==False:
        RHSmat = convert_mp_2_np(RHSmat)

    return RHSmat

def getADER_evol_matr_L2_proj(order_old, order_new, nodes_type, mp_flag=False):
    RHSmat = getADER_RHS_L2_proj(order_old, order_new, nodes_type, mp_flag=True)
    _,_,ADERMat ,_,_,_ ,_ ,\
                nodes_quad, w_quad, basis_quad, evol_un= getADER_matrix(int(order_new), nodes_type, mp_flag=True)
    evolMat = mp.inverse(ADERMat)@RHSmat
    if mp_flag==False:
        evolMat=convert_mp_2_np(evolMat)
    return evolMat

def getADER_interpolation_matrix(order_old,order_new,nodes_type, mp_flag=False):
    if nodes_type=="orthogonal":
        raise ValueError("Cannot do interpolation with orthogonal polynomials")
    nodes_poly_old, w_poly_old = get_nodes(int(order_old),nodes_type, mp_flag=True)
    nodes_poly_new, w_poly_new = get_nodes(int(order_new),nodes_type, mp_flag=True)

    # generate  RHS matrix RHS_ij =int phi_i^new phi_j^old
    H = mp.zeros(int(order_new),int(order_old))
    for i in range(order_new):
        for j in range(order_old):
            H[i,j] = lagrange_basis(nodes_poly_old,nodes_poly_new[i],j)

    if mp_flag==False:
        H = convert_mp_2_np(H)
    return H


def getADER_evol_matr_u_interp(order_old, order_new, nodes_type, mp_flag=False):
    _,_,_ ,_,_,evolMatold,_,\
                nodes_quad, w_quad, basis_quad, evol_un = getADER_matrix(int(order_old), nodes_type, mp_flag=True)
    H = getADER_interpolation_matrix(int(order_old), int(order_new), nodes_type, mp_flag=True)
    evolMat = H@evolMatold
    if mp_flag==False:
        evolMat=convert_mp_2_np(evolMat)
    return evolMat


def check_M(M_sub, order, nodes_type):
    if M_sub==-1:
        if "equispaced" in nodes_type:
            M_sub = order-1
        elif nodes_type=="gaussLobatto":
            M_sub = int(np.ceil(order/2))
        elif nodes_type=="gaussLegendre":
            M_sub = int(np.ceil((order-1)/2))
        else:
            M_sub = order-1
    
    if M_sub==0 and (nodes_type == "gaussLobatto" or "equispaced" in nodes_type):
        print(f"M_steps=0 is not enough to run with {nodes_type}, I run with M_steps=1")
        M_sub = 1
    return M_sub