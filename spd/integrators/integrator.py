from typing import Callable,Tuple
import numpy as np

class Integrator:
    def __init__(
        self,
        m: int =  1,
        ndim: int = 1,

    ):
        self.m = m #Time  order
        self.ndim = ndim
        
    def integrate(self):
        pass

    def high_order_fluxes(self):
        pass

    def update(self):
        pass    
