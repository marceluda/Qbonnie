#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Toolkit para resolver ecuaciones de matriz de densidad con modelo Liville+Limbland

La aproximación al problema está basada en la de estas publicaciones de referencia:


    Warren, Z., Shahriar, M. S., Tripathi, R., & Pati, G. S. (2017). 
    Experimental and theoretical comparison of different optical excitation 
    schemes for a compact coherent population trapping Rb vapor clock. 
    Metrologia, 54(4), 418–431. 
    https://doi.org/10.1088/1681-7575/aa72bb

    Downes, L. (2023). 
    Simple Python tools for modelling few-level atom-light interactions.
    Journal of Physics B: Atomic, Molecular and Optical Physics (Vol. 56, Issue 22, p. 223001). IOP Publishing. 
    https://doi.org/10.1088/1361-6455/acee3a
    https://github.com/LucyDownes/OBE_Python_Tools

"""


import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from time import time
import copy


from sympy import MatrixSymbol, Matrix, symbols, linear_eq_to_matrix, simplify
from sympy.parsing.sympy_parser import parse_expr
import sympy  as sy

from sympy import lambdify

# import numba
# import cupy as cp
# import scipy


#%% Herramientas útiles

class tools:
    """
    Collection of tools for building an visualizing your matrix system
    """
    def view_matrix(M,logscale=False):
        """
        Shows a fast image of a matrix 
        """
        is_complex = isinstance(M[0][0],complex)

        if is_complex:
            fig, axx = plt.subplots(1,3 , figsize=(13,7) )
        else:
            fig, ax  = plt.subplots(1,1 , figsize=(13,7) )

        mat = abs(M)
        min_val = mat[mat>0].min() if logscale else mat.min()
        
        margen  = np.array([0,M.shape[1],M.shape[0],0])+0.5
        norm    = LogNorm(vmin=min_val ) if logscale else None

        cmap = plt.get_cmap()
        bad = list(cmap(0))
        bad[3] = 0.8
        cmap.set_bad(bad)

        if is_complex:
            ax = axx[0]
        im = ax.imshow( mat, extent=margen, norm=norm , cmap=cmap)
        ax.set_title('amplitud')
        plt.colorbar(im)

        for ax in axx.flatten() if is_complex else [ax]:
            if M.shape[0]<=16:
                ax.set_xticks( np.arange(1,M.shape[1]+1) )
                ax.set_yticks( np.arange(1,M.shape[0]+1) )
            ax.set_xlim( 0.5,  M.shape[1]+0.5 )
            ax.set_ylim(  M.shape[0]+0.5 , 0.5 )
            ax.xaxis.set_ticks_position('top')
        #ax.invert_yaxis()
        
        if is_complex:
            im = [im]
            for mm,ax,name in zip([M.real,M.imag],axx[1:],'real imag'.split()):
                max_val = abs(M.real).max()
                imm = ax.imshow( mm, extent=margen, cmap='bwr', vmin=-max_val,vmax=max_val) 
                ax.set_title(name)
                plt.colorbar(imm)
                im.append(im)
            ax = axx
        return ax , im

    def trace(M):
        """
        Calcs the Trace of a matrix
        """
        return np.sum( np.diag( M ))



def dynamic_global_variable(nombre, valor, env=globals()):
        env[nombre] = valor

        try:
            get_ipython().user_global_ns[nombre] = valor
        except Exception as e:
            print()

            

class Matrix_sym(Matrix):
    def __new__(cls, *args):
        # Si se pasa una matriz de NumPy, conviértela a lista
        # if len(args)>0 and isinstance(args[0], np.ndarray):
        #     mat = mat.tolist()
        # Llama al constructor original de Matrix
        return super().__new__(cls, *args)
    
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.parent = None 
        self._dict_fast_eval_num = None 
        self._dict_fast_eval_sym = None
    
    def copy_from(self,M):
        parent = self.parent
        self.__dict__ = copy.deepcopy(M.__dict__)
        self.parent = parent

    def set_parent(self,parent):
        self.parent = parent 
    
    def prepare_fast_eval(self):
        """
        Creates dictionary of matrices for each linear variable of
        the model to make fast calculations
        """
        self._prepare_fast_eval_symbolic()
        self._prepare_fast_eval_numeric()
    

    def _prepare_fast_eval_symbolic(self):
        if self.parent is None:
            raise ValueError('parent is not defined')
        rta = {}

        simbolos_extraer = [ self.parent.parameters[name] for name in self.parent.variables ]

        # Armo matrices de cada simbolo a extraer
        for simbolo in simbolos_extraer:
            rta[simbolo.name] = Matrix(np.zeros(self.shape))
        rta['cte'] = Matrix(np.zeros(self.shape))
        
        for ii,M_elemnto in enumerate(self):
            factores, cte = linear_eq_to_matrix(M_elemnto,*simbolos_extraer)
            for simbolo,factor in zip(simbolos_extraer,factores):
                rta[simbolo.name][ii] = factor
            rta['cte'][ii] = -cte
        self._dict_fast_eval_sym = rta 
    
    def _prepare_fast_eval_numeric(self):
        if self.parent is None:
            raise ValueError('parent is not defined')
        if not isinstance(self._dict_fast_eval_sym,dict):
            raise ValueError('_dict_fast_eval_sym is not defined')
        
        substitutions = { self.parent.parameters[name]:value for name,value in self.parent.values.items() }
        self._dict_fast_eval_num = {}

        for key,val in self._dict_fast_eval_sym.items():
            self._dict_fast_eval_num[key] = np.array(val.subs(substitutions)).astype(complex)
        
    
    def eval(self,*args,**kargs):
        """
        Evaluates symbols with values
        """
        if not hasattr(self,'parent') or self.parent is None:
            print("There isn't pre-defined constants (not parent)")

            values = {}
            not_defined = []
            for symbol in self.free_symbols:
                try:
                    values[symbol] = kargs[symbol.name]
                except KeyError as e:
                    not_defined.append(symbol.name)
            if len(not_defined)>0:
                raise ValueError(f'ERROR: the following symbols should be defined:\n' + 
                                 '{'+','.join(not_defined)+'}')

            values = { symbol:kargs[symbol.name] for symbol in self.free_symbols }
            return np.array(self.subs(values)).astype(complex)
        
        try:
            self._dict_fast_eval_num.items
        except Exception as e:
            self.prepare_fast_eval()
        
        values = { key:val for key,val in zip(self.parent.variables,args) }
        values.update({ key:val for key,val in kargs.items() if key in self.parent.variables})
        
        if not len(values) == len(self.parent.variables):
            raise ValueError("Arguments and variables numbres don't match")
        
        values['cte'] = 1

        rta = np.zeros(self.shape).astype(complex)

        for key, mat in self._dict_fast_eval_num.items():
            rta += mat*values[key]
        return rta




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Build your system

class qbonnie():
    """
    Hamiltonian + Limbbland builder and solver
    """
    def __init__(self,N=3):
        """
        N is the problem dimension
        """
        # Stores the problem dimension
        self.N = N
        self.Dro = None 
        self.Ms = None
        self.parameters = {}
        self.values = {}
        self.variables = ()
        self.rho = self.create_rho()

        self._environment = self.__dict__

    def s(self,m,n,numpy=False):
        """
        Representa la matriz/operador:
            |m><n|
        """
        N = self.N
        if min(m,n)<0 or max(m,n)>N:
            raise ValueError(f'n,m shpuld be between 0 and {N}')
        rta = np.zeros((N,N)).astype(int)
        rta[m,n] = 1 

        if numpy:
            return rta
        else:
            rta = Matrix_sym(rta)
            rta.set_parent(self)
            return rta

    def define_environment(self,environment):
        self._environment = environment

    def new_parameters(self,names,*args, **kwargs):
        """
        Crea los símbolos necesraios para construir las
        ecuaciones
        """
        L = len(names.split())
        for name in names.split():
            if name in self.parameters:
                raise ValueError(f'Parameter {name} was already created')
        rta = sy.symbols(names,*args, **kwargs)

        for symbol in rta if L>1 else [rta]:
            self.parameters[symbol.name] = symbol 
            # globals()[symbol.name] = symbol
            dynamic_global_variable(symbol.name,symbol,env=self._environment)
        self._update_variable_parameters()
        return rta 
    
    def make_empty_matrix(self,N=None, name=None):
        if N is None:
            N = self.N 
        
        rta = Matrix_sym(np.zeros((N,N)))
        rta.set_parent(self)
        if isinstance(name,str):
            setattr(self,name,rta)
        return rta


    def create_rho(self):
        """
        Crea la matriz simbólica rho que representa la matriz densidad
        """
        N = self.N
        ro =  Matrix( np.zeros((N,N)).astype(int) )
    
        for ii in range(ro.shape[0]):
            for jj in range(ro.shape[1]):
                if ii==jj:
                    ro[ii,jj] = symbols(f'ro{ii}_{jj}', real=True)
                else:
                    ro[ii,jj] = symbols(f'ro{ii}_{jj}')
        
        self.rho = ro
        return ro
    
    def load_Drho(self,Drho):
        """
        Crea la matriz d(rho)/dt 
        """
        self.Drho = Matrix_sym()
        self.Drho.copy_from(Drho)
        self.Drho.set_parent(self)


    def make_symbolic_M(self):
        """
        Toma la lista de elementos de rho y la matriz Dro
        y arma l Matriz M que permite procesar la version vectorial de rho
        
        Devuelve:
            Ms (M en forma simbólica)
        """
        # t0 = time()
        
        N = self.N
        
        # Lista ordenada de elementos de matriz de ro
        # Corresponden a la base vectorial sobre la que se escribirán 
        # los vectores de cálculo
        lista_de_simbolos = []
        for ii in range(N):
            for jj in range(N):
                lista_de_simbolos += [ self.rho[ii,jj] ]
        
        # Vamos a vectorizar la matriz del Hamiltoniano
        # Cada elemento va a ser escrito como combinacion lineal los elementos de 
        # matriz del operador densidad.
        # Luego, lista_de_simbolos será la nueva base de descripción del problema algebráico
        
        
        Ms = Matrix_sym([[]])
        
        
        for ii in range(N):
            for jj in range(N):
                # print(f'M[{ii},{jj}]')
                
                # Definimos ecuación
                eqn = self.Drho[ii,jj]
                
                # Extraemos expresion de fila de matriz
                M, b = linear_eq_to_matrix(eqn, *lista_de_simbolos)
                
                # Apila las filas en Ms
                Ms = Ms.row_insert( Ms.shape[0], M )
        # del M
        # print(f'make_Ms, Time: {round(time()-t0,2)} seg')

        self.Ms = Ms
        self.Ms.set_parent(self)
        return self.Ms

    def set_constant(self,**kwargs):
        """
        Establecemos parámetros constantes del problema
        """
        for key,val in kwargs.items():
            if not key in self.parameters:
                raise ValueError(f"The constant {key} doesn't correspond to any parameter")
        for key,val in kwargs.items():
            if isinstance(val,(int,float)):
                self.values[key] = val 
            elif val is None:
                if key in self.values:
                    del self.values[key]
        self._update_variable_parameters()
        
    def _update_variable_parameters(self):
        """
        Actualiza las variables libres del problema
        """
        all_parameters_names = set(self.parameters.keys())
        constant_parameters_names = set(self.values.keys())
        self.variables = all_parameters_names - constant_parameters_names

    def _evol_numpy(self,t,rho_initial_vec,*args,**kargs):
        """
        Calcula la evolución temporal.
        Tenemos: dρ/dt = M·ρ
        Y      :     M = AA·D·AA^{-1}    con D diagonal
        Luego  :  ρ(t) = AA·exp(D t)·AA^{-1} · ρ_0
        
        _evol_numpy(tiempo,rho0,...):
            tiempo: tiempo
            roh0  : ρ_0 (en su forma vectorial)
            
            devuelve: ρ(t) (en forma vectorial) 
        """
        # t0 = time()
        M = self.Ms.eval(*args,**kargs)
        eigenvalues, eigenvectors  = np.linalg.eig(M)
        eigenvectors_inv           = np.linalg.inv(eigenvectors)

        rta = eigenvectors.dot(
                    np.eye(M.shape[0])*np.exp(eigenvalues*t)
                ).dot(eigenvectors_inv).dot(rho_initial_vec)
        # print(f'eval_M, Time: {round(time()-t0,2)} seg')
        
        # if fix_rho:
        #     rta = rta.reshape(N,N)
        #     rta = (rta+ rta.T.conjugate())/2
        return rta

    def evol(self,t,rho_initial,*args,fix_hermiticity=False,**kargs):
        rho_initial_vec = rho_initial.flatten()

        rta = self._evol_numpy(t,rho_initial_vec,*args,**kargs)
        rta = rta.reshape(self.N,self.N)

        if fix_hermiticity:
            rta += rta.conj().T
            rta /= 2
        return rta

if __name__ == '__main__':
    q = qbonnie(3)

    q.new_parameters('Delta delta Omega_A Omega_B Gamma')

    ro = q.create_rho()

    H  = q.s(0,0)*Delta - q.s(1,1)*Delta -2*q.s(2,2)*delta
    H += q.s(0,2)*Omega_A + q.s(1,2)*Omega_B
    H += q.s(2,0)*Omega_A + q.s(2,1)*Omega_B
    H /= 2

    S = q.s(2,2) * Gamma
    L = (q.s(0,0) + q.s(1,1))* Gamma/2 * ro[2,2]


    ###########################################################
    Dro = -(H*ro - ro*H)*parse_expr('1j') - ( S*ro + ro*S )/2 + L
    ###########################################################


    q.load_Drho(Dro)
    #prepare_fast_eval
    q.set_constant(Gamma=6e6, Omega_A=1e5, Omega_B=1e5)

    q.make_symbolic_M()
    q.Ms.prepare_fast_eval()
    tools.view_matrix(q.Ms.eval(0,1e3))
    plt.show()
