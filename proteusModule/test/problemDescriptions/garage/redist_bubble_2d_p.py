from pyadh import *
from pyadh.default_p import *
from math import *
from bubble import *

coefficients = RedistanceLevelSet(epsFact=epsFact_redistance,nModelId=1,rdModelId=2)

#now define the Dirichlet boundary conditions

def getDBC(x):
    pass
    
dirichletConditions = {0:getDBC}

weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs}
#weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs2}
#weakDirichletConditions = None


initialConditions  = None

fluxBoundaryConditions = {0:'noFlow'}

advectiveFluxBoundaryConditions =  {}

diffusiveFluxBoundaryConditions = {0:{}}