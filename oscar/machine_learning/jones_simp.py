# file to simplify the expression for rho
from sympy import *
from sympy.physics.quantum import TensorProduct, Dagger

# create angles
theta1, theta2, alpha1, alpha2, phi = Symbol('theta1'), Symbol('theta2'), Symbol('alpha1'), Symbol('alpha2'), Symbol('phi')

## jones matrices ##
def R (alpha): return Matrix([[cos(alpha1), -sin(alpha)], [sin(alpha), cos(alpha)]])
def H (theta): return Matrix([[cos(2*theta), sin(2*theta)], [sin(2*theta), -cos(2*theta)]])
def Q(alpha): return R(alpha) * diag(exp(pi/4*I), exp(-pi/4*I)) * R(-alpha)

H1 = H(theta1)
H2 = H(theta2)
Q1 = Q(alpha1)
Q2 = Q(alpha2)
QP = diag(1, exp(phi*I))
PhiP = (Matrix([[1, 0, 0, 1]]).T)*1/sqrt(2)

P = TensorProduct(H1, H2*Q2*Q1*QP*H1)
print(P)
rho = P*Dagger(P)
print(rho)
