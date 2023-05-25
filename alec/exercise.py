import numpy as np
import matplotlib.pyplot as plt


phi_plus = np.array([1,0,0,1])/np.sqrt(2)
phi_minus = np.array([1,0,0,-1])/np.sqrt(2)
psi_plus = np.array([0,1,1,0])/np.sqrt(2)
psi_minus = np.array([0,1,-1,0])/np.sqrt(2)

def get_correlations(state, theta, phi):
    # define the basis vectors
    e1 = np.array([
        [np.cos(theta)],
        [np.exp(1j*phi)*np.sin(theta)]])
    e2 = np.array([
        [np.sin(theta)],
        [-np.exp(1j*phi)*np.cos(theta)]])

    # define the two-qubit basis
    e1e1 = np.kron(e1,e1)
    # e1e2 = np.kron(e1,e2)
    # e2e1 = np.kron(e2,e1)
    e2e2 = np.kron(e2,e2)

    # obtain the coefficients in the new basis
    return np.abs(np.vdot(e1e1, state))**2 + np.abs(np.vdot(e2e2, state))**2

def get_correlation_grid(state, thetas, phis):
    correlations = np.zeros((len(phis), len(thetas)))
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            correlations[j,i] = get_correlations(state, theta, phi)
    return correlations

thetas = np.linspace(0, np.pi/2, 50)
phis = np.linspace(0, np.pi, 100)
theta_grid, phi_grid = np.meshgrid(thetas, phis)

fig = plt.figure()

kwargs = {'edgecolor':'royalblue', 'lw':0.2, 'rstride':8, 'cstride':8, 'alpha':0.3}
ax = fig.add_subplot(221, projection='3d')
ax.set_title('$\Phi^+$')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('$\\phi$')
ax.plot_surface(theta_grid, phi_grid, get_correlation_grid(phi_plus, thetas, phis), **kwargs, label='$\\Phi^+$')

ax = fig.add_subplot(222, projection='3d')
ax.set_title('$\Phi^-$')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('$\\phi$')
ax.plot_surface(theta_grid, phi_grid, get_correlation_grid(phi_minus, thetas, phis), **kwargs, label='$\\Phi^-$')

ax = fig.add_subplot(223, projection='3d')
ax.set_title('$\Psi^+$')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('$\\phi$')
ax.plot_surface(theta_grid, phi_grid, get_correlation_grid(psi_plus, thetas, phis), **kwargs, label='$\\Psi^+$')

ax = fig.add_subplot(224, projection='3d')
ax.set_title('$\Psi^-$')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('$\\phi$')
ax.plot_surface(theta_grid, phi_grid, get_correlation_grid(psi_minus, thetas, phis), **kwargs, label='$\\Psi^-$')

# ax.legend()
plt.show()