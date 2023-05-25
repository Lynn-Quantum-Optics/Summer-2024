import numpy as np



# set the parameters
phi = np.pi/2.77
theta = np.pi*3.2/8

e1 = np.array([np.cos(theta),np.exp(1j*phi)*np.sin(theta)]).reshape(2,1)
e2 = np.array([np.sin(theta),-np.exp(1j*phi)*np.cos(theta)]).reshape(2,1)

e1e1 = np.kron(e1,e1)
e1e2 = np.kron(e1,e2)
e2e1 = np.kron(e2,e1)
e2e2 = np.kron(e2,e2)

phi_plus = ((np.cos(theta)**2 + np.sin(theta)**2 * np.exp(-2j*phi))*e1e1 + \
    np.sin(theta)*np.cos(theta)*(1-np.exp(-2j*phi))*(e1e2+e2e1) + \
    (np.sin(theta)**2 + np.cos(theta)**2 * np.exp(-2j*phi))*e2e2)/np.sqrt(2)

phi_minus = ((np.cos(theta)**2-np.sin(theta)**2 * np.exp(-2j*phi))*e1e1 + \
    np.sin(theta)*np.cos(theta)*(1+np.exp(-2j*phi))*(e1e2+e2e1) + \
    (np.sin(theta)**2 - np.cos(theta)**2 * np.exp(-2j*phi))*e2e2)/np.sqrt(2)

psi_plus = np.exp(-1j*phi)*(np.sin(2*theta)*(e1e1 - e2e2) - np.cos(2*theta) * (e1e2+e2e1))/np.sqrt(2)

psi_minus = np.exp(-1j*phi)*(e2e1-e1e2)/np.sqrt(2)


