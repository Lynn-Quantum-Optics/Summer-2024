import numpy as np
from matplotlib import pyplot as plt
from qo_tools import bloch_coords, expectation_value, ID, SX, SY, SZ

def scatter3D(points, s=0.8, alpha=0.5, color='b', title='my plot', figsize=(12,12)):
    # setup the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    # scatter the points
    ax.scatter(points[:,0], points[:,1], points[:,2], s=s, alpha=alpha, color=color)
    # label axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # set the initial view
    ax.view_init(elev=30., azim=45)
    # set the window title
    fig.canvas.manager.set_window_title(title)

def plot_on_bloch_sphere(kets, s=0.8, alpha=0.5, figsize=(12,12), title='my plot'):
    # get the coordinates
    coords = np.array([bloch_coords(ket) for ket in kets])
    # setup the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    # scatter the points
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=s, alpha=alpha)
    # label axes
    ax.set_xlabel(r'$\langle \sigma_x \rangle$')
    ax.set_ylabel(r'$\langle \sigma_y \rangle$')
    ax.set_zlabel(r'$\langle \sigma_z \rangle$')
    # set the initial view
    ax.view_init(elev=30., azim=45)
    # title the figure
    ax.set_title(title)

def pauli_histograms_1qubit(kets, bins=30, figsize=(8,12), title='my plot'):
    # setup the figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    # make histograms
    for i, (x, M) in enumerate(zip('xyz', [SX, SY, SZ])):
        ax = fig.add_subplot(3,1,i+1)
        values = [expectation_value(ket, M) for ket in kets]
        ax.hist(values, bins=bins)
        ax.set_ylabel('Frequency')
        ax.set_xlabel(f'$\\langle \\sigma_{x} \\rangle$')
    # fix overlap
    fig.tight_layout()

def pauli_histograms_2qubit(kets, bins=30, figsize=(12,12), title='my plot'):
    basis = []
    labels = []
    for al, a in zip('ixyz', [ID, SX, SY, SZ]):
        for bl, b in zip('ixyz', [ID, SX, SY, SZ]):
            if al+bl == 'ii': continue
            basis.append(np.kron(a,b))
            labels.append(r'$\langle \sigma_{%s%s} \rangle$' % (al, bl))

    # setup the figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Distribution of Bloch Coordinates')
    # make histograms
    for i, (l, M) in enumerate(zip(labels, basis)):
        ax = fig.add_subplot(8,2,i+1)
        values = [expectation_value(ket, M) for ket in kets]
        ax.hist(values, bins=bins)
        ax.set_ylabel('Frequency')
        ax.set_xlabel(l)
    # set the window title
    fig.canvas.manager.set_window_title(title)
    fig.tight_layout()