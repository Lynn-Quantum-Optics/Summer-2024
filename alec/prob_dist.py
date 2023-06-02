import numpy as np
import matplotlib.pyplot as plt

N = 300

cube_points = np.random.rand(N, 3)


sphere_points = np.random.rand(N, 3)
sphere_points2 = np.random.randn(N, 3)
hsphere_points = np.random.randn(N, 15)

for i, (r, theta, phi) in enumerate(sphere_points):
    sphere_points[i] = r*np.array([np.sin(2*np.pi*theta)*np.cos(2*np.pi*phi), np.sin(2*np.pi*theta)*np.sin(2*np.pi*phi), np.cos(2*np.pi*theta)])

for i in range(N):
    sphere_points2[i] /= np.linalg.norm(sphere_points2[i])
    sphere_points2[i] *= np.random.rand()

for i in range(N):
    hsphere_points[i] = np.random.rand()/np.linalg.norm(hsphere_points[i])


cube_dists = []
sphere_dists = []
sphere_dists2 = []
hsphere_dists = []

for i in range(N):
    for j in range(i):
        cube_dists.append(np.linalg.norm(cube_points[i] - cube_points[j]))
        sphere_dists.append(np.linalg.norm(sphere_points[i] - sphere_points[j]))
        sphere_dists2.append(np.linalg.norm(sphere_points2[i] - sphere_points2[j]))
        hsphere_dists.append(np.linalg.norm(hsphere_points[i] - hsphere_points[j]))


BINS = 70

fig = plt.figure()
ax = fig.add_subplot(221)
ax.hist(cube_dists, bins=BINS)
ax.set_title('cube')
ax.set_xlabel('pairwise distance')
ax.set_ylabel('Count')

ax = fig.add_subplot(222)
ax.hist(sphere_dists, bins=BINS)
ax.set_title('sphere')
ax.set_xlabel('pairwise distance')
ax.set_ylabel('Count')

ax = fig.add_subplot(223)
ax.hist(sphere_dists2, bins=BINS)
ax.set_title('sphere (gaussian generation)')
ax.set_xlabel('pairwise distance')
ax.set_ylabel('Count')

ax = fig.add_subplot(224)
ax.hist(hsphere_dists, bins=BINS)
ax.set_title('15-dimensional hyper-sphere')
ax.set_xlabel('pairwise distance')
ax.set_ylabel('Count')

fig.tight_layout()

plt.show()

        



