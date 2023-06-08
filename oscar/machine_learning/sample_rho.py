# file to hold sample density matrices
import numpy as np

## sample bell states##
PhiP_s = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
PhiP= PhiP_s @ np.conjugate(PhiP_s.reshape((1,4)))

PhiM_s = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).reshape((4,1))
PhiM= PhiM_s @ np.conjugate(PhiM_s.reshape((1,4)))

PsiP_s = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0]).reshape((4,1))
PsiP= PsiP_s @ np.conjugate(PsiP_s.reshape((1,4)))

PsiM_s = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0]).reshape((4,1))
PsiM= PsiM_s @ np.conjugate(PsiM_s.reshape((1,4)))