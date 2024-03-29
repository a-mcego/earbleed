import numpy as np
import time

#implementation of MDCT

def MDCTmatrix(N):
    nums = np.arange(2*N)+0.5
    ws = nums[:,None]/N
    ks = nums[:N]*np.pi
    window = (np.pi*0.5)*ws
    matrix = (ws+0.5)*ks
    return np.cos(matrix)*(np.sin(window)*np.sqrt(2.0/N))

class MDCT:
    def __init__(self, N):
        self.N = N
        self.matrix = MDCTmatrix(self.N)

    def forward(self, x):
        #TODO: do padding for non-multiple-of-window-size data!
        #      for now, it only works with data that is a multiple of the window size!
    
        #pad with zeros to get rid of edge artifacts
        x = np.concatenate([np.zeros(self.N), x, np.zeros(self.N)])
        L = len(x)
        
        windows = [x[l*self.N:(l+2)*self.N] for l in range(0, L // self.N - 1)]
        out = np.matmul(windows,self.matrix)
        return out

    def backward(self, X):
        out = np.matmul(X, np.transpose(self.matrix))
        out = out[1:, :self.N] + out[:-1, self.N:]
        return out.flatten()

WINDOW_SIZE = 8192
TOTALSIZE = 512
#TODO: load wave files or something?
x = np.random.rand(TOTALSIZE * WINDOW_SIZE)*2.0-1.0  # Example input signal

time1 = time.time()
mdct = MDCT(WINDOW_SIZE)
time2 = time.time()
X = mdct.forward(x)
time3 = time.time()
x_reconstructed = mdct.backward(X)
time4 = time.time()

print("Init: ", time2-time1)
print("Forward: ", time3-time2)
print("Backward: ", time4-time3)

print(np.allclose(x, x_reconstructed, atol=1e-2), end=" ")

