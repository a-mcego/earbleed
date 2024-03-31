import glob
import json
import lzma
import numpy as np
import scipy
import sys
import time
import zlib


def save_ebl(eblname, compressed_bytes, params):
    # Convert params dictionary to JSON string
    params_json = json.dumps(params)
    params_bytes = params_json.encode('utf-8')
    
    with open(eblname, 'wb') as file:
        file.write(b'EBLE')
        file.write(len(params_bytes).to_bytes(8, byteorder='little'))
        file.write(params_bytes)
        file.write(len(compressed_bytes).to_bytes(8, byteorder='little'))
        file.write(compressed_bytes)

def load_ebl(eblname):
    with open(eblname, 'rb') as file:
        fourcc = file.read(4)
        if fourcc != b'EBLE':
            print("Error: not an EarBleed file. Expected fourcc code 'EBLE'.", file=sys.stderr)
            sys.exit(1)
        params_length = int.from_bytes(file.read(8), byteorder='little')
        params_bytes = file.read(params_length)
        params = json.loads(params_bytes.decode('utf-8'))
        compressed_length = int.from_bytes(file.read(8), byteorder='little')
        compressed_bytes = file.read(compressed_length)
    
    return compressed_bytes, params

class LZMACompressor:
    def __init__(self):
        self.LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
    
    def compress(self, array):
        compressed_data = lzma.compress(array.tobytes(), format=lzma.FORMAT_RAW, filters=self.LZMA_FILTERS)
        return compressed_data

    def decompress(self, compressed_data, dtype):
        binary_data = lzma.decompress(compressed_data, format=lzma.FORMAT_RAW, filters=self.LZMA_FILTERS)
        array = np.frombuffer(binary_data, dtype=dtype)
        return array

class ZlibCompressor:
    def __init__(self):
        pass
        
    def compress(self, array):
        binary_data = array.tobytes()
        compressed_data = zlib.compress(binary_data, level=9, wbits=-15)
        return compressed_data

    def decompress(self, compressed_data, dtype):
        binary_data = zlib.decompress(compressed_data, wbits=-15)
        array = np.frombuffer(binary_data, dtype=dtype)
        return array

Compressor = LZMACompressor()

def read_wave(filename, window_size):
    samplerate,x = scipy.io.wavfile.read(filename) #get wave data
    print(f"Read wav file: {x.shape} at {samplerate} Hz")
    x = x.astype(np.float32)
    #x = np.average(x,axis=-1,keepdims=True) #uncomment this to convert everything to mono
    
    if x.shape[-1] == 2: #use joint stereo
        print("Using joint stereo")
        jointstereo = np.array([[1,1],[-1,1]])*0.5
        x = np.matmul(x, jointstereo)

    x = np.transpose(x)
    if x.shape[1]%window_size != 0:
        x = np.concatenate([x, np.zeros((x.shape[0],window_size-x.shape[0]%window_size))], axis=-1)
    
    
    x = x/32768.0
    return x,samplerate

def write_wave(filename, data, samplerate):
    if data.shape[-1] == 2: #use joint stereo
        print("Using joint stereo")
        jointstereo = np.array([[1,-1],[1,1]])
        data = np.matmul(data, jointstereo)
    scipy.io.wavfile.write(filename, samplerate, (np.clip(data,-1.0,1.0)*32767.0).astype(np.int16))

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
        #pad with zeros to get rid of edge artifacts
        x = np.concatenate([np.zeros((x.shape[0],self.N)), x, np.zeros((x.shape[0],self.N))], axis=-1)
        L = x.shape[-1]
        windows = [x[:,l*self.N:(l+2)*self.N] for l in range(0, L // self.N - 1)]
        out = np.matmul(windows,self.matrix)
        return out

    def backward(self, X):
        out = np.matmul(X, np.transpose(self.matrix))
        out = out[1:, :, :self.N] + out[:-1, :, self.N:]
        out = np.transpose(out, (0,2,1))
        out = np.reshape(out, (-1,out.shape[-1]))
        print(out.shape)
        return out

def compressDCT(X,multiplier):
    X = np.round(X*multiplier)
    print(np.min(X), np.max(X))
    compressedX = Compressor.compress(X.flatten().astype(np.int8))
    return compressedX
    
def decompressDCT(compressedX,params):
    decompressedX = Compressor.decompress(compressedX, np.int8)
    decompressedX = decompressedX.astype(np.float32) / params['m']
    decompressedX = np.reshape(decompressedX, (-1, params['c'], WINDOW_SIZE))
    return decompressedX


MULTIPLIER = 9.13
WINDOW_SIZE = 2048
#wavename = "Kapittel_1.wav"
#wavename = "rule001.wav"
wavename = "fmi_0_gt.wav"
x,samplerate = read_wave(wavename, WINDOW_SIZE)
channels = x.shape[0]
mdct = MDCT(WINDOW_SIZE)
X = mdct.forward(x)
print(X.shape, " X shape")
originalsize = X.shape[0]*X.shape[2]
X = compressDCT(X, MULTIPLIER)

compressedsize = len(X)
print(f"compressedsize: {compressedsize}")
seconds = originalsize/samplerate
print(f"seconds: {seconds}")
bitrate = compressedsize*8/1000/seconds
print(f"bitrate: {bitrate} kbps")

eblname = wavename[:-4]+".ebl"
params = {
    'w':WINDOW_SIZE,
    's':samplerate,
    'c':channels,
    'm':MULTIPLIER,
    'b':bitrate,
    'l':seconds
}
save_ebl(eblname, X, params)

X,params = load_ebl(eblname)
X = decompressDCT(X, params)
x_reconstructed = mdct.backward(X)

write_wave("out.wav", x_reconstructed, samplerate)
