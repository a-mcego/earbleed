#EarBleed audio format compressor

import glob
import iso226
import json
import lzma
import numpy as np
import scipy
import sys
import time
import zlib

def save_ebl(eblname, compressed_bytes, params):
    params_json = json.dumps(params)
    params_bytes = params_json.encode('utf-8')
    
    with open(eblname, 'wb') as file:
        file.write(b'EBLE')
        file.write(len(params_bytes).to_bytes(8, byteorder='little'))
        file.write(params_bytes)
        file.write(len(compressed_bytes).to_bytes(8, byteorder='little'))
        file.write(compressed_bytes)
        
    print(params)

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

def read_wave(filename, window_size):
    samplerate,x = scipy.io.wavfile.read(filename) #get wave data
    print(f"Read wav file: {x.shape} at {samplerate} Hz")
    x = x.astype(np.float32)
    #x = np.average(x,axis=-1,keepdims=True) #uncomment this to convert everything to mono
    
    if x.shape[-1] == 2: #use joint stereo
        jointstereo = np.array([[1,1],[-1,1]])*0.5
        x = np.matmul(x, jointstereo)

    x = np.transpose(x)
    if x.shape[1]%window_size != 0:
        x = np.concatenate([x, np.zeros((x.shape[0],window_size-x.shape[0]%window_size))], axis=-1)
    
    x = x/32768.0
    return x,samplerate

def write_wave(filename, data, samplerate):
    print("writewave: ", data.shape)
    data = np.transpose(data)
    if data.shape[-1] == 2: #use joint stereo
        jointstereo = np.array([[1,-1],[1,1]])
        data = np.matmul(data, jointstereo)
    scipy.io.wavfile.write(filename, samplerate, (np.clip(data,-1.0,1.0)*32767.0).astype(np.int16))

#implementation of MDCT
class MDCT:
    def MDCTmatrix(self, N):
        nums = np.arange(2*N)+0.5
        ws = nums[:,None]/N
        ks = nums[:N]*np.pi
        window = (np.pi*0.5)*ws
        matrix = (ws+0.5)*ks
        return np.cos(matrix)*(np.sin(window)*np.sqrt(2.0/N))

    def __init__(self, N):
        self.N = N
        self.matrix = self.MDCTmatrix(self.N)

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
        out = np.transpose(out, (1,0,2))
        out = np.reshape(out, (out.shape[0],-1))
        return out

def compressDCT(X,params):
    X = X*params['mult']
    if params['psyaco'] == 1:
        fv = np.array(iso226.get_freq_values(params['smplrate'], params['window']))
        X = X*fv;
    
    X = np.round(X) #quantization step

    if np.min(X) >= -128 and np.max(X) <= 127:
        params['dataint'] = 1
    else:
        params['dataint'] = 2
       
    typename = np.int8 if params['dataint']==1 else np.int16
    X = np.transpose(X, (1,0,2))
    #saved shape is now: (channels, windows, window size)
    X = X.flatten().astype(typename)
    compressedX = Compressor.compress(X)
    return compressedX
    
def decompressDCT(compressedX,params):
    typename = np.int8 if params['dataint']==1 else np.int16
    X = Compressor.decompress(compressedX, typename)
    X = X.astype(np.float32)
    X = np.reshape(X, (params['channels'], -1, params['window']))
    if params['psyaco'] == 1:
        fv = np.array(iso226.get_freq_values(params['smplrate'], params['window']))
        X = X / np.where(fv<0.00000001, 1, fv);
    X = X / params['mult']
    X = np.transpose(X, (1,0,2))
    return X

Compressor = LZMACompressor()

params = {}
params['mult'] = 10.0 #quantization multiplier. higher value > better quality and bigger file
params['window'] = 2 #MDCT window size
params['psyaco'] = 0 #use psychoacoustic model? 1=yes, 0=no

#wavename = "Kapittel_1.wav"
#wavename = "rule001.wav"
wavename = "fmi_0_gt.wav"
x,params['smplrate'] = read_wave(wavename, params['window'])
params['channels'] = x.shape[0]
mdct = MDCT(params['window'])
X = mdct.forward(x)
originalsize = X.shape[0]*X.shape[2]

X = compressDCT(X, params)

params['length'] = originalsize/params['smplrate']
params['bitrate'] = len(X)*8/1000/params['length']
print(f"compressedsize: {len(X)}")
print(f"seconds: {params['length']}")
print(f"bitrate: {params['bitrate']} kbps")

eblname = wavename[:-4]+".ebl"
save_ebl(eblname, X, params)

X,params = load_ebl(eblname)
X = decompressDCT(X, params)
x_reconstructed = mdct.backward(X)

write_wave("out.wav", x_reconstructed, params['smplrate'])
