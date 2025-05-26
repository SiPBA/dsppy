import numpy as np
import matplotlib.pyplot as plt
import scipy


def signalRange(s):
    '''
    Detemina el rango en el que varia una señal en función del tipo de dato del 
    array en el que está almacenada

    Args:
        s: numpy.array 
            Señal de la que se determinará el rango
    
    Return:
        Tupla de 2 elementos con el límite inferior y el límite superior del 
        rango de la señal 's'.
    '''

    if np.issubdtype(s.dtype, np.integer):
        info = np.iinfo(s.dtype)
        return (info.min, info.max + 1)
    elif np.min(s)>=0: 
        return (0, 1)
    else:
        return (-1, 1)
    

def toDB(power, pref=1):
    '''
    Convierte un valor de potencia en decibelios
    
    Args:
        power: float
            Valor de potencia a convertir en decibelios

        pref: float, opcional (por defecto: 1)
            Valor de referencia
    
    Return:
        Valor 'power' en decibelios
    '''
    
    return np.round(10 * np.log10(power/pref), 4)



def genDither(nSamples, q, pdf, dtype=float):
    '''
    Genera dither con una PDF y potencia determinada
    
    Args:
        nSamples: int
            Número de muestras que se generarán
            
        q: 
            Valor del cuanto del cuantificador para el que se va a usar el 
            dither generado. Este valor determina la potencia del dither
            
        pdf: {'gaussian', 'rectangular', 'triangular'}
            PDF del dither que se va a generar
    
        dtype: dtype, opcional (por defecto: float)
            Tipo de datos de las muestras del dither
            
    Return:
        Array de numpy con las muestras del dither generado
    '''

    if pdf == 'triangular':
        dither = np.random.triangular(-q, 0, q, nSamples)
    elif pdf == 'rectangular':
        dither = np.random.uniform(-q/2, q/2, nSamples)
    elif pdf == 'gaussian':
        dither = np.random.normal(0, q/2, nSamples)
    else:
        print('Invalid PDF type:', pdf)
        dither = np.zeros(nSamples)
    return dither.astype(dtype)


def addDither(x, dither):
    '''
    Añade dither a una señal evitando el desbordamiento del tipo de datos
    
    Args:
        x: numpy.array
            Señal a la que se va a añadir dither

        dither: numpy.array
            Dither que se va a añadir a 'x'

    Return:
        Señal 'x' con el dither añadido
    
    '''
    
    xMin, xMax = signalRange(x)
    xDither = np.clip(x.astype(float) + dither, xMin, xMax)
    return xDither.astype(x.dtype)


def getNumberOfBits(s):
    '''
    Aproxima el número de bits que se usaron para cuantificar una señal
    
    Args:
        s: numpy.array
            Senal de la que se van a determinar los bits usados en su 
            cuantificación
            
    Return:
        Número de bits aproximado (cota inferior) que se usó para 
        cuantificar 's'
    '''
    
    return int(np.ceil(np.log2(len(np.unique(s)))))


def mse(s1, s2):
    '''
    Calcula el error cuadrático medio (MSE) entre 2 señales
    
    Args:
        s1: numpy.array
            Primera señal
            
        s2: numpy.array
            Segunda señal
    
    Return:
        MSE entre 's1' y 's2'
    
    '''
    
    s1 = s1.flatten().astype(float)
    s2 = s2.flatten().astype(float)
    n = np.min((len(s1), len(s2)))
    return np.mean((s1[:n]-s2[:n])**2)

    
def snr(s1, s2):
    '''
    Calcula la relación señal ruido (SNR) en decibelios entre 2 señales
    
    Args:
        s1: numpy.array
            Primera señal
            
        s2: numpy.array
            Segunda señal
    
    Return:
        SNR entre 's1' y 's2'
    
    '''
    s1 = s1.flatten().astype(float)
    s2 = s2.flatten().astype(float)
    n = np.min((len(s1), len(s2)))
    pSig = np.sum(s1[:n]**2)
    pErr = np.sum((s1[:n]-s2[:n])**2)
    return 10 * np.log10(pSig / pErr)
    

def psnr(s1, s2, peak=None):
    '''
    Calcula la relación señal ruido pico (PSNR) en decibelios entre 2 señales
    
    Args:
        s1: numpy.array
            Primera señal
            
        s2: numpy.array
            Segunda señal
    
    Return:
        PSNR entre 's1' y 's2'
    
    '''
    if peak is None: peak = signalRange(s1)[1]**2
    mse_ = mse(s1,s2)
    return 10 * np.log10(peak / mse_)


def partitionImage(img, step1, step2):
    '''
    Divide una imagen bidimensional en bloques rectangulares. Completa la 
    imagen con ceros si es necesario.
    
    Args:
        img: numpy.array
            Imagen que se va a dividir en bloques 
            
        step1: int
            Alto (en pixeles) de los bloques en los que se va a dividir 'img'
            
        step2: int
            Ancho (en pixeles) de los bloques en los que se va a dividir 'img'

    Return:
        Array de numpy con los pixels de cada bloque en forma de vector 1D
        Tiene dimensión AxB, donde A es el número de bloques y B es el número 
        de píxeles en un bloque (B='step1'*'step2')
    '''
    
    # Asegura imagen de 2 dimensiones
    if img.ndim == 3: img = img.reshape(img.shape[0], -1, order='F')

    # Ajusta filas y cols para que sean divisibles por step1 y step2
    nRow, nCol = img.shape
    padRow = (step1 - nRow % step1) % step1
    padCol = (step2 - nCol % step2) % step2 
    img = np.pad(img, [(0, padRow), (0, padCol)], mode='edge')
    
    # Particiona en bloques de step1 x step2
    blocks = []
    for r in range(0, nRow, step1):
        for c in range(0, nCol, step2):
            blocks.append(img[r:r+step1, c:c+step2].flatten())

    return np.array(blocks)


def composeImage(blocks, step1, step2, finalShape):
    '''
    Reconstruye una imagen bidimensional que previamente se ha divivido en
    bloque usando 'partition image'.
    
    Args:
        blocks: numpy.array
            Array de tamaño AxB con los bloques de la imagen, donde A es el 
            número de bloques y B es el número de píxeles en un bloque

        step1: int
            Alto (en pixeles) de los bloques en los que se dividió 'img'
            
        step2: int
            Ancho (en pixeles) de los bloques en los que se dividió 'img'
            
        finalShape: tuple
            Tamaño de la imagen tras la reconstrucción.
    
    Return:
        Array de numpy con la imagen reconstruida.
        
    '''
    
    nRow, nCol = finalShape[:2]
    if len(finalShape) == 3: nCol *= finalShape[2]
    
    bRow, bCol = np.ceil((nRow / step1, nCol/step2))
    img = np.zeros((int(bRow * step1), int(bCol * step2)), dtype=blocks.dtype)

    r, c = 0, 0
    for i in range(len(blocks)):
        block = blocks[i].reshape((step1, step2))
        img[r:r+step1, c:c+step2] = block
        c += step2
        if c >= nCol: r, c = r+step1, 0 

    return img[:nRow, :nCol].reshape(finalShape, order='F')


def zigzag(m):
    '''
    Recorre un array 2D en forma de zig-zag
    
    Args:
        m: numpy.array
            Array 2D que va a se recorrido en zig-zag
            
    Return:
        Array 1D de numpy con los elementos de m leídos en zig-zag
    
    '''
    
    h,w = m.shape
    zz = []
    for s in range(w+h+1):
        for i in range(h)[::s%2*2-1]:
            if -1<s-i<w:
                zz.append(m[i, s-i])
    return np.array(zz)



def zigzagIdx(h,w):
    '''
    Recorre un array 2D en forma de zig-zag
    
    Args:
        h: int
            Número de filas del array 2D
            
        w: int
            Número de columnas del array 2D
            
    Return:
        Tupla con los índices correspondientes a recoger un array 2D
        con h filas y w columnas en zig-zag
    
    '''
    
    idxr, idxc = [], []
    for s in range(w+h+1):
        for i in range(h)[::s%2*2-1]:
            if -1<s-i<w:
                idxr.append(i)
                idxc.append(s-i)
    return (idxr, idxc)


def downsample(s, n):
    '''
    Decrementa la frecienca de muestreo un factor n
    
    Args:
        s: numpy.array
            Señal unidimensional cuya frecuencia de muestreo va a ser 
            decrementada.
            
        n: int
            Factor de decremento de la frecuencia de muestreo.

    Return:
        Señal 's' con la frecienca de muestreo decrementada un factor n

    '''
    return s[::n]


def upsample(s, n):
    '''
    Aumenta la frecienca de muestreo un factor n
    
    Args:
        s: numpy.array
            Señal unidimensional cuya frecuencia de muestreo va a ser 
            aumentada.
            
        n: int
            Factor de aumento de la frecuencia de muestreo.

    Return:
        Señal 's' con la frecienca de muestreo aumentada un factor n

    '''
    ss = np.zeros(len(s)*n, dtype=s.dtype)
    ss[::n] = s
    return ss


def bode(b, a=1, fs=None):
    '''
    Calcula y representa la respuesta en frecuencia de un sistema
    
    Args:
        b: numpy.array
            Array con los coeficientes del numerador de la función de
            transferencia del sistema
            
        a: numpy.arry, opcional (por defecto: 1)
            Array con los coeficientes del denominador de la función de
            transferencia del sistema
            
        fs: int, opcional (por defecto: None)
            Frecuencia de muestreo. Se usa para determinar el rango de
            frecuencias del eje horizontal en Hz. Si es None, las frecuencias
            del eje horizonal se muestran en rad/muestra normalizados.
    Return:
        None
        
    '''
    
    if fs is None:
        xfactor = np.pi
        xlim = [0,1]
        xlabel = 'Frecuencia normalizada ($x \\pi$ rad/muestra)'
        fs = 2*np.pi
    else:
        xfactor = 1
        xlim = [0,fs/2]
        xlabel = 'Frecuencia (Hz)'
    
    
    w, H = scipy.signal.freqz(b, a, fs=fs)
    
    fig, (ax1, ax2) = plt.subplots(2,1, layout='tight')
    mag = 20*np.log10(np.abs(H))
    mRange = [np.min((-10, np.min(mag))), np.max((3, np.max(mag)))]
    ax1.plot(w/xfactor, mag)
    ax1.set(ylabel='Magnitud (dB)' , xlim=xlim)
    if not np.any(np.isinf(mRange)) and not np.any(np.isnan(mRange)):
        ax1.set(ylim=mRange)
    ax1.grid()

    ax2.plot(w/xfactor, np.unwrap(np.angle(H))*180/np.pi) 
    ax2.set(xlabel=xlabel)
    ax2.set(ylabel='Fase (grados)', xlim=xlim)
    ax2.grid()


def filterDelay(b, a=1):
    '''
    Calcula y representa el retardo de grupo de un sistema
    
    Args:
        b: numpy.array
            Array con los coeficientes del numerador de la función de
            transferencia del sistema
            
        a: numpy.arry, opcional (por defecto: 1)
            Array con los coeficientes del denominador de la función de
            transferencia del sistema
            
    Return:
        None
        
    '''
    
    w, d = scipy.signal.group_delay((b,a))
    fig, ax = plt.subplots(layout='tight')
    ax.plot(w/np.pi, np.round(d))
    ax.set(xlabel='Frecuencia normalizada ($x \\pi$ rad/muestra)')
    ax.set(ylabel='Retardo (muestras)', xlim=[0,1])


def showPSD(s, fs):
    '''
    Representa el espectro de densidad de potencia de una señal
    
    Args:
        s: numpy.array
            Array de numpy con las muestras de la señal
            
        fs: int
            Frecuencia de muestreo
            
    Return:
        None
        
    '''
    fv, Px = scipy.signal.periodogram(s, fs)
    
    fig, ax = plt.subplots(layout='tight', figsize=[5, 5])
    ax.plot(fv, 10*np.log10(Px))
    ax.set(xlabel='Frecuencia (Hz)', ylabel='Potencia (dB/Hz)')


def getG722lpf():
    '''
    Devuelve los coeficientes del filtro paso-baja de análisis definido en el
    estándar G.722
    
    Args:
        (No tiene)
    Return:
        Array de numpy con los coeficientes de la respuesta al impulso del
        filtro
        
    '''
    
    h1 = [.366211e-3, -.134277e-2, -.134277e-2, .646973e-2, .146484e-2, -.190430e-1,\
          .390625e-2, .44189e-1, -.256348e-1, -.98266e-1, .116089, .473145]
    h1 = h1 + h1[::-1]
    return np.array(h1)

