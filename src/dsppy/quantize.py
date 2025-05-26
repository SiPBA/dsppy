import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn.cluster



class UniformSQ:
    '''
    Construye un cuantificador escalar uniforme
    
    Atributos:
        q: float
            Tamaño del cuanto
        
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación
        
    '''
    
    q, L, C = None, None, None
    precision = None
    roundData = lambda x: None
    
    def __init__(self, b, inputRange, qtype='midrise', precision=4):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador
                
            inputRange: tuple
                Rango de entrada del cuantificador. Debe ser una tupla con dos 
                valores, los límites inferior y superior del rango.
        
            qtype: {'midrise', 'midtread'} (por defecto: 'midrise')
                Tipo del cuantificador (media contrahuella o media huella)
        
        '''
        
        self.precision = precision
        self.xMin, self.xMax = inputRange
        if qtype == 'midrise':
            self.L = 2**b
            self.q = (float(self.xMax)-float(self.xMin)) / self.L
            self.roundData = lambda x: np.floor((x - self.xMin) / self.q)
        elif qtype == 'midtread':
            self.L = 2**b-1
            self.q = (float(self.xMax)-float(self.xMin)) / self.L
            self.roundData = lambda x: np.round((x - self.xMin) / self.q - 0.5)
        else:
            print('Invalid type:', qtype)
            
        self.C = np.arange(self.xMin + self.q/2, self.xMax, self.q)
        if self.precision is not None: 
            self.C = np.round(self.C, self.precision)

        
    def quantize(self, data):
        '''
        Cuantifica una señal
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con las muestras de la señal cuantificadas. Tendrá
            las mismas dimensiones que 'data'
            
        '''
        
        code = self.encode(data)
        return self.decode(code).astype(data.dtype).reshape(data.shape)
        

    def encode(self, data):
        '''
        Realiza el mapeo del codificador (primera parte de la cuantificación)
        correspondiente a 'data'
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con el código correspondiente a 'data'
            
        '''

        code = self.roundData(np.array(data, dtype=float)).astype(int).flatten()
        code = np.clip(code, 0, self.L-1)
        return code
        

    def decode(self, code):
        '''
        Realiza el mapeo del descodificador (segunda parte de la 
        cuantificación).
        
        Args:
            code: numpy.array
                Array de numpy con las muestras de la señal codificadas
                
        Return:
            Array de numpy con las muestras cuantificadas de la señal
            codificada en 'code'. 

        '''
        
        dataQ = (code + 0.5) * self.q + self.xMin
        if self.precision is not None: dataQ = np.round(dataQ, self.precision)
        return dataQ


    def plot(self, ax=None):
        '''
        Representa la entrada frente a la salida del cuantificador
        
        Args:
            ax: Axes, opcional (por defecto: None)
                Ejes cartesianos en los que se representará el cuantificador.
                Si es None, se crearán unos nuevos ejes en una nueva figura.
        
        '''
        
        data = np.linspace(self.xMin, self.xMax, 1000)
        dataQ = self.quantize(data)
    
        if ax is None: fig, ax = plt.subplots(layout='tight')
        ax.plot(data, dataQ)
        ax.set(xlabel='Entrada', ylabel='Salida')
        ax.grid('on')    
        

class CompandedSQ(UniformSQ):
    '''
    Construye un cuantificador escalar con un esquema comprimido/logaritmico
    
    Atributos:
        q: float
            Tamaño del cuanto del cuantificador escalar uniforme subyacente
        
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación del 
            cuantificador escalar uniforme subyacente
        
    '''

    def __init__(self, b, inputRange, compressFcn, expandFcn, qtype='midrise', normalize=True):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador
                
            inputRange: tuple
                Rango de entrada del cuantificador. Debe ser una tupla con dos 
                valores, los límites inferior y superior del rango.
        
            compressFcn: callable
                Función que lleva a cabo la compresión de la señal. Admite, 
                como único argumento de entrada, la señal sin comprimir
                (numpy.array) y devuelve la señal comprimida (numpy.array)
        
            expandFcn: callable
                Función que lleva a cabo la expansión de la señal. Admite, 
                como único argumento de entrada, la señal comprimida
                (numpy.array) y devuelve la señal sin comprimir (numpy.array)

            qtype: {'midrise', 'midtread'} (por defecto: 'midrise')
                Tipo del cuantificador (media contrahuella o media huella)
                
            normalize: bool (por defecto: True)
                Indica si se normaliza la señal (se traslada la rango (-1,1))
                antes de la compresión y la expansión
        
        '''
        
        super().__init__(b, inputRange, qtype)
        
        self.compress = lambda x: x
        self.expand = lambda x: x
        if callable(compressFcn): self.compress = compressFcn
        if callable(expandFcn): self.expand = expandFcn
        
        if normalize:
            offset = self.xMax - (self.xMax - self.xMin)/2
            self.normalize = lambda x: (x - offset) / (self.xMax - offset)
            self.unnormalize = lambda x: x * (self.xMax - offset) + offset
        else:
            self.normalize = lambda x: x
            self.unnormalize = lambda x: x
    
    def encode(self, data):
        '''
        Realiza el mapeo del codificador (primera parte de la cuantificación)
        correspondiente a 'data'
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con el código correspondiente a 'data'
            
        '''

        dataNorm = self.normalize(data.astype(float))
        dataNormComp = self.compress(dataNorm)
        dataComp = self.unnormalize(dataNormComp)
        return super().encode(dataComp)

    def decode(self, code):
        '''
        Realiza el mapeo del descodificador (segunda parte de la 
        cuantificación).
        
        Args:
            code: numpy.array
                Array de numpy con las muestras de la señal codificadas
                
        Return:
            Array de numpy con las muestras cuantificadas de la señal
            codificada en 'code'. 

        '''
        
        dataQComp = super().decode(code)
        dataQCompNorm = self.normalize(dataQComp.astype(float))
        dataQNorm = self.expand(dataQCompNorm)
        dataQ = self.unnormalize(dataQNorm)
        return dataQ
    

class muLawSQ(CompandedSQ):
    '''
    Construye un cuantificador escalar con un esquema comprimido basado en la 
    ley Mu
    
    Atributos:
        q: float
            Tamaño del cuanto del cuantificador escalar uniforme subyacente
        
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación del 
            cuantificador escalar uniforme subyacente
        
    '''

    def __init__(self, b, inputRange, mu=255, qtype='midrise', normalize=True):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador
                
            inputRange: tuple
                Rango de entrada del cuantificador. Debe ser una tupla con dos 
                valores, los límites inferior y superior del rango.
            
            mu: float (por defecto: 255)
                Parámetro Mu del compresor/expansor
        
            qtype: {'midrise', 'midtread'} (por defecto: 'midrise')
                Tipo del cuantificador (media contrahuella o media huella)
                
            normalize: bool (por defecto: True)
                Indica si se normaliza la señal (se traslada la rango (-1,1))
                antes de la compresión y la expansión
        
        '''
        
        self.mu = mu
        super().__init__(b, inputRange, self.compress, self.expand, qtype, normalize)

    def compress(self, data):
        '''
        Realiza la compresión de la señal
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con las muestras de la señal comprimidas
            
        '''

        return (np.log(1+(self.mu * np.abs(data))) / np.log(1+self.mu)) * np.sign(data)

    def expand(self, data):
        '''
        Realiza la expansión de la señal
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con las muestras de la señal expandidas
            
        '''

        return ((1+self.mu)**np.abs(data)-1) * np.sign(data) / self.mu


class ALawSQ(CompandedSQ):
    '''
    Construye un cuantificador escalar con un esquema comprimido basado en la 
    ley A
    
    Atributos:
        q: float
            Tamaño del cuanto del cuantificador escalar uniforme subyacente
        
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación del 
            cuantificador escalar uniforme subyacente
        
    '''

    def __init__(self, b, inputRange, A=87.6, qtype='midrise', normalize=True):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador
                
            inputRange: tuple
                Rango de entrada del cuantificador. Debe ser una tupla con dos 
                valores, los límites inferior y superior del rango.
            
            A: float (por defecto: 87.6)
                Parámetro A del compresor/expansor
        
            qtype: {'midrise', 'midtread'} (por defecto: 'midrise')
                Tipo del cuantificador (media contrahuella o media huella)
                
            normalize: bool (por defecto: True)
                Indica si se normaliza la señal (se traslada la rango (-1,1))
                antes de la compresión y la expansión
        
        '''
        
        self.A = A
        super().__init__(b, inputRange, self.compress, self.expand, qtype, normalize)

    def compress(self, data):
        '''
        Realiza la compresión de la señal
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con las muestras de la señal comprimidas
            
        '''

        return np.sign(data) * np.where(
            np.abs(data) < (1 / self.A),
            (self.A * np.abs(data)) / (1 + np.log(self.A)),
            (1 + np.log(self.A * np.abs(data))) / (1 + np.log(self.A))
            )

    def expand(self, data):
        '''
        Realiza la expansión de la señal
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con las muestras de la señal expandidas
            
        '''

        return np.sign(data) * np.where(
            np.abs(data) < (1 / (1+np.log(self.A))),
            np.abs(data) * (1 + np.log(self.A)) / self.A,
            np.exp(np.abs(data)*(1+np.log(self.A))-1) / self.A
            )


class OptimalVQ:
    '''
    Construye un cuantificador vectorial optimizado a la PDF de la señal y
    basado en el algoritmo de Lloyd-Max
    
    Atributos:
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación
        
    '''
    
    C = None
    
    def __init__(self, b, data, algorithm='kmeans'):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador
                
            data: numpy.array
                Datos de entrenamiento usados para definir los intervalos de
                cuantificación óptimos mediante el algoritmo de Lloyd-Max.
                Es un array de numpy de 2 dimensiones con las muestras de la 
                señal. La primera dimensión indica el número de bloques 
                (vectores que serán tratados como un solo objeto por el 
                cuantificador vectorial) y la segunda dimensión es el tamaño
                del bloque (N).
            
            algorithm: 'kmeans' o función, opcional (por defecto: 'kmeans')
                Algorimo para crear el conjunto de niveles de cuantificación.
                Por defecto se usa la clase KMeans de sklearn pero si este 
                argumento es una función se llamará a esa función. La función
                debe aceptar dos argumenos de entrada: 'b' y 'data' y debe
                devolver un array de Numpy con el conjunto de de niveles de
                cuantificación.

        '''
        
        if algorithm == 'kmeans':
            obj = sklearn.cluster.KMeans(2**b, n_init='auto').fit(data)
            C = obj.cluster_centers_
        elif algorithm == 'kmeans2':
            C,_ = scipy.cluster.vq.kmeans2(data.astype(float), 2**b, minit='++')
        elif callable(algorithm):
            C = algorithm(b, data)
        else:
            print('Algoritmo no válido:', algorithm)
            
        self.C = np.squeeze(np.sort(C, axis=0))
            
    def quantize(self, data):
        '''
        Cuantifica una señal
        
        Args:
            data: numpy.array
                Array de numpy de 2 dimensiones con las muestras de la señal.
                La primera dimensión indica el número de bloques (vectores que
                serán cuantificados como un solo objeto por el cuantificador
                vectorial) y la segunda dimensión es el tamaño del bloque (N).
                
        Return:
            Array de numpy con las muestras de la señal cuantificadas. Tendrá
            las mismas dimensiones que 'data'
            
        '''
        
        code = self.encode(data)
        return self.decode(code).astype(data.dtype)


    def encode(self, data):
        '''
        Realiza el mapeo del codificador (primera parte de la cuantificación)
        correspondiente a 'data'
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con el código correspondiente a 'data'
            
        '''
        
        idx = np.zeros(len(data), dtype=int)
        for j in range(len(data)):
            distance = np.abs(self.C - data[j])
            if distance.ndim > 1: distance = np.sum(distance, axis=1)
            idx[j] = np.argmin(distance)        
        return idx
        

    def decode(self, code):
        '''
        Realiza el mapeo del descodificador (segunda parte de la 
        cuantificación).
        
        Args:
            code: numpy.array
                Array de numpy con las muestras de la señal codificadas
                
        Return:
            Array de numpy con las muestras cuantificadas de la señal
            codificada en 'code'. 

        '''
        return self.C[code.astype(int)]


class OptimalSQ (OptimalVQ):
    '''
    Construye un cuantificador escalar optimizado a la PDF de la señal y 
    basado en el algoritmo de Lloyd-Max
    
    Atributos:
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación
        
    '''

    def __init__(self, b, data, algorithm='kmeans'):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador
                
            data: numpy.array
                Datos de entrenamiento usados para definir los intervalos de
                cuantificación óptimos mediante el algoritmo de Lloyd-Max.
            
            algorithm: 'kmeans' o función, opcional (por defecto: 'kmeans')
                Algorimo para crear el conjunto de niveles de cuantificación.
                Por defecto se usa la clase KMeans de sklearn pero si este 
                argumento es una función se llamará a esa función. La función
                debe aceptar dos argumenos de entrada: 'b' y 'data' y debe
                devolver un array de Numpy con el conjunto de de niveles de
                cuantificación.
        
        '''
        
        super().__init__(b, data.reshape(-1,1), algorithm)
        
    def quantize(self, data):
        '''
        Cuantifica una señal
        
        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal
                
        Return:
            Array de numpy con las muestras de la señal cuantificadas. Tendrá
            las mismas dimensiones que 'data'
            
        '''
        
        dataQ = super().quantize(data.reshape(-1,1))
        return dataQ.reshape(data.shape)

    def plot(self, ax=None):
        '''
        Representa la entrada frente a la salida del cuantificador
        
        Args:
            ax: Axes, opcional (por defecto: None)
                Ejes cartesianos en los que se representará el cuantificador.
                Si es None, se crearán unos nuevos ejes en una nueva figura.
        
        '''
        
        q = np.diff(self.C).mean()
        data = np.linspace(self.C[0]-q, self.C[-1]+q, 1000)
        dataQ = self.quantize(data)
    
        if ax is None: fig, ax = plt.subplots(layout='tight')
        ax.plot(data, dataQ)
        ax.set(xlabel='Entrada', ylabel='Salida')
        ax.grid('on')    

