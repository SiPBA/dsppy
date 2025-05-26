import numpy as np


class RunCountCoder:

    b = 1
    bCount = 5
    maxZeros = 2**bCount
    count = 0
    value = 0

    def __init__(self, b, bCount=5, value=0):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Número de bits de cada palabra distinta de 0. Debe ser un
                entero positivo (>=0)
            
        '''

        self.b = min(max(int(b), 1), 64)
        self.bCount = min(max(int(bCount), 1), 64)
        self.maxZeros = 2**self.bCount
        self.count = 0
        self.value = value
        
    def encode(self, data):
        '''
        Codifica un mensaje
        
        Args:
            data: numpy.array
                Array de numpy con el mensaje a codificar. Sus elementos deben
                ser números enteros mayores o iguales a 0 y menores a 2**b
                
        Return:
            Array de numpy con la cadena de bits correspondiente a la 
            codificacióin de 'data'
            
        '''
        
        code = []
        if np.array(data).ndim == 0: data = np.array([data])
        
        for elem in data.flatten():
            if elem == self.value: 
                self.count += 1
                if self.count >= self.maxZeros:
                    code += ['1'] * self.bCount + [*f'{self.value:0{self.b}b}']
                    self.count = 0
            else:
                code += [*f'{self.count:0{self.bCount}b}']
                code += [*f'{elem:0{self.b}b}']
                self.count = 0
                
        code += [*f'{self.count:0{self.bCount}b}'] 
        return np.array(code, dtype='uint8')

    def decode(self, code):
        data = []
        i = 0
        
        while i<(len(code) - self.bCount):
            zeros = int(''.join(code[i:i+self.bCount].astype('<U1')),2)
            i += self.bCount            
            elem = int(''.join(code[i:i+self.b].astype('<U1')),2)
            i += self.b
            data += [self.value] * zeros + [elem]
        
        zeros = int(''.join(code[i:i+self.bCount].astype('<U1')),2)
        data += [self.value] * zeros 
        return np.array(data)


class FixedLengthCoder:
    '''
    Construye un codificador de palabras de longitud fija
    
    Atributos:
        b: int
            Número de bits de cada palabra. Debe ser un entero positivo (>=0)
        
    '''

    b = 1

    def __init__(self, b):
        '''
        Constructor de la clase
        
        Args:
            b: int
                Número de bits de cada palabra. Debe ser un entero positivo (>=0)
            
        '''

        self.b = min(max(int(b), 1), 64)

    def encode(self, data):
        '''
        Codifica un mensaje
        
        Args:
            data: numpy.array
                Array de numpy con el mensaje a codificar. Sus elementos deben
                ser números enteros mayores o iguales a 0 y menores a 2**b
                
        Return:
            Array de numpy con la cadena de bits correspondiente a la 
            codificacióin de 'data'
            
        '''
        
        if np.array(data).ndim == 0: data = np.array([data])
        data = np.clip(data, 0, 2**self.b-1)
        code = [*''.join([f'{x:0{self.b}b}' for x in data])]
        return np.array(code, dtype='uint8')

    def decode(self, code):
        '''
        Descodifica un mensaje
        
        Args:
            code: numpy.array
                Array de numpy con 0s y 1s (secuencia de bits).
                
        Return:
            Array de numpy con el mensaje codificado en 'code'
            
        '''
        
        return np.array([int(''.join(code[i:i+self.b].astype('<U1')),2) 
                for i in range(0,len(code), self.b)])


