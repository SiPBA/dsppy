import numpy as np
import scipy

def dct(N):
    '''
    Construye la matriz de coeficientes de la transformada de coseno discreto
    (DCT) de tamaño N x N
    
    Args:
        N: int
            Número de filas/columnas de la matriz de tranformación
            
    Return:
        Matriz de transformación

    '''

    return scipy.fft.dct(np.eye(N), axis=0, norm='ortho')


def dst(N):
    '''
    Construye la matriz de coeficientes de la transformada de seno discreto
    (DST) de tamaño N x N
    
    Args:
        N: int
            Número de filas/columnas de la matriz de transformación
            
    Return:
        Matriz de transformación

    '''

    return scipy.fft.dst(np.eye(N), type=1, norm='ortho')


def dft(N):
    '''
    Construye la matriz de coeficientes de la transformada discreta de Fourier
    (DFT) de tamaño N x N
    
    Args:
        N: int
            Número de filas/columnas de la matriz de transformación
            
    Return:
        Matriz de transformación

    '''

    return scipy.fft.fft(np.eye(N), axis=0, norm='ortho')


def dwht(N):
    '''
    Construye la matriz de coeficientes de la transformada de Walsh-Hadamard
    (DWHT) de tamaño N x N
    
    Args:
        N: int
            Número de filas/columnas de la matriz de transformación
            
    Return:
        Matriz de transformación

    '''
    H = scipy.linalg.hadamard(N)
    sign_changes = np.array([np.sum(np.diff(H[i,:]) != 0) for i in range(N)])
    H = H[np.argsort(sign_changes),:] / np.sqrt(N)
    return H
