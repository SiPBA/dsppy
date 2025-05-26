from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


class CRCCoder:
    '''
    Construye un codificador/descodificador basado en un código de redundancia 
    cíclica
    
    Atributos:
        p: numpy.array
            Polinomio generador
    
    '''
    
    p = None
    def __init__(self, polgen):
        '''
        Contructor de la clase.
        
        Args:
            poligen: iterable
                Polinomio generador. El primer elemento corresponde al
                coeficiente de mayor grado.
        '''
        
        self.p = np.array(polgen)

    def register(self, bits):
        '''
        Función interna. Opera el registro de desplazamiento
        
        '''
        
        r = np.zeros(len(self.p)-1, dtype='uint8')
        
        for i in range(len(bits)):
            r = np.roll(r, -1)
            last = r[-1]
            r[-1] = bits[i]            
            if last: r = np.logical_xor(r, self.p[1:])

        return r

    def encode(self, bits):
        '''
        Genera una palabra de código. Para ello calcula el CRC corresponediente
        a una palabra de mensaje y se lo añade al final.
        
        Args:
            bits: numpy.array
                Palabra de mensaje
                
        Return:
            Array de numpy con la palabra de código
        '''
        
        bitsAll = np.concatenate((bits, np.zeros(len(self.p)-1, dtype='uint8')))
        crc = self.register(bitsAll)
        return np.concatenate((bits, crc))
    
    def decode(self, bits):
        '''
        Comprueba una palabra de código y determina si ha habido o no errores. 
        
        Args:
            bits: numpy.array
                Palabra de mensaje
                
        Return:
            Tupla con un array de numpy con la palabra de mensaje y un valor
            booleano indicando si ha habido o no errores            

        '''
        
        msg = bits[:-(len(self.p)-1)]
        syndrome = self.register(bits)
        err = bool(np.sum(syndrome))
        return msg, err


class ConvCoder:
    '''
    Construye un codificador convolucional
    
    Atributos:
        K: int
            Longitud de restricción
        
        g: numpy.array
            Vectores generadores (un vector por fila)
            Dimensiones: (n, k*K)

        k: int
            Número de bits que son procesados a con cada entrada
            
        n: int
            Número de bits de salida para cada entrada
            
        M: int
            Número de bits de memoria. Equivale a (K-1)*k
        
    '''

    def __init__(self, K, g):
        '''
        Constructor de la clase

        Args:
            K: int
                Longitud de restricción

            g: numpy.array
                Vectores generadores (un vector por fila)
                Dimensiones: (n, k*K)
        '''

        self.K = K
        self.g = np.array(g).T
        kK, self.n = self.g.shape
        self.k = kK // K
        self.M = (K - 1) * self.k

        # Listas de entradas y estados posibles
        self.inputs = list(product([0, 1], repeat=self.k))
        self.states = list(product([0, 1], repeat=self.M))

        # Estado inicial
        self.state = self.states[0]

        # Inicializar transiciones (cambios de estado) y salidas
        self.transitions = {} 
        self.outputs = {}
        for state in self.states:
            for u in self.inputs:
                reg = np.array(u + state, dtype=int)
                out = np.mod(np.dot(reg, self.g), 2)
                self.outputs[state, u] = tuple(out)
                self.transitions[state, u] = (u + state)[:self.M]

        
    def encode(self, data, ending=True):
        '''
        Codifica un array de datos
    
        Args:
            data: numpy.array
                Array 1D con los bits de datos a codificar
    
            ending: bool
                Añade bits de cola necesarios para volver al estado 0
    
        Return:
            Array 1D con los bits codificados
        '''
        
        # La longitud de data debe ser múltiplo de k
        data = np.concatenate((data, [0]*((-len(data)) % self.k))).astype(int)
        
        # Añade bits de cola si es necesario
        if ending: data = np.concatenate((data, [0]*self.M))
                
        code = []
        for i in range(0, len(data), self.k):
            u = tuple(data[i:i+self.k])
            output = self.outputs[self.state, u]
            code.append(output)
            self.state = self.transitions[self.state, u]
    
        return np.concatenate(code)


    def decode(self, code, show=False):
        '''
        Decodifica una secuencia de código usando el algoritmo de Viterbi
    
        Args:
            code: numpy.array
                Array 1D con los bits codificados
    
        Return:
            Array 1D con datos correspondientes a descodificar code
        '''

    
        # Inicialización
        code = np.array(code, dtype=int).reshape((-1, self.n))
        metrics = {state: np.inf for state in self.states}
        metrics[self.states[0]] = 0        
        paths = {}
    
        # Viterbi
        steps = len(code)
        for t in range(steps):
            metricsT = {state: np.inf for state in self.states}
            for state in self.states:
                if metrics[state] < np.inf:
                    for u in self.inputs:
                        nextState = self.transitions[state, u]
                        out = self.outputs[state, u]
                        metric = metrics[state] + \
                                 np.sum(np.bitwise_xor(out, code[t]))
    
                        if metric < metricsT[nextState]:
                            metricsT[nextState] = metric
                            paths[t, nextState] = (state, u)
            metrics = metricsT
    
        # Reconstruir camino más probable (camino Viterbi)
        state = self.states[np.argmin(metrics)]    
        data, statePath = [], []
        for t in reversed(range(steps)):
            statePath.insert(0, state)
            state, u = paths[t, state]
            data.extend(u[::-1])
    
        # Visualizar el camino en el diagrama Trellis
        if show: 
            path = [(t, s) for t, s in enumerate(statePath[:-1])]
            self.plot_trellis(steps=len(code), path=path)
    
        return data[::-1]


    def plotTrellis(self, steps=5, path=None):
        '''
        Dibuja el diagrama de rejilla (Trellis)

        Args:
            steps: int
                Número de pasos de tiempo a mostrar

            path: list of tuple, opcional (por defecto: None)
                Camino de descodificación que será resaltado
        '''

        toStr = lambda bits: "".join(map(str, bits))
        toInt = lambda bits: int("".join(map(str, bits)), 2)
        G = nx.DiGraph()
        colors = plt.get_cmap('tab10')

        # Estado inicial
        activeStates = [self.states[0]]
        for t in range(steps):
            nextActiveStates = []
            for state in activeStates:
                for idx_u, u in enumerate(self.inputs):
                    nextState = self.transitions[state, u]
                    nextActiveStates.append(nextState)
                    out = self.outputs[state, u]

                    G.add_node((t, state), 
                               pos=(t, -toInt(state)), 
                               label=toStr(state))
                    G.add_node((t + 1, nextState), 
                               pos=(t+1, -toInt(nextState)),
                               label=toStr(nextState))
                    G.add_edge((t, state), (t + 1, nextState),
                               output=toStr(out),
                               color=colors(idx_u % 10))

            activeStates = nextActiveStates

        # Figura
        fig = plt.figure(figsize=((steps+1)*2, 2**(self.M+1)))

        # Representa el grafo
        positions = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos=positions, width=2,
                node_size=1500, node_color='lightblue', edgecolors='black',
                edge_color=[G[u][v]['color'] for u, v in G.edges()],
                labels=nx.get_node_attributes(G, 'label'), font_size=9)

        # Representa etiquetas de las aristas
        labels = nx.get_edge_attributes(G, 'output')
        for (src, dst), label in labels.items():
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2) ** 0.5
            offset = 0.4
            xl = x1 + offset * dx / length
            yl = y1 + offset * dy / length
        
            plt.text(xl, yl, label, ha='center', va='center',
                     fontsize=8, color=G[src][dst].get('color', 'black'),
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, 
                               boxstyle='round,pad=0.1'))

        # Leyenda
        patches = [mpatches.Patch(color=colors(i), label=toStr(u))
            for i, u in enumerate(self.inputs)]
        fig.legend(handles=patches, title="Entrada", loc='lower left')

        # Camino Viterbi
        if path is not None:
            v_edges = list(zip(path[:-1], path[1:]))
            v_edges = [e for e in v_edges if e[0] in G.nodes and e[1] in G.nodes]
            nx.draw_networkx_edges(G, pos=positions, edgelist=v_edges, edge_color='black',
                                   width=4, style='dashed')

        fig.show()


    def plotStateDiagram(self):
        '''
        Dibuja el diagrama de estados del codificador convolucional
        '''
        
        toStr = lambda bits: "".join(map(str, bits))
    
        G = nx.DiGraph()
        colors = plt.get_cmap("tab10")
    
        for state in self.states:
            for idx, u in enumerate(self.inputs):
                nextState = self.transitions[state, u]
                out = self.outputs[state, u]
    
                G.add_edge(
                    toStr(state),
                    toStr(nextState),
                    output=toStr(out),
                    color=colors(idx % 10)
                )

        # Figura
        fig = plt.figure(figsize=(5, 5))
        
        # Representa el grafo
        positions = nx.circular_layout(G)
        curvature = 0.1
        nx.draw(G, pos=positions, 
                node_size=1500, width=2, with_labels=True,
                node_color='lightblue', edgecolors='black', 
                edge_color=[G[u][v]['color'] for u, v in G.edges()],
                connectionstyle='arc3,rad='+str(curvature),font_size=10)


        # Dibujar etiquetas de aristas manualmente en curvas
        edge_labels = nx.get_edge_attributes(G, 'output')
        for (src, dst), label in edge_labels.items():
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            color = G[src][dst]['color']
        
            if src == dst:
                offset = 0.25    # Distancia etiqueta-arista
                xl = x1
                yl = y1 + offset
            else:
                mx, my = (x1+x2) / 2, (y1+y2) / 2   # Coord. del punto medio
                dx, dy = x2-x1, y2-y1               # Desplazamiento ortogonal

                length = (dx**2 + dy**2) ** 0.5                
                offset = abs(curvature) * length / 2
                xl = mx + offset * dy / length * np.sign(curvature)
                yl = my + offset * -dx / length * np.sign(curvature)
                
            plt.text(xl, yl, label, color=color, fontsize=8, 
                     ha='center', va='center', 
                     bbox=dict(facecolor='white', edgecolor='none', 
                               alpha=0.7, boxstyle='round,pad=0.1'))

        # Leyenda
        patches = [mpatches.Patch(color=colors(i), label=format(i, f'0{self.k}b'))
            for i in range(len(self.inputs))]
        fig.legend(handles=patches, title="Entrada", loc='lower left')
        fig.show()
           


#%% 

def canalBSC(bits, ber):
    '''
    Simula la transmisión de datos por un canal BSC.
    
    Args:
        bits: numpy.array
            Array unidimensional con valores '0' o '1' que contiene los datos
            transmitidos por el canal.
            
        ber: float
            Probabilidad de error de bits del canal. Por ejemplo, un valor de 
            0.05 equivale a un BER del 5%
            
    Return:
        Array de numpy con los datos de 'bits' tras la transmisión por el canal
        
    '''
    
    idx = np.random.random(len(bits)) <= ber
    output = np.copy(bits)
    output[idx] = (output[idx] + 1) % 2
    return output
