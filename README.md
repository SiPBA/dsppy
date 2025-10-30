# Digital signal processing for educational purposes

Digital signal processing library adapted to the knowledge and needs of students of subjects related to signal processing.

## Overview

**dsppy** is an open source Python library designed for educational and practical purposes in the field of digital signal processing (DSP). Its goal is to provide a simple and approachable interface for students and instructors studying or teaching signal processing concepts.

## Features

- Simple API inspired by standard DSP notations
- Essential DSP functions (filtering, transforms, convolution, etc.)
- Visualization tools with Matplotlib support
- Integration with popular libraries: NumPy, SciPy, scikit-learn, NetworkX
- Educational examples and clear documentation

## Installation

You can install dsppy using pip:

```bash
pip install dsppy
```

## Requirements

- Python 3.8 or newer
- numpy
- matplotlib
- scipy
- scikit-learn
- networkx

## Usage Example

```python
import numpy as np
import dsppy

# Example: Create a simple signal and apply a filter
x = np.sin(2 * np.pi * np.linspace(0, 1, 100))
y = dsppy.filters.lowpass(x, cutoff=0.2)

dsppy.plot.time(x, label="Original")
dsppy.plot.time(y, label="Filtered")
```

More examples can be found in the `examples/` folder.

## Documentation

- [API Reference](https://github.com/SiPBA/dsppy/)
- Tutorials and notebooks (in progress)

## Contributing

Contributions, suggestions, and bug reports are welcome! Please open an [issue](https://github.com/SiPBA/dsppy/issues) or submit a pull request.

## License

This project is licensed under the [GPL-3.0-or-later](LICENSE) license.

