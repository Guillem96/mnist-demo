# MNIST Demo

![](https://corochann.com/wp-content/uploads/2017/02/mnist_plot-800x600.png)

1. *Data injestion*. The `download_mnist.py` script will download the MNIST dataset and store it under the `data/MNIST` directory. 
Each file will be named with the following naming convenion: {image_class}_{img_id}.png where **image_class** is the image label and **img_id** is an autonumeric value which is different for every image.

```
`-- data
    `-- MNIST
          |-- 0_0.png # Label = 0, id = 0
          |-- 0_1.png # Label = 0, id = 1
          .
          .
          .
          |-- 8_0.png # Label = 8, id = 0
          
```
2. *Indexed access to the data*. Using the `dai.data.DirectoryDatasource` class we will define a custom indexed data accessor where the idx will be the image path.

```python
import numpy as np
from PIL import Image

import driftai as dai


class ImageTensorDatasource(dai.data.DirectoryDatasource):

    def loader(self, idx):
        return np.asarray(Image.open(idx)).reshape(28, 28, 1)
```

3. Define the runnable approach

4. Mix together using the config file
