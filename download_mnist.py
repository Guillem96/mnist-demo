import gzip
import requests
from pathlib import Path

import numpy as np
from PIL import Image

data_url = "http://yann.lecun.com/exdb/mnist/"
data_files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz"
]

# Save image
def save_file(file_name, file_content):
    with file_name.open("wb") as f:
        f.write(file_content)


def save_images(image_file, label_file, out_path, num_images):
    img_file = gzip.open(str(data_path.joinpath(image_file)), 'r')

    image_size = 28

    img_file.read(16)
    buf = img_file.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    label_file = gzip.open(str(data_path.joinpath(label_file)), 'r')
    label_file.read(8)
    buf = label_file.read(1 * data.shape[0])
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    digit_counter = {k:0 for k in range(11)}

    for i in range(num_images):
        img = Image.fromarray(data[i].reshape(28,28), "I")
        img_out_path = str(out_path.joinpath("{}_{}.png".format(labels[i], digit_counter[labels[i]])))
        img.save(img_out_path)
        digit_counter[labels[i]] += 1

    return data, labels

# Create path to store the data
data_path = Path("data", "MNIST")
data_path.mkdir(exist_ok=True, parents=True)

for f in data_files:
    r = requests.get("{}{}".format(data_url, f))
    if r.status_code == 200:
        save_file(data_path.joinpath(f), r.content)
    
data, labels = save_images("train-images-idx3-ubyte.gz", 
			   "train-labels-idx1-ubyte.gz", 
                           data_path, 60000)