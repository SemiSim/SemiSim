1. Parsed data from source files will be in .csv format. Load the .csv using pandas:

```python
import pandas as pd

df = pd.read_csv("your_data.csv")

```

2. Split features and labels with:

```python
features = df[["feature1", "feature2"]]
labels = df["label"]

```

3. Convert to TensorFlow dataset for keras:

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)

```

Converting from Pandas DataFrames feature matrix (as a `pandas.DataFrame` or `.csv`) to a TensorFlow dataset (as a `tf.data.Dataset`NOT numpy arrays) that can be used to train a model

In python, you'd go from smth like this:
```python
import pandas as pd

df = pd.DataFrame({
    "feature1": [1, 2, 3],
    "feature2": [4, 5, 6],
    "label": [0, 1, 0]
})

```
To smth like this:
```python
import tensorflow as tf

features = df[["feature1", "feature2"]]
labels = df["label"]

dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))

```

4. Define Keras model like:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(1, activation='sigmoid')  # for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

```

5. Train Keras model with `tf.data.Dataset`

```python
model.fit(dataset, epochs=10)

```
we're feeding the TF Datset into `model.fit()`
























