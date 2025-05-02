
==Mo 4/25: See very bottom for Layers please! After this, I have some Qs on how you'd like to proceed?==

==Ry 4/26: I added some comments/questions in pink! What were your q's? :)==

==Mo 4/27: See very bottom after== "**Explanation of code parts so far" **==please. Need you to read about the layers we already have and suggestions following. I also answered your questions.==

==Part 1: Core Model (Bi-LSTM + Dense + BatchNorm + Dropout)==

==Import necessary Libraries==
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
```
==Added Line 1, 10 -Ry 4/26==

==Suggested code to append after library imports: -Ry 4/26==
==Mo 4/27: yes but it needs to be placed correctly, we first need to split into x_train, x_val, x_test before scaling. Otherwise, you're leaking validation/test information into training via scaling, which is not a good plan lol. So I have fixed this section and implemented properly.==
 
```python
df = pd.read_csv('your_feature_matrix.csv') #Loads feature matrix into keras

#separates features (x) from labels (y)
X = df.drop(columns=['LLPS_propensity'])  # replace with label column name when finalized
y = df['LLPS_propensity']

#splitting code should go here ....i thinl

#normalizes feature data from matrix
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


==Custom Focal Loss (optional)==
```python
def focal_loss(gamma=2., alpha=0.25):
   def loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_tru
      y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true  alpha + (1 - y_true)  (1 - alpha)
        modulating_factor = keras.backend.pow((1 - p_t), gamma)
        return -keras.backend.mean(alpha_factor  modulating_factor  keras.backend.log(p_t))
    return loss_fn
```


==Part 2: Main LLPSNet Model and Monte Carlo Simulation Block==
==should the code for remaining layers be appended below line 15? -Ry==
==Mo 4/27: yes, this is the correct place to add more dense layers or other processing. I have fixed it to where we can add more dense layers, attention, etc.==
```python
class MonteCarloSimulation(keras.layers.Layer):
   def init(self, temperature=0.1):
      super().__init__()
      self.temperature = temperature
      self.energy_proj = Dense(1)
   
   def call(self, x):
      energy = self.energy_proj(x)
      return tf.sigmoid(-energy / self.temperature)

def build_combined_llps_model(seq_len, num_aa=20, num_additional_features=16):
   sequence_input = Input(shape=(seq_len,), name='sequence_input')
   embedding_layer = Embedding(input_dim=num_aa, output_dim=64)(sequence_input)
   lstm_out = LSTM(128)(embedding_layer)
   lstm_out = Dropout(0.5)(lstm_out)
 
   additional_input = Input(shape=(num_additional_features,), name='additional_input')
   merged = Concatenate()([lstm_out, additional_input])
   
   x = BatchNormalization()(merged)
   x = Dense(64, activation='relu')(x)
   x = Dropout(0.3)(x)

classification_output = Dense(1, activation='sigmoid', name='classification_output')(x)
monte_carlo_output = MonteCarloSimulation(name='monte_carlo_output')(x)

model = Model(inputs=[sequence_input, additional_input],
outputs=[classification_output, monte_carlo_output])
    return model   
```


==Part 3: Training Function==

==Import Metrics==
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```


==Data Splitting==
```python
def train_model(sequence_data, additional_data, labels, epochs=20, batch_size=32):
    total_size = len(sequence_data)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    X_seq_train, X_add_train, y_train = sequence_data[train_idx], additional_data[train_idx], labels[train_idx]
    X_seq_val, X_add_val, y_val = sequence_data[val_idx], additional_data[val_idx], labels[val_idx]
    X_seq_test, X_add_test, y_test = sequence_data[test_idx], additional_data[test_idx], labels[test_idx]
```
==Does this only split sequence data into training and test sets? or did I miss a snippet for splitting the full matrix of data? -Ry 4/26==
==Otherwise, we could use this to split all of the data:==
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
==Mo 4/27: Regarding the above code, the current code splits both. We have sequence data, additional feature data, and labels. It splits all three simultaneously, so we don't need the suggested code you have provided :) ==

==Model Building and Compilation==
```python
model = build_combined_llps_model(seq_len=sequence_data.shape[1],num_aa=20,
num_additional_features=additional_data.shape[1])
   model.compile(optimizer='adam',
      loss=focal_loss(gamma=2., alpha=0.25),
      metrics=['accuracy', 
```


==Epoch Loop==
```python
tf.keras.metrics.AUC(name='AUC')])
   for epoch in range(epochs):
      print(f"\nEpoch {epoch + 1}/{epochs}")
      history = model.fit([X_seq_train, X_add_train], y_train,
         batch_size=batch_size,
         epochs=1,
         validation_data=([X_seq_val, X_add_val], y_val),
         class_weight={0: 1, 1: 2},
         verbose=0)
      val_pred = model.predict([X_seq_val, X_add_val])
      val_binary = (val_pred > 0.5).astype(int)
      val_acc = accuracy_score(y_val, val_binary)
      print(f"  Train Loss: {history.history['loss'][0]:.4f} | Val Acc: {val_acc:.4f}")
```


==Final Evaluation    ==
```python
print("\nFinal Test Set Evaluation:")
    y_pred = model.predict([X_seq_test, X_add_test])
    y_binary = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_binary, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_binary))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    return model
```

==Save model:==
```python
model.save('semisim.h5')
```

==Part 4: K-Fold Cross Validation==
```python
def run_kfold(sequence_data, additional_data, labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(sequence_data)):
        print(f"\nFold {fold + 1}")
        model = build_combined_llps_model(seq_len=sequence_data.shape[1],                              num_additional_features=additional_data.shape[1])
      
   model.compile(optimizer='adam',
      loss={'classification_output': 'binary_crossentropy','monte_carlo_output': 'binary_crossentropy'},
      loss_weights={'classification_output': 1.0, 'monte_carlo_output': 0.5},
      metrics={'classification_output': ['accuracy'], 'monte_carlo_output': ['accuracy']})

   model.fit([sequence_data[train_idx], additional_data[train_idx]],
      {'classification_output': labels[train_idx],'monte_carlo_output': labels[train_idx]},
      validation_data=([sequence_data[val_idx], additional_data[val_idx]],
         {'classification_output': labels[val_idx],'monte_carlo_output': labels[val_idx]}),
      epochs=10, batch_size=32)            
```


==Example: Simulated Data==

```python
if name == "__main__":
    sequence_data = np.random.randint(0, 20, size=(1000, 100))  # Simulated sequence input
    additional_data = np.random.rand(1000, 16)  # Simulated feature input
    labels = np.random.randint(0, 2, size=(1000,))  # Binary labels
    model = train_model(sequence_data, additional_data, labels)
    run_kfold(sequence_data, additional_data, labels
```


**==To do next:==**== Part 5: Ready for LLPS Datasets==
- `X`==: should be preprocessed outputs from ==**==PONDR, PLAAC, SAPS, STRING==**== (flattened or padded per residue).==
- `y`==: binary LLPS label (1 = forms LLPS under physiological conditions, 0 = does not)==



**Explanation of code parts so far:**

==Part 1==
**Import necessary libraries**
- Brings in libraries for model building (`keras`), numerical data (`numpy`), and evaluation (`sklearn`).


**Custom Focal Loss**
- Focal loss is used to **handle class imbalance**, by focusing more on hard-to-classify examples.

- `alpha`: balances importance of positive vs. negative class.
- `gamma`: increases focus on incorrect predictions.


**Main Model (LLPSNet)**
- **Inputs**: Protein features shaped as `(residues, features_per_residue)`
- **Bi-LSTM**: Models residue sequence information both forward and backward.
- **Dense Layers**: Classic deep layers with non-linearity (ReLU).
- **BatchNorm + Dropout**: Improve generalization and training stability.
- **Output**: Sigmoid for binary classification (LLPS or not).


**==Part 2==**
**Monte Carlo Simulation Proxy**
- Mimics **Boltzmann-like sampling** of LLPS formation.
- Projects latent features to "energy", then uses temperature-scaled sigmoid to simulate probability.
- Represents **stochastic phase separation behavior** under varying physiological conditions.


**Add Monte Carlo to the model**
Produces two outputs:
- `final_output`: regular LLPS probability.
- `mc_layer`: simulated LLPS probability under thermal/energy constraints.


==Part 3==
**Import metrics**
- To evaluate model performance — just like PyTorch’s final testing block.

**Data Splitting**
- Defines a **70/15/15 split** for train/validation/test sets — commonly used when no external split is provided.
- Shuffles the indices to ensure **random distribution** of data before splitting.
- Creates index slices for each dataset split.
- Uses the index splits to extract:
    - `X_seq_*`: amino acid indices (input to LSTM)
    - `X_add_*`: physiological features (e.g., PONDR, PLAAC, etc.)
    - `y_*`: LLPS binary labels

**Model Building and Compilation**
- Builds a **hybrid model** with:
    - **Embedding + LSTM** for sequence input
    - **Dense layers** for concatenated physiological features
- Compiles the model with:
    - `adam`: Adaptive optimizer
    - `focal_loss`: Handles **class imbalance** by focusing on hard-to-classify examples
    - `accuracy` and `AUC`: Performance metrics for binary classification

**Epoch Loop**
- Begins a custom epoch loop, printing the current epoch number.
- Trains for **one epoch at a time** (mimicking PyTorch `for epoch in range()` style):
    - Uses class weights to rebalance positive (LLPS) and negative (non-LLPS) classes
    - Accepts **two inputs** (`sequence` and `features`)
    - `validation_data` lets us evaluate after every epoch
- After each epoch:
    - Gets **predictions** on validation set
    - Converts probabilities to binary classes
    - Calculates **validation accuracy** using `sklearn`
- Logs training loss and validation accuracy for each epoch — just like a typical PyTorch loop.

**Final Evaluation**
- At the end of training, evaluates the model on the **held-out test set**.


==Part 4==
**K-Fold Cross-Validation**
- Splits the data into 5 folds for **cross-validation**.
- Trains and validates the model on each fold to reduce overfitting and get a **more reliable performance estimate**.


## ==Layers==
**==Refer to Part 2==**
==Input Layer==
```
sequence_input = Input(shape=(seq_len,), name='sequence_input')

additional_input = Input(shape=(num_additional_features,), name='additional_input')
```

- **Sequence input** = protein sequence (amino acid IDs).
- **Additional input** = external features (e.g., disorder scores, interaction scores).


==Embedding Layer==
```
embedding_layer = Embedding(input_dim=num_aa, output_dim=64)(sequence_input)
```
- **Purpose**: Transforms amino acid IDs into **dense vectors** (size 64). 
- **Why**: Allows the model to learn meaningful properties of residues beyond just their IDs.


==LSTM Layer==
```
lstm_out = LSTM(128)(embedding_layer)
```
- **Purpose**: Reads the sequence order and context dependencies using memory cells.
- **128 units** = output a 128-dimensional summary of the sequence.
- **Why**: Captures long-range interactions, important in proteins.

==Should the LSTM layer include ==`return_sequences=False`==?==
```python
lstm_out = LSTM(128, return_sequences=False)(embedding_layer)
```

==Dropout after LSTM==
```
lstm_out = Dropout(0.5)(lstm_out)
```
- **Purpose**: Randomly zeroes 50% of neurons during training.
- **Why**: Forces model to generalize and avoid overfitting (="regularization").


==Concatenation==
```
merged = Concatenate()([lstm_out, additional_input])
```
- **Purpose**: Merge sequence-derived features with additional protein properties.


==Batch Normalization==
```
x = BatchNormalization()(merged)
```
- **Purpose**: Normalize activations across the batch.
- **Why**: Speeds up training, stabilizes gradients.


==Dense (64, ReLU)==
```
x = Dense(64, activation='relu')(x)
```
- **Purpose**: Learn complex interactions between features.
- **Why**: Adds non-linearity to help model more complicated relationships.


==Dropout after Dense==
```
x = Dropout(0.3)(x)
```
- **Purpose**: 30% of neurons randomly dropped after dense layer.
- **Why**: Again, helps prevent overfitting.


==Outputs==
==*Classification Output==
```
classification_output = Dense(1, activation='sigmoid', name='classification_output')(x)
```
- **Purpose**: Output LLPS probability (0–1).


==*Monte Carlo Output==
```
monte_carlo_output = MonteCarloSimulation(name='monte_carlo_output')(x)
```
- **Purpose**: Secondary output based on energy minimization concepts.
- **Custom Layer**: Applies a learned energy term and softens prediction via "temperature."


==Model Assembly==
```
model = Model(inputs=[sequence_input, additional_input], outputs=[classification_output, monte_carlo_output])
```
- **Purpose**: Full model that predicts two outputs.


==Mo 4/25 Based on this, do you think we should add more layers? I asked GPT to analyze the code and regarding the risks of adding more and this is what it came up with:==

| ==Layer Addition== | ==Purpose== | ==Risk== |
|---|---|---|
| **==Bidirectional LSTM==** | ==Capture both forward and backward sequence dependencies== | ==Slightly more parameters (~2x LSTM params)== |
| **==Stack another LSTM==** | ==Learn deeper sequence structure== | ==Training slower, risk of overfitting== |
| **==Extra Dense layer==**== (after merge)== | ==Allow deeper mixing of sequence + external features== | ==Might overfit if dataset small== |
| **==Attention Layer==** | ==Let model "focus" on important residues== | ==Higher complexity, tuning required== |


==(Mo continued): So if anything, I would probably change the LSTM to a Bidirectional LSTM. Then after the first Dense(64) and Dropout(0.3), maybe add another Dense Layer?==
==Also, depending on if our dataset is large (by large I mean like over 10,000 examples, which I don't think it is) we could stack another LSTM layer, but only if that is the case otherwise it's pointless. What do you think?==


























