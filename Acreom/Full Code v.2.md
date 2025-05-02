==Mo 4/27: I've compiled all the changes you inquired about/suggested in the following code block below. I answered your questions regarding them in the ==== page, but I figured this would be more neat. ==

```python
#Part 1: Core Model
#Imports
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

#Custom Focal Loss
def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = keras.backend.pow((1 - p_t), gamma)
        return -keras.backend.mean(alpha_factor * modulating_factor * keras.backend.log(p_t))
    return loss_fn

#Part 2: Main LLPSNet Model and Monte Carlo Simulation
#Monte Carlo Simulation
class MonteCarloSimulation(keras.layers.Layer):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.energy_proj = Dense(1)

    def call(self, x):
        energy = self.energy_proj(x)
        return tf.sigmoid(-energy / self.temperature)

#Main LLPS Model Builder
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
#We can add more layers here if you want, lmk:
   # x = Dense(32, activation='relu')(x)
   # x = Dropout(0.2)(x)

    classification_output = Dense(1, activation='sigmoid', name='classification_output')(x)
    monte_carlo_output = MonteCarloSimulation(name='monte_carlo_output')(x)

    model = Model(inputs=[sequence_input, additional_input],
                  outputs=[classification_output, monte_carlo_output])
    return model

#Load Dataset
df = pd.read_csv('final_semisim_dataset.csv')
df['sequence_encoded'] = df['sequence_encoded'].apply(eval)
sequence_data = np.array(df['sequence_encoded'].tolist())
y = df['LLPS_propensity'].to_numpy()

#Split Data into train/val/test
X_additional = df.drop(columns=['LLPS_propensity', 'sequence_encoded'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_additional)

X_seq_train, X_seq_temp, X_add_train, X_add_temp, y_train, y_temp = train_test_split(
    sequence_data, X_scaled, y, test_size=0.3, random_state=42)
X_seq_val, X_seq_test, X_add_val, X_add_test, y_val, y_test = train_test_split(
    X_seq_temp, X_add_temp, y_temp, test_size=0.5, random_state=42)

#Part 3: Training
#Training Function
def train_model(sequence_data, additional_data, labels, epochs=20, batch_size=32):
    total_size = len(sequence_data)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    X_seq_train, X_add_train, y_train = sequence_data[train_idx], additional_data[train_idx], labels[train_idx]
    X_seq_val, X_add_val, y_val = sequence_data[val_idx], additional_data[val_idx], labels[val_idx]
    X_seq_test, X_add_test, y_test = sequence_data[test_idx], additional_data[test_idx], labels[test_idx]

    #Model Building and Compilation
    model = build_combined_llps_model(seq_len=sequence_data.shape[1],
                                      num_aa=20,
                                      num_additional_features=additional_data.shape[1])

    model.compile(optimizer='adam',
                  loss={'classification_output': 'binary_crossentropy',
                        'monte_carlo_output': 'binary_crossentropy'},
                  loss_weights={'classification_output': 1.0, 'monte_carlo_output': 0.5},
                  metrics={'classification_output': ['accuracy', tf.keras.metrics.AUC(name='AUC')],
                           'monte_carlo_output': ['accuracy']})

    #Epoch Loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        history = model.fit([X_seq_train, X_add_train],
                            {'classification_output': y_train, 'monte_carlo_output': y_train},
                            batch_size=batch_size,
                            epochs=1,
                            validation_data=([X_seq_val, X_add_val],
                                             {'classification_output': y_val, 'monte_carlo_output': y_val}),
                            verbose=1)

    #Final Evaluation
    print("\nFinal Test Set Evaluation:")
    y_pred = model.predict([X_seq_test, X_add_test])[0]  # Only classification output
    y_binary = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_binary, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_binary))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    #Save model
    model.save('semisim.h5')
    print("\nModel saved as 'semisim.h5' 🎉")
    return model

#K-Fold Cross Validation Function
def run_kfold(sequence_data, additional_data, labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(sequence_data)):
        print(f"\nFold {fold + 1}")

        model = build_combined_llps_model(seq_len=sequence_data.shape[1],
                                          num_additional_features=additional_data.shape[1])

        model.compile(optimizer='adam',
                      loss={'classification_output': 'binary_crossentropy',
                            'monte_carlo_output': 'binary_crossentropy'},
                      loss_weights={'classification_output': 1.0, 'monte_carlo_output': 0.5},
                      metrics={'classification_output': ['accuracy'],
                               'monte_carlo_output': ['accuracy']})

        model.fit([sequence_data[train_idx], additional_data[train_idx]],
                  {'classification_output': labels[train_idx],
                   'monte_carlo_output': labels[train_idx]},
                  validation_data=([sequence_data[val_idx], additional_data[val_idx]],
                                   {'classification_output': labels[val_idx],
                                    'monte_carlo_output': labels[val_idx]}),
                  epochs=10, batch_size=32)

#Run Training
# model = train_model(sequence_train, X_train, y_train)
# run_kfold(sequence_train, X_train, y_train)
```