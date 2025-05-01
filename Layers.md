==Current Layers==

| Layer | Purpose |
|---|---|
| `Input(shape=(seq_len,), name='sequence_input')` | Sequence input layer (protein sequence) |
| `Embedding(input_dim=num_aa, output_dim=64)` | Turns amino acid integers into 64-dim vectors |
| `LSTM(128)` | Reads sequential amino acid relationships |
| `Dropout(0.5)` | Regularization to prevent LSTM overfitting |
| `Input(shape=(num_additional_features,), name='additional_input')` | Additional features input (PONDR, PLAAC, etc.) |
| `Concatenate()` | Merges sequence and feature branches |
| `BatchNormalization()` | Stabilizes merged features |
| `Dense(64, activation='relu')` | Learns patterns from combined features |
| `Dropout(0.3)` | Regularization after Dense layer |
| `Dense(1, activation='sigmoid')` | Outputs LLPS probability |
| `MonteCarloSimulation` (custom) | Outputs energy-based simulation result |


==Layer by layer visual==

```plaintext
[Sequence Input] → Embedding → LSTM → Dropout → 
                                              |
                                    [Concatenate with Additional Input]
                                              ↓
                                    BatchNorm → Dense(64) → Dropout(0.3)
                                              ↓
                          → Dense(1, sigmoid)           → MonteCarloSimulation
```

==Suggestions:==
## **==Change LSTM → Bidirectional LSTM==**
==This will improve the biological sequence models so that the proteins don't only move forward. ==
| Why? | How? |
|---|---|
| - A normal LSTM reads only forward (residue 1 → residue 100). - A Bidirectional LSTM reads forward and backward at the same time, capturing both N-terminal and C-terminal influences.\n
 | Very simple! Just replace: `lstm_out = LSTM(128)(embedding_layer)` with: `from tensorflow.keras.layers import Bidirectional` `lstm_out = Bidirectional(LSTM(128))(embedding_layer)` |


## **==Add Another Dense Layer After Dropout(0.3)==**
==This will help deepen the feature processing to where it's not too heavy, but it is more powerful.==
| Why? | How? |
|---|---|
| - One Dense(64) may not be enough to learn the complex interaction of sequence + external features. - Adding another Dense(32) will allow the network to refine its understanding before making a decision. - Also, another small Dropout helps regularization.\n
 | Right **after** `Dropout(0.3)`, insert: `python<br>x = Dense(32, activation='relu')(x)<br>x = Dropout(0.2)(x)<br>` |
