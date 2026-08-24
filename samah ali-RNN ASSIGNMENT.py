import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.optimizers import Adam

#load the Dataset
data = pd.read_csv(r"C:\Daily_minimum_temps.csv")
#direction of the csv file should be the same as mine...
print(data.head())

data.info()

print("\nMissing Values:")
print(data.isnull().sum())

print("\nNumber of Records:")
print(len(data))

#Prepare the Data
data["Temp"] = data["Temp"].astype(str)

data["Temp"] = data["Temp"].str.replace( "?","-", regex=False)

data["Temp"] = pd.to_numeric(data["Temp"])


data["Date"] = pd.to_datetime(data["Date"])

#Plot Temperature Over Time
plt.figure(figsize=(12, 5))

plt.plot(
    data["Date"],
    data["Temp"]
)

plt.title("Daily Minimum Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Use previous 7 days to predict the 8th day

temperatures = data["Temp"].values.reshape(-1, 1)

sequence_length = 7

X = []
y = []

for i in range(len(temperatures) - sequence_length):

    X.append(temperatures[i:i + sequence_length])
    y.append(temperatures[i + sequence_length])

X = np.array(X)
y = np.array(y)

print("\nSequence Shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the Data
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Normalization
# Normalization scales the values to a smaller range,
# which helps RNN training and numerical stability.

scaler = MinMaxScaler()

scaler.fit(X_train.reshape(-1, 1))

X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

X_test_scaled = scaler.transform( X_test.reshape(-1, 1)).reshape(X_test.shape)

y_train_scaled = scaler.transform(y_train.reshape(-1, 1))

y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

print("\nRNN Input Shapes:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)

print("time_steps = 7")
print("features = 1")

# Model A 16 RNN Units
model_A = Sequential([
    Input(shape=(7, 1)),
    SimpleRNN(16, activation="tanh"),
    Dense(1)
])

model_A.compile(
    optimizer=Adam(),
    loss="mse"
)

model_A.summary()

#Train Model A
history_A = model_A.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Model A Training Loss

plt.figure(figsize=(8, 5))

plt.plot(
    history_A.history["loss"],
    label="Model A - 16 Units"
)

plt.title("Model A - Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()

plt.tight_layout()
plt.show()

#Model A Prediction

pred_A_scaled = model_A.predict( X_test_scaled)

pred_A = scaler.inverse_transform(pred_A_scaled)

#Model A Evaluation

mae_A = mean_absolute_error(y_test,pred_A)

mse_A = mean_squared_error( y_test, pred_A)

print("\nModel A Results:")
print("MAE:", mae_A)
print("MSE:", mse_A)

#Model A Actual vs Predicted

plt.figure(figsize=(12, 5))

plt.plot( y_test, label="Actual Temperature")

plt.plot( pred_A,label="Predicted Temperature - Model A")

plt.title("Model A - Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.legend()

plt.tight_layout()
plt.show()

#Model B - 32 RNN Units

model_B = Sequential([
    Input(shape=(7, 1)),
    SimpleRNN(32, activation="tanh"),
    Dense(1)
])

model_B.compile(optimizer=Adam(),loss="mse")

model_B.summary()

#Train Model B
history_B = model_B.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=50,
    batch_size=32,
    verbose=1
)
#Compare Training Loss
plt.figure(figsize=(8, 5))
plt.plot(
    history_A.history["loss"],
    label="Model A - 16 Units"
)

plt.plot(
    history_B.history["loss"],
    label="Model B - 32 Units"
)

plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()

plt.tight_layout()
plt.show()


#Model B Prediction

pred_B_scaled = model_B.predict(X_test_scaled)

pred_B = scaler.inverse_transform(pred_B_scaled)
#Model B Evaluation
mae_B = mean_absolute_error(y_test,pred_B)

mse_B = mean_squared_error(y_test, pred_B)

print("\nModel B Results:")
print("MAE:", mae_B)
print("MSE:", mse_B)


#Compare Models
results = pd.DataFrame({
    "Model": ["Model A", "Model B"],
    "RNN Units": [16, 32],
    "MAE": [mae_A, mae_B],
    "MSE": [mse_A, mse_B]
})
print("\nModel Comparison:")
print(results)

#Actual vs Predicted Both Models

plt.figure(figsize=(12, 5))
plt.plot(y_test,label="Actual Temperature")

plt.plot(pred_A,label="Model A - 16 Units")

plt.plot(pred_B, label="Model B - 32 Units")

plt.title("Actual vs Predicted Temperature")
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.legend()

plt.tight_layout()
plt.show()

#Conclusion

if mae_B < mae_A and mse_B < mse_A:

    print(
        "\nModel B performed better because it achieved "
        "lower MAE and MSE than Model A."
    )

elif mae_A < mae_B and mse_A < mse_B:

    print(
        "\nModel A performed better because it achieved "
        "lower MAE and MSE than Model B."
    )

else:

    print(
        "\nThe models produced mixed results. "
        "One model has a lower MAE while the other "
        "has a lower MSE."
    )

print(
    "\nIncreasing RNN units from 16 to 32 does not always "
    "produce a better model because more units increase "
    "model capacity but may also increase overfitting."
)