import numpy as np
from data.load_data import load_mnist_data

def predict_sample(model):
    (_, _), (x_test, y_test) = load_mnist_data()
    index = np.random.randint(0, len(x_test))
    sample = np.expand_dims(x_test[index], axis=0)
    prediction = model.predict(sample)
    print(f"Predicted: {np.argmax(prediction)}, Actual: {y_test[index]}")