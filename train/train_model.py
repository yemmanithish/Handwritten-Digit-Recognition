from data.load_data import load_mnist_data
from model.cnn_model import build_model

def train():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    return model