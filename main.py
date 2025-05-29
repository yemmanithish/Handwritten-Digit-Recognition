from train.train_model import train
from predict.predict_digit import predict_sample

if __name__ == "__main__":
    model = train()
    predict_sample(model)