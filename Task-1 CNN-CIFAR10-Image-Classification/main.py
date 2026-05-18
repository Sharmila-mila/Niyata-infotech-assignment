import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings('ignore')

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_multiple_images

def main():

    model, history, X_test, y_test = train_model()

    evaluate_model(model, X_test, y_test)

    predict_multiple_images(model,"test_images")

if __name__ == "__main__":
    main()
