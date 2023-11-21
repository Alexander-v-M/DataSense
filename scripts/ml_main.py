from DataSense.elements.load_data import get_data_wf, get_data

from DataSense.pipelines.TrainModel.ml_train_ml_model import train_model
from DataSense.pipelines.Forecast.ml_forecast import Predictor


if __name__ == '__main__':
    # Load or obtain the time series data, for example, from a dataset 'tn'.
    data = get_data()

    # Train a seasonal time series forecasting model and obtain the model filepath and type.
    model_fp, model_type = train_model(
        data=data,
        seasonal=True,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        m=12,
        start_P=0
    )

    # Create a Predictor object with the trained model, dataset, and number of predictions to make.
    p = Predictor(model_storage_fp=model_fp,
                  dataset=data,
                  num_prediction=50,
                  model_type=model_type)


