from DataSense.pipelines.Forecast.dl_forecast_v2 import Predictor
from DataSense.elements.load_data import get_data_wf, get_data
from DataSense.elements.visualize import plot_predictions_and_forecasting
from DataSense.pipelines.TrainModel.dl_train_dl_model import train_model

if __name__ == '__main__':
    # get dataframe with data
    # data = get_data_wf(dataset='tn')
    data = get_data()
    # train and save model, trained on given data
    meta_fp, df_result = train_model(
        data=data,
        batch_size=128,
        n_lag=50,
        features='generated',  # 'generated' or 'assigned'
        vis_data=True,
        hidden_dim=128,
        layer_dim=3,
        dropout=0.2,
        n_epochs=10000,
        learning_rate=1e-5,
        model_type='gru',  # rnn lstm gru
        scaler_in='minmax',  # robust, minmax
    )

    # initialize the predictor class
    P = Predictor(model_storage_fp=meta_fp,
                  dataset=data,
                  num_prediction=50)

    # get forecast predictions
    df_forecasting = P.forecast()

    # plot in one plot, predictions, forecasts and actual values
    plot_predictions_and_forecasting(df_result, df_forecasting)
