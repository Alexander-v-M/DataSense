from doepy import build

from DataSense.pipelines.Forecast.dl_forecast_v2 import Predictor
from DataSense.elements.load_data import get_data_wf
from DataSense.elements.visualize import plot_predictions_and_forecasting
from DataSense.pipelines.TrainModel.dl_train_dl_model import train_model
from DataSense.elements.save_meta_data import save_txt

ex_d = build.full_fact({
    "lags": [50, 100],
    "h_layer": [32, 64, 128],
    "dropout": [2, 5],
    "lr": [3, 5]
})


def exp():
    for model in ['lstm', 'gru']:
        ex_d = build.full_fact({
            "lags": [50, 100],
            "h_layer": [32, 64, 128],
            "dropout": [2, 5],
            "lr": [3, 5]
        })

        rmse = []
        r = []
        smape = []
        smase = []
        u = []

        # get dataframe with data
        data_frame = get_data_wf(dataset='tn')

        for lags, h_layer, dropout, lr in ex_d.values:
            # train and save model, trained on given data
            metrics = train_model(
                data=data_frame,
                batch_size=128,
                n_lag=int(lags),
                features='generated',  # 'generated' or 'assigned'
                vis_data=False,
                hidden_dim=int(h_layer),
                layer_dim=3,
                dropout=float(dropout / 10),
                n_epochs=1000 if lr == 3 else 10000,  #
                learning_rate=1 / 10 ** lr,
                model_type=model,
                scaler_in='minmax',  # robust
                opt={
                    'del_non_std_val': (False, 400)
                },
                give_metr=True
            )

            rmse.append(metrics[0])
            r.append(metrics[1])
            smape.append(metrics[2])
            smase.append(metrics[3])
            u.append(metrics[4])

        ex_d['RSME'] = rmse
        ex_d['R2'] = r
        ex_d['sMAPE'] = smape
        ex_d['sMASE'] = smase
        ex_d['U'] = u

        save_txt(ex_d, f'{model}allmetrics')


if __name__ == '__main__':
    exp()
