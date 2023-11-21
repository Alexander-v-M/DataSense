import numpy as np

from DataSense.elements.load_data import get_data_wf
from DataSense.pipelines.TrainModel.ml_baseline import baseline

if __name__ == '__main__':
    data = get_data_wf(dataset='tn')

    l_rsme = []
    l_smape = []
    l_mase = []
    l_u = []

    for i in range(5):

        rmse, r2, smape, mase, u = baseline(data)

        l_rsme.append(rmse)
        l_smape.append(smape)
        l_mase.append(mase)
        l_u.append(u)

    rsme_mean = np.mean(l_rsme), np.std(l_rsme)
    print(rsme_mean)
    smape_mean = np.mean(l_smape), np.std(l_smape)
    print(smape_mean)
    mase_mean = np.mean(l_mase), np.std(l_mase)
    print(mase_mean)
    u_mean = np.mean(l_u), np.std(l_u)
    print(u_mean)
