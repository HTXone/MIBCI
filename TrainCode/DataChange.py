import pandas as pd
import scipy
from scipy import io

OriginPath = "D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TestCode/TestData/S1"

for j in range(4, 6):
    features_struct = scipy.io.loadmat((OriginPath + "/block%d.mat") % (j))
    features = features_struct['data']
    dfdata = pd.DataFrame(features)
    datapath1 = (OriginPath + "/%d.csv") % (j)
    dfdata.to_csv(datapath1, index=False)


