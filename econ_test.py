from model_modules import *
import warnings
warnings.filterwarnings('ignore')

window_size_= 8
ecom_mod= econ_test('data/ESP.IDXEUR_Candlestick_1_Hour_BID_01.01.2015-31.12.2020.csv', window_size_)
ecom_mod.pp_test()
ecom_mod.kpss_test()
ecom_mod.autocorr_graph('autocorr_graph')