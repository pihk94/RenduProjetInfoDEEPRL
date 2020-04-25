import datetime
from simulation import Portfolio_managment

#Module pour lancer une simulation
SYMBOLS = ['ETHBTC','XRPBTC','EOSBTC','ZECBTC','ETCBTC','XMRBTC']

## Dans un premier temps nous trainons sur différentes périodes en fonction de différents trend (haussier, descendant , stagnant)

# #TRAIN SUR DES TRENDS MONTANTS 
# #BTC de 3700€ a 4400
# START = datetime.datetime(2019,3,15)
# END = datetime.datetime(2019,4,15)
# PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=0,episode_fin=200)
# #BTC de 4700€ a 7300
# START = datetime.datetime(2019,4,15)
# END = datetime.datetime(2019,5,15)
# PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=0,episode_fin=200)

# #TRAIN SUR DES TRENDS DESCENDANTS
# #BTC de 4400€ a 3700
# START = datetime.datetime(2018,11,1)
# END = datetime.datetime(2018,12,1)
# PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=0,episode_fin=200)
# #BTC de 7700€ a 4800
# START = datetime.datetime(2020,2,15)
# END = datetime.datetime(2020,3,15)
# PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=0,episode_fin=200)

# #TRAIN SUR DES TRENDS =
# #BTC de 5600 a 5600
# START = datetime.datetime(2018,9,1)
# END = datetime.datetime(2018,10,1)
# PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=0,episode_fin=200)
# #BTC de 3100€ a 3000
# START = datetime.datetime(2019,1,1)
# END = datetime.datetime(2019,2,1)
# PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=0,episode_fin=200)


### TEST SUR LA PERIODE SUIVANTE POUR LES TROIS MEILLEURS DES 6 
# BEST UP TREND on Next PERIOD
# START = datetime.datetime(2019,4,1)
# END = datetime.datetime(2019,5,1)
# PM = Portfolio_managment(SYMBOLS,START,END,'test',LR = 9e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=200,episode_fin=400,poids='Model/Model_CNN_2019-03-15_2019-04-15_nbep_200/Poids/Agent_poids_train_ep_200.h5')

# # BEST DOWN TREND on Next Period
# START = datetime.datetime(2018,11,15)
# END = datetime.datetime(2018,12,15)
# PM = Portfolio_managment(SYMBOLS,START,END,'test',LR = 9e-5,reward='avg_log_cum_return')
# PM.simulate(episode_depart=200,episode_fin=400,poids='Model/Model_CNN_2018-11-01_2018-12-01_nbep_200/Poids/Agent_poids_train_ep_200.h5')

# Best EQUAL trend next period
START = datetime.datetime(2018,9,15)
END = datetime.datetime(2018,10,15)
PM = Portfolio_managment(SYMBOLS,START,END,'test',LR = 9e-5,reward='avg_log_cum_return')
PM.simulate(episode_depart=200,episode_fin=400,poids='Model/Model_CNN_2018-09-01_2018-10-01_nbep_200/Poids/Agent_poids_train_ep_200.h5')