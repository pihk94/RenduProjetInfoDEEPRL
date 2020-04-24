import datetime
from simulation import Portfolio_managment

#Module pour lancer une simulation
SYMBOLS = ['ETHBTC','XRPBTC','EOSBTC','ZECBTC','ETCBTC','XMRBTC']

#TRAIN SUR DES TRENDS MONTANTS 
#BTC de 3700€ a 4400
START = datetime.datetime(2019,3,15)
END = datetime.datetime(2019,4,15)
PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
PM.simulate(episode_depart=0,episode_fin=200)
#BTC de 4700€ a 7300
START = datetime.datetime(2019,4,15)
END = datetime.datetime(2019,5,15)
PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
PM.simulate(episode_depart=0,episode_fin=200)

#TRAIN SUR DES TRENDS DESCENDANTS
#BTC de 4400€ a 3700
START = datetime.datetime(2018,11,1)
END = datetime.datetime(2018,12,1)
PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
PM.simulate(episode_depart=0,episode_fin=200)
#BTC de 7700€ a 4800
START = datetime.datetime(2020,2,15)
END = datetime.datetime(2020,3,15)
PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
PM.simulate(episode_depart=0,episode_fin=200)

#TRAIN SUR DES TRENDS =
#BTC de 5600 a 5600
START = datetime.datetime(2018,9,1)
END = datetime.datetime(2018,10,1)
PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
PM.simulate(episode_depart=0,episode_fin=200)
#BTC de 3100€ a 3000
START = datetime.datetime(2019,1,1)
END = datetime.datetime(2019,2,1)
PM = Portfolio_managment(SYMBOLS,START,END,'train',LR = 2e-5,reward='avg_log_cum_return')
PM.simulate(episode_depart=0,episode_fin=200)
