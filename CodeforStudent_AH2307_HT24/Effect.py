import numpy as np
import pandas as pd
from tabulate import tabulate
def Effect(v_car, v_pt, v_slow, dist, carparking, transitprice, EU):
    
    # Here, calculate the vehicle kilometers traveled and externalities    
    VKT = dist * v_car
    wt = 0.01 * VKT
    acc = 0.25 * VKT
    noise = 0.081 * VKT
    emission = 0.004 * VKT
    co2 = 3.15 * 0.085 * 1.5 * VKT
    parking_rev = v_car * carparking
    transit_rev = v_pt * transitprice
    
    
    # create the table with externalities 
    table = {
        "Car Trips"            : round(np.sum(v_car)),
        "Transit Trips"       : round(np.sum(v_pt)),
        "Slow Trips"           : round(np.sum(v_slow)),
        "Expected Utility"     : round(np.sum(EU))
    }
    
    return pd.DataFrame(table.values(), index=list(table.keys()), columns=['INFORMATION'])