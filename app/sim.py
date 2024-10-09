import doctest
import json
from functools import reduce
from operator import __or__
from random import random
import math


# MODELING & SIMULATION

init = {
    'Planet': {
        'time': 0, 
        'timeStep': 0.01, 
        'x': 0, 
        'y': 0.1, 
        'vx': 0.1, 
        'vy': 0
        },
    'Satellite': {
        'time': 0, 
        'timeStep': 0.01, 
        'x': 0, 
        'y': 1, 
        'vx': 1, 
        'vy': 0
        },
}

"""Tuning parameters for the simulation"""
Gr = 1              # Gravity Number
Dr = 1 / 4          # Drag Number (10 is exaggerated)
rho_min = 0.000001  # Density value (normalized) at R = 1.  
                    # This is nonzero but small so that the satellite is always acted upon by drag

def f_grav(dx, dy):
    fg_x = dx / (dx**2 + dy**2)**(3/2)
    fg_y = dy / (dx**2 + dy**2)**(3/2)
    
    return fg_x, fg_y

def f_drag(vx, vy, dx, dy):
    R = (dx**2 + dy**2)**(1/2)
    density_variation = math.exp(math.log(rho_min) * R)
    fd = (vx**2 + vy**2) * density_variation
    fd_x = fd * (dy / R)
    fd_y = -1 * fd * (dx / R)

    return fd_x, fd_y
    

def propagate(agentId, universe):
    """Propagate agentId from `time` to `time + timeStep`."""
    state = universe[agentId]
    time, timeStep, x, y, vx, vy = state['time'], state['timeStep'], state['x'], state['y'], state['vx'], state['vy']

    if agentId == 'Planet':     # assumes a linear trajectory for the planet
        x += vx * timeStep
        y += vy * timeStep
    elif agentId == 'Satellite':
        px, py, pvx, pvy = universe['Planet']['x'], universe['Planet']['y'], universe['Planet']['vx'], universe['Planet']['vy']
        dx = px - x
        dy = py - y

        #
        #   First step of the Runge-Kutta Iteration
        #
        fg_x_n, fg_y_n = f_grav(dx, dy)
        fd_x_n, fd_y_n = f_drag(vx, vy, dx, dy)

        k1_vx = (Gr * fg_x_n + Dr * fd_x_n) * timeStep
        k1_vy = (Gr * fg_y_n + Dr * fd_y_n) * timeStep
        k1_px = pvx * timeStep
        k1_py = pvy * timeStep
        k1_x = vx * timeStep
        k1_y = vy * timeStep

        #
        #   Second step of the Runge-Kutta Iteration
        #

        step_size = 0.5
        fg_x_half, fg_y_half = f_grav(dx + step_size*k1_px - step_size*k1_x, 
                                      dy + step_size*k1_py - step_size*k1_y)

        fd_x_half, fd_y_half = f_drag(vx + step_size*k1_vx, 
                                      vy + step_size*k1_vy, 
                                      dx + step_size*k1_px - step_size*k1_x, 
                                      dy + step_size*k1_py - step_size*k1_y)
        
        k2_vx = (Gr * fg_x_half + Dr * fd_x_half) * timeStep
        k2_vy = (Gr * fg_y_half + Dr * fd_y_half) * timeStep
        k2_x = (vx + 0.5*k1_vx) * timeStep
        k2_y = (vy + 0.5*k1_vy) * timeStep

        #
        #   Third step of the Runge-Kutta Iteration
        #
        step_size = 0.5
        fg_x_half2, fg_y_half2 = f_grav(dx + step_size*k1_px - step_size*k2_x, 
                                      dy + step_size*k1_py - step_size*k2_y)

        fd_x_half2, fd_y_half2 = f_drag(vx + step_size*k2_vx, 
                                      vy + step_size*k2_vy, 
                                      dx + step_size*k1_px - step_size*k2_x, 
                                      dy + step_size*k1_py - step_size*k2_y)
        
        k3_vx = (Gr * fg_x_half2 + Dr * fd_x_half2) * timeStep
        k3_vy = (Gr * fg_y_half2 + Dr * fd_y_half2) * timeStep
        k3_x  = (vx + 0.5*k2_vx) * timeStep
        k3_y  = (vy + 0.5*k2_vy) * timeStep

        #
        #   Fourth step of the Runge-Kutta Iteration
        #
        step_size = 1
        fg_x_full, fg_y_full = f_grav(dx + step_size*k1_px - step_size*k3_x, 
                                      dy + step_size*k1_py - step_size*k3_y)
        
        fd_x_full, fd_y_full = f_drag(vx + step_size*k3_vx, 
                                      vy + step_size*k3_vy, 
                                      dx + step_size*k1_px - step_size*k3_x, 
                                      dy + step_size*k1_py - step_size*k3_y)
        
        k4_vx = (Gr * fg_x_full + Dr * fd_x_full) * timeStep
        k4_vy = (Gr * fg_y_full + Dr * fd_y_full) * timeStep
        k4_x  = (vx + k3_vx) * timeStep
        k4_y  = (vy + k3_vy) * timeStep
        

        #
        #   Estimate position and velocity
        #
        
        vx += (1/6)*(k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
        vy += (1/6)*(k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)
        x  += (1/6)*(k1_x  + 2*k2_x  + 2*k3_x  + k4_x)
        y  += (1/6)*(k1_y  + 2*k2_y  + 2*k3_y  + k4_y)

    return {
        'time': time + timeStep, 
        'timeStep': 0.01+random()*0.09, 
        'x': x, 
        'y': y, 
        'vx': vx, 
        'vy': vy
        }

# DATA STRUCTURE

class QRangeStore:
    """
    A Q-Range KV Store mapping left-inclusive, right-exclusive ranges [low, high) to values.
    Reading from the store returns the collection of values whose ranges contain the query.
    ```
    0  1  2  3  4  5  6  7  8  9
    [A      )[B)            [E)
    [C   )[D   )
           ^       ^        ^  ^
    ```
    >>> store = QRangeStore()
    >>> store[0, 3] = 'Record A'
    >>> store[3, 4] = 'Record B'
    >>> store[0, 2] = 'Record C'
    >>> store[2, 4] = 'Record D'
    >>> store[8, 9] = 'Record E'
    >>> store[2, 0] = 'Record F'
    Traceback (most recent call last):
    IndexError: Invalid Range.
    >>> store[2.1]
    ['Record A', 'Record D']
    >>> store[8]
    ['Record E']
    >>> store[5]
    Traceback (most recent call last):
    IndexError: Not found.
    >>> store[9]
    Traceback (most recent call last):
    IndexError: Not found.
    """
    def __init__(self): self.store = []
    def __setitem__(self, rng, value): 
        (low, high) = rng
        if not low < high: raise IndexError("Invalid Range.")
        self.store.append((low, high, value))
    def __getitem__(self, key):
        ret = [v for (l, h, v) in self.store if l <= key < h] 
        if not ret: raise IndexError("Not found.")
        return ret
    
doctest.testmod()

# SIMULATOR

def read(t):
    try:
        data = store[t]
    except IndexError:
        data = []
    return reduce(__or__, data, {})

store = QRangeStore()
store[-999999999, 0] = init
times = {agentId: state['time'] for agentId, state in init.items()}
print(times)

for i in range(2000):
    for agentId in init:
        t = times[agentId]
        universe = read(t-0.001)

        try:
            R = ((universe['Planet']['x'] - universe['Satellite']['x'])**2 + (universe['Planet']['y'] - universe['Satellite']['y'])**2)**(1/2)
        except:
            pass
        else:
            if R < 0.1:
                break

        if set(universe) == set(init):
            newState = propagate(agentId, universe)
            store[t, newState['time']] = {agentId: newState}
            times[agentId] = newState['time']

with open('./public/data.json', 'w') as f:
    f.write(json.dumps(store.store, indent=4))