import numpy as np
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i 
        
    dj_db = (1/m) * dj_db
    dj_dw = (1/m) * dj_dw
    
    return dj_db,dj_dw






def simple_gradient_decent(x,y,w_in,b_in,alpha, iteration, scale):
    w = w_in
    b = b_in
    
    for i in range(iteration):
        dj_db,dj_dw = compute_gradient(x, y, w, b)
        w = w - (dj_dw * alpha)
        b = b - (dj_db * alpha)
        
    return f"w = {w/scale},b = {b}"
scale = 600      
x_train = np.array([100, 250, 400, 600])/scale
y_train = np.array([18, 34, 52, 80])
x = x_train
y= y_train
w_in = 0
b_in = 0
alpha = 0.01
iteration = 10000
p=simple_gradient_decent(x, y, w_in, b_in, alpha, iteration,scale)
print(p)