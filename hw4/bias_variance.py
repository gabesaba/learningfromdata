import math,time
import numpy as np
from scipy import integrate

def get_fit(x,y):
    
    xdagger = (x.getT()*x).getI()*x.getT()
    w = np.dot(xdagger, y)
    return w.A1

def fit_polynomial(target_fcn = lambda x: math.sin(math.pi*x), polynomial = [0, 1]):
    
    '''
    fits two data points(x1, x2) using model specified by polynomial 
    '''

    x1 = np.random.uniform(-1,1)
    x2 = np.random.uniform(-1,1) 
    y1 = target_fcn(x1)
    y2 = target_fcn(x2)
    
    x1_vector = np.array([])
    x2_vector = np.array([])
    for i in polynomial:
        fx1 = x1**i
        fx2 = x2**i

        x1_vector = np.append(x1_vector,fx1)
        x2_vector = np.append(x2_vector,fx2)
        
    X = np.matrix([x1_vector,x2_vector])
    y = np.array([y1, y2])
    
    return get_fit(X, y)

def calc_variance(g_bar_x, polynomial, start, end, n):

    running_sum = 0.0
    for i in range(n):

        g_d = fit_polynomial(polynomial=polynomial)
        g_d_x = get_g_bar_x(g_d, polynomial)

        results = integrate.quad(lambda x : (g_bar_x(x) - g_d_x(x))**2, start, end)
        running_sum += results[0] / float(end-start)
        
    return running_sum / n

def calc_bias(g_bar_x,f_x,start,end):
    results = integrate.quad(lambda x : (g_bar_x(x) - f_x(x))**2, start, end)
    return results[0] / float(end-start)

def get_g_bar(polynomial, n):
    #is there a closed form solution to this??
    
    f = lambda x: math.sin(math.pi*x)    

    running_sum = 0.0
    for i in range(0, n):
        running_sum += fit_polynomial(f, polynomial)
    g_bar = running_sum / (n)
    return g_bar

def get_g_bar_x(g_bar, polynomial):
    
    return lambda x: sum([g_bar[i] * x**power for i, power in enumerate(polynomial)])

def question4():
    
    polynomial = [1]
    return get_g_bar(polynomial, n = 10000)

def question5():

    f = lambda x: math.sin(math.pi*x)
    polynomial = [1]
    g_bar = get_g_bar(polynomial, n = 10000)
    g_bar_x = get_g_bar_x(g_bar, polynomial)
    bias = calc_bias(g_bar_x,f,-1,1)
    return bias

def question6():
    
    polynomial = [1]

    g_bar = get_g_bar(polynomial, n = 10000)
    g_bar_x = get_g_bar_x(g_bar, polynomial)
    return calc_variance(g_bar_x, polynomial, -1, 1, n = 10000)

def question7():
    
    f = lambda x: math.sin(math.pi*x)
    a = [0] #b
    b = [1] #ax
    c = [0, 1] #ax + b
    d = [2] #ax^2
    e = [0, 2] #ax^2 + b
    polynomials = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e}
    min_value = float('inf')
    min_key = ''
    for polynomial in polynomials:
        g_bar = get_g_bar(polynomials[polynomial], n = 10000)
        g_bar_x = get_g_bar_x(g_bar, polynomials[polynomial])
        bias = calc_bias(g_bar_x,f,-1,1)
        variance = calc_variance(g_bar_x, polynomials[polynomial], -1, 1, n = 10000)
        error = bias + variance
        if error < min_value:
            min_value = error
            min_key = polynomial
        print('basis: %s error: %f'%(polynomials[polynomial], error))
    return min_key

    
if __name__ == '__main__':
    print('Q4: %f'%float(question4()))
    print('Q5: %f'%question5())
    print('Q6: %f'%question6())
    print('Q7: %s'%question7())
