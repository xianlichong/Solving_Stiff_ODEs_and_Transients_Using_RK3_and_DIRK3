# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 01:43:00 2021

@author: xianli
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:50:22 2021

@author: xianli
"""
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Demonstrate the correct implementation of the algorithms rk3 
 and dirk3 with 2 bvector inputs as function.
"""

def bvector1(x):
    """
    bvector1 function depends on input x and output is vector b(x)=0
    """
    X = x*0
    return X

def bvector2(x):
    """
    bvector2 function depends on input x and output is vector b(x) =
    [[cos (10x) − 10 sin (10x)],
    [199 cos (10x) − 10 sin (10x)],
    [208 cos (10x) + 10 000 sin (10x)]]

    """
    cos = np.cos(10*x)
    sin = np.sin(10*x)
    B = np.array([cos-10*sin, 199*cos-10*sin, 208*cos+10000*sin])
    return B

def rk3(A, bvector, y0, interval, N):
    """
    The function represent explicit 3rd order Runge-Kutta method for evenly 
    spaced grid, return numerical solution of an IVP using N interation on the 
    given interval using N+1 points (including endpoints) with initial value y0.
      
      Parameters (Inputs)
    ----------
    A : TYPE - Array
        Input matrix A is nxn matrix with constant coefficients
    
    bvector : TYPE -  Function
        Input bivector function depends only x and output is vector b
    
    y0 : TYPE - Array
        Initial data.
    
    interval : TYPE - List
        A list with start and end values.
    
    N : TYPE - Integer
        Number of steps the algorithm should take.
    
    Return (Outputs)
    -------
    x : TYPE - Array
        An array of the interval coordinate between steps
    y : TYPE - Array
        Contains the approximate solution for the rk3 method with dimension A
    """
    assert N > 1 # Steps must be more than 1
    assert type(A) == np.ndarray # input A only takes in array
    assert len(A) == y0.size # only accept array which fits the dimension  y0
    assert type(N) == int # input N only takes integer
    assert type(interval) == list #input interval only takes list
    

    h = (interval[1] - interval[0]) / N
    x = np.linspace(interval[0], interval[1], N+1)
    y = np.zeros((len(y0), N+1))
    b = bvector
    y[:, 0] = y0
    for n in range(N):
        y1 = y[:,n] + h*(np.dot(A,y[:,n])+b(x[n]))    
        y2 = 0.75*y[:,n] +0.25*y1+0.25*h*(np.dot(A,y1)+b(x[n]+h))
        y[:,n+1] = (1/3)*y[:,n]+(2/3)*y2+(2/3)*h*(np.dot(A,y2)+b(x[n]+h))
    return x, y

def dirk3(A, bvector, y0, interval, N):
    """
    The function represent two stage 3rd order diagonally implicit unge-Kutta method for 
    an uniform equispaced value,returns numerical solution of an IVP using N interation on \
    the given interval using N+1 points
    (including endpoints) with initial value y0.
      
      Parameters (Inputs)
    ----------
    A : TYPE - Array
        Input matrix A is nxn matrix with constant coefficients
    
    bvector : TYPE -  Function
        Input bivector function depends only x and output is vector b
    
    y0 : TYPE - Array
        Initial data.
    
    interval : TYPE - List
        A list with start and end values.
    
    N : TYPE - Integer
        Number of steps the algorithm should take.
    
    Return (Outputs)
    -------
    x : TYPE - Array
        An array of the interval coordinate between steps
    y : TYPE - Array
        An array containing the approximate solution for the rk3 method with dimension A
    
    """
    assert N > 1 # Steps must be more than 1
    assert type(A) == np.ndarray # input A only takes in array
    assert len(A) == y0.size # only accept array which fits the dimension  y0
    assert type(N) == int # input N only takes integer
    assert type(interval) == list #input interval only takes list
    
    h = (interval[1] - interval[0]) / N
    x = np.linspace(interval[0], interval[1], N+1)
    y = np.zeros((len(y0), N+1))
    #define constant
    mu = 0.5*(1-1/np.sqrt(3))
    nu = 0.5*(np.sqrt(3)-1)
    gamma = 3/(2*(3+np.sqrt(3)))
    lambd = 1.5*(1+np.sqrt(3))/(3+np.sqrt(3))
    #identity matrix
    I = np.identity(len(A))
    y = np.zeros((len(y0), N+1))
    b = bvector
    y[:, 0] = y0
    M = I - h*mu*A
    for n in range(N):
        y1 = np.dot(np.linalg.inv(M), (y[:,n]+h*mu*b(x[n]+h*mu)))
        y2 = np.dot(np.linalg.inv(M), (y1 +h*nu*(np.dot(A,y1)+ b(x[n]+h*mu)) + h*mu*b(x[n]+h*nu+2*h*mu)))
        y[:,n+1] = (1-lambd)*y[:,n] + lambd*y2+h*gamma*(np.dot(A,y2)+b(x[n]+h*nu+2*h*mu))
    return x, y  





def Q_3():
    """
    Q3 solves the IVP using rk3 and dirk3 in a test case with inputs of 
    a bvector1, interval1, A1, and initial value y0_1 
    Returns 4 plots: figure 1- rk3error vs h
                     figure 2- dirk3error vs h
                     figure 3- max resolution rk3 vs exact solution
                     figure 4- max resolution dirk3 vs exact solution
    """
   
    #defines the problem
    A1 = np.array([[-1000,0], [1000,-1]]) # n x n matrix 1
    interval1 = [0, 0.1] #interval input 1
    y0_1 = np.array([1,0]) #initial value 1
   
    
    Y1 = [] #empty list for rk3 error calculation 
    Y2 = [] #empty list for dirk3 error calculation 
    H1 = [] #empty list for h
    #computes error 
    for i in range(1,11):
        N = 40*i
        h = (interval1[1] - interval1[0]) / N
        x = np.linspace(interval1[0], interval1[1], N+1)
        ye = np.array([np.exp(-1000*x), (1000/999)*(np.exp(-x)-np.exp(-1000*x))]) #Solves exact value
#        Solve rk3
        x, y = rk3(A1, bvector1, y0_1, interval1 , N)
        H1.append(h)
        E =[] 
        for j in range(1,N):
            
            e = abs((y[1, j]-ye[1, j])/ye[1,j]) #calculate error for rk3
            E.append(e)
    
        error1 = h*sum(E)
        Y1.append(error1)
#        solve DIRK3
        x, y = dirk3(A1, bvector1,  y0_1, interval1 , N)
#        error calculation
        E =[] 
        for j in range(1,N):
            
            e = abs((y[1, j]-ye[1, j])/ye[1,j]) #calculate error for dirk3
            E.append(e)

            error2 = h*sum(E)
        Y2.append(error2)
        
    # polyfit to fit the curve
    z = np.polyfit(H1, Y1, 3)    
    p1 = np.poly1d(z)
    plt.figure(1)
    plt.plot(H1, Y1,'x', label='RK3')
    plt.plot(H1, p1(H1),'--', label='fit')
    #plot error normally because log-log error only fits at 6th order
    plt.title('Plot 1 RK3 Error vs h')
    plt.xlabel(r'$step size h$' )
    plt.ylabel(r'$Error$')

    plt.legend()
    plt.grid()
    plt.figure(2)
    #3rd order interpolation
    q = np.polyfit(H1, Y2, 3)    
    p2 = np.poly1d(q)
    # plot error on log10-log10 scale

    plt.plot(np.log10(H1), np.log10(Y2),'*', label='DIRK3')
    plt.plot(np.log10(H1), np.log10(p2(H1)),'--', label='fit')
    plt.legend()
    plt.title('Plot 2 log10 DIRK3 Error vs log10 h')
    plt.xlabel(r'$step size, log10 h$')
    plt.ylabel(r'$Error,log10 $')
    plt.grid()
    

#    Defines the input for max resolution
    N = 400
    h = (interval1[1] - interval1[0]) / N
    A1 = np.array([[-1000,0],[1000,-1]])
    x = np.linspace(interval1[0], interval1[1], N+1)
    #solves exact solution
    ye = np.array([np.exp(-1000*x),1000/999*np.exp(-x)-np.exp(-1000*x)])
     #Solves rk3
    x, y = rk3(A1, bvector1,  y0_1, interval1 , 400)
#    plot numerical vs exact
    plt.figure(3)
    plt.subplot(1, 2, 1) # 1x2 grid, 1st plot
    plt.plot(x, np.log10(y[0]), 'x', color='red', label='RK3')
    plt.plot(x, np.log10(ye[0]), '-', color='blue', label='exact')
    plt.legend()
    plt.title('Highest resolution RK3 vs exact')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$log10 y1$')
    #plt.xlim(-5, 5)
    #plt.ylim(0, 0.85)
    plt.grid()
    #    plot numerical vs exact
    plt.subplot(1, 2, 2) # 1x2 grid, 2nd plot
    plt.plot(x, y[1], 'x', color='red', label='RK3')
    plt.plot(x, ye[1], '-', color='blue', label='exact')
    plt.title('Highest resolution RK3 vs exact')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y2$')
    #plt.xlim(-5, 5)
    #plt.ylim(0, 0.85)
    plt.grid()
    plt.suptitle('PLot 3')
    plt.subplots_adjust(right=1.2)
    
#    solves dirk3
    x, y = dirk3(A1, bvector1,  y0_1, interval1 , 400)
    #    plot numerical vs exact
    plt.figure(4)
    plt.subplot(1, 2, 1) # 1x2 grid, 2nd plot
    plt.plot(x, np.log10(y[0]), '*', color='red', label='DIRK3')
    plt.plot(x, np.log10(ye[0]), '-', color='blue', label='exact')
    plt.title('Highest resolution DIRK3 vs exact')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$log10 y1$')
    plt.grid()
    plt.subplot(1, 2, 2) # 1x2 grid, 2nd plot
    plt.plot(x, y[1], '*', color='red', label='DIRK3')
    plt.plot(x, ye[1], '-', color='blue', label='exact')
    plt.title('Highest resolution DIRK3 vs exact')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y2$')
    plt.grid()
    plt.suptitle('PLot 4')
    plt.subplots_adjust(right=1.2)

def Q_4():
    """
    Q4 solves the IVP using rk3 and dirk3 in a test case with inputs of 
    a bvector2, interval2, A2, and initial value y0_2 
    Returns 3 plots:
                     figure 1- dirk3error vs h
                     figure 2- max resolution rk3 vs exact solution
                     figure 3- max resolution dirk3 vs exact solution
    """ 
#    Defines the problem
    interval2 = [0, 1] #interval input 2
   
    A2 = np.array([[-1, 0, 0], [-99, -100, 0], [-10098, 9900, -10000]])# inputA2
    
    y0_2 = np.array([0, 1, 0]) #initial value 2
    

    Y3 = [] #empty list for rk3 error calculation 
    Y4 = [] #empty list for dirk3 error calculation 
    H2 = [] #empty list for h
     #computes error 
    for i in range(4,17):
        N = 200*i
        h = (interval2[1] - interval2[0]) / N
        x = np.linspace(interval2[0], interval2[1], N+1)
        cos = np.cos(10*x)
        sin = np.sin(10*x)
        #solves rk3
        x, y = rk3(A2, bvector2, y0_2, interval2 , N)
        #solves exact solution
        ye = np.array([cos-np.exp(-x), cos+np.exp(-x)-np.exp(-100*x), sin+2*np.exp(-x)-np.exp(-100*x)-np.exp(-10000*x)])
        H2.append(h)
#        error calculation
        E = [] 
        for j in range(1,N):
            
            e = abs((y[2, j]-ye[2, j])/ye[2,j])
            E.append(e)
    
        error1 = h*sum(E)
        Y3.append(error1)
        #solves dirk3
        x, y = dirk3(A2, bvector2, y0_2, interval2 , N)
        E = [] 
        for j in range(1,N):
            
            e = abs((y[2, j]-ye[2, j])/ye[2,j])
            E.append(e)
            
        error2 = h*sum(E)
        Y4.append(error2)
    #6th order polyfit to interpolate
    q = np.polyfit(H2, Y4, 6)    
    p2 = np.poly1d(q)
    #    plot numericalerror vs h on log10-log10
    plt.figure(5)
    plt.plot(np.log10(H2), np.log(Y4),'x', label='DIRK3')
    plt.plot(np.log10(H2), np.log(p2(H2)),'--', label='fit')
    #plt.plot(np.log10(H2), np.log10(Y1),'x', label='RK3')
    #plt.plot(np.log10(H2), np.log10(p1(H2)),'--', label='fit')
    plt.title('Plot 5 log10 DIRK3 Error vs log10 h')
    plt.xlabel(r'$step size log10(h)$' )
    plt.ylabel(r'$Error, Log10()$')
    #plt.xlabel(r'$step size, log10 h$')
    #plt.ylabel(r'$Error,log10$')
    plt.legend()
    plt.grid()

    #define max resolution
    N = 3200
    h = (interval2[1] - interval2[0]) / N
    x = np.linspace(interval2[0], interval2[1], N+1)
    cos = np.cos(10*x)
    sin = np.sin(10*x)
    #solves exact solution
    ye = np.array([cos-np.exp(-x), cos+np.exp(-x)-np.exp(-100*x), sin+2*np.exp(-x)-np.exp(-100*x)-np.exp(-10000*x)])
    # solve rk3
    x, y = rk3(A2, bvector2, y0_2, interval2, N)
    #    plot numerical vs exact
    plt.figure(6)
    plt.subplot(1, 3, 1) # 1x2 grid, 1st plot
    plt.plot(x, y[0], 'x', color='red', label='RK3')
    plt.plot(x, ye[0], '-', color='blue', label='exact')
    plt.legend()
    plt.suptitle('Plot 6 Highest resolution RK3 vs exact')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y1$')
    plt.grid()
     #    plot numerical vs exact
    plt.subplot(1, 3, 2) # 1x2 grid, 2nd plot
    plt.plot(x, y[1], 'x', color='red', label='RK3')
    plt.plot(x, ye[1], '-', color='blue', label='exact')
#    plt.title('Highest resolution rk3 vs exact')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y2$')
    plt.grid()
     #    plot numerical vs exact
    plt.subplot(1, 3, 3) # 1x2 grid, 2nd plot
    plt.plot(x, y[2], 'x', color='red', label='RK3')
    plt.plot(x, ye[2], '-', color='blue', label='exact')
#    plt.title('Highest resolution rk3 vs exact')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y3$')
    plt.grid()
    plt.subplots_adjust(right=1.5)
   
    #solves dirk3
    x, y = dirk3(A2, bvector2, y0_2, interval2 , N)
     #    plot numerical vs exact
    plt.figure(7)
    plt.subplot(1, 3, 1) # 1x2 grid, 1st plot
    plt.plot(x, y[0], 'x', color='red', label='DIRK3')
    plt.plot(x, ye[0], '-', color='blue', label='exact')
    plt.legend()
    plt.suptitle('Plot 7 Highest resolution DIRK3 vs exact')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y1$')
    plt.grid()
     #    plot numerical vs exact
    plt.subplot(1, 3, 2) # 1x2 grid, 2nd plot
    plt.plot(x, y[1], 'x', color='red', label='DIRK3')
    plt.plot(x, ye[1], '-', color='blue', label='exact')
#    plt.title('Highest resolution rk3 vs exact')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y2$')
    plt.grid()
     #    plot numerical vs exact
    plt.subplot(1, 3, 3) # 1x2 grid, 2nd plot
    plt.plot(x, y[2], 'x', color='red', label='DIRK3')
    plt.plot(x, ye[2], '-', color='blue', label='exact')
#    plt.title('Highest resolution rk3 vs exact')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y3$')
    plt.grid()
    plt.subplots_adjust(right=1.5)


print("Solution to question 3:")
print(Q_3())

print("Solution to question 4:")
print(Q_4())    

