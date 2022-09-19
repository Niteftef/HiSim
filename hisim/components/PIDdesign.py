import numpy as np 
from sympy.integrals.transforms import laplace_transform
import sympy
import math 
import control 
import scipy.signal as sig
from scipy.integrate import odeint
from scipy.signal import ss2tf
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes 
import sympy as sym
from sympy.abc import s,t,x,y,z
from sympy.integrals import laplace_transform
from sympy.integrals import inverse_laplace_transform
from numpy.linalg import inv
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

plt.close("all")
""" Basic simulation parameters """
ns=20000
t=np.linspace(0,ns,ns+1)   # time
u=np.zeros(ns+1)           # input step signal
t_out=np.zeros(ns+1)           # input step signal
for i in range (ns):       
    if i==0:
        u=0
        t_out=0
    else:
        u=22
        t_out=40
        
        


""" state space building ES.ME.SFH.05.Gen.ReEx.001.003 """
A=np.matrix([[-0.5*(0.00107426)]])

B=np.matrix([[0.5*(2.01427888e-06), 0.5*9.34890390e-04, 0.5*1.39367956e-04, 0.5*2.01427888e-06,
         0.5*2.06942017e-06, 0.5*2.23363860e-06]])

C=np.matrix([[0.90179265], [0.92647941]])
D=np.matrix([[0.00062211, 0.05516379, 0.04304355, 0.00062211, 0.00024348,
         0.        ],
        [0.00024348, 0.05667391, 0.01684668, 0.00024348, 0.00025015,
         0.        ]])

Process_gain=B[0,0]/-A[0,0]
t_out_gain=B[0,1]/-A[0,0]
phi_ia_gain=B[0,3]/-A[0,0]
phi_st_gain=B[0,4]/-A[0,0]
phi_m_gain=B[0,5]/-A[0,0]
print("proceess gain is:", Process_gain)
""" state space building  "ES.ME.SFH.04.Gen.ReEx.001.001" """

# A=np.matrix([[-0.5*0.00217474]])
# B=np.matrix([[0.5*1.99199704e-06,2.00152385e-03,1.73214103e-04,1.99199704e-06,2.05743462e-06,2.13276460e-06]])

""" transfer function of T_m to thermal power input """
t_m_initial= 0
tf_tm=control.TransferFunction([B[0,0]], [1,-(A[0,0])])

# open loop step response 
timestep_tm_o, tm_o =control.forced_response(tf_tm,t,u)

#steady state value
tm_steady_state=tm_o[ns]

#value at 63.2%
tm_at_tau=t_m_initial-0.632*(t_m_initial-tm_steady_state)

#find time constant tau_p

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
tm_at_tau_tf_tm=find_nearest(tm_o, tm_at_tau)
for i in range(ns):
    if tm_o[i]==tm_at_tau_tf_tm:
        time_constant_tm=i  
        
# plot open loop step respone 
plt.figure(1)
plt.plot(timestep_tm_o, tm_o,label="tm open loop step response")
plt.legend()

""" transfer function of T_m to t_amb input """
tf_tm_d1=control.TransferFunction([B[0,1]], [1,-(A[0,0])])
print("open loop tranfer function of t_m realtive to t_out", tf_tm_d1)

# open loop step response 
timestep_tm_o, tm_d1_o =control.forced_response(tf_tm_d1,t,t_out)

#steady state value
tm_d1_steady_state=tm_d1_o[ns]

#value at 63.2%
tm_d1_at_tau=t_m_initial-0.632*(t_m_initial-tm_d1_steady_state)

#find time constant tau_p

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
tm_at_tau_tf_tm_d1=find_nearest(tm_d1_o, tm_d1_at_tau)
for i in range(ns):
    if tm_d1_o[i]==tm_at_tau_tf_tm_d1:
        time_constant_tm_d1=i  
        
# plot open loop step respone 
plt.figure(2)
plt.plot(timestep_tm_o, tm_d1_o,label="tm relative to tout open loop step response")
plt.legend()

""" other disturbance """

tf_tm_d2=control.TransferFunction([B[0,2]], [1,-(A[0,0])])
tf_tm_d3=control.TransferFunction([B[0,3]], [1,-(A[0,0])])
tf_tm_d4=control.TransferFunction([B[0,4]], [1,-(A[0,0])])
tf_tm_d5=control.TransferFunction([B[0,5]], [1,-(A[0,0])])
tf_tm_sol=control.TransferFunction([0.00194583117912788], [1861,1])
print(tf_tm_d2)
print("disturbance 3",tf_tm_d3)
print("disturbance 4",tf_tm_d4)
print("disturbance 5",tf_tm_d5)
print(tf_tm_sol)
""" PI Controller Design with Pole Placement for controlling t_m:
    settling time (Ts) = 4/ (damping ratio * natural frequency (omega_n))
    damping frequency (omega_d)=natural frequency * sqrt(1-damping ratio^2)
    
    desired pole = (- natural frequency * damping ratio) +/-  j (natural frequency * sqrt(1-damping ratio^2))
    
    Closed loop transfer function = (transfer function plant * transfer function controller ) / (1+(transfer function plant * transfer function controller ))
    
    simplified Closed loop transfer function of a first order system 1/ms+b= (Kp s + Ki) / ms^2+(b+Kp)s+Ki
    
    
    
    denominator = s^2 + (2 * damping ratio * natural frequency ) s + (natural frequency)^2

"""
# Few assumptions 
settling_time=time_constant_tm*0.5
over_shooting=10

damping_ratio=-np.log(over_shooting/100)/(np.pi**2+(np.log(over_shooting/100))**2)**(1/2)
natural_frequency=4/(settling_time * damping_ratio)
damping_frequency=natural_frequency * np.sqrt(1-damping_ratio**2)

m=1/B[0,0]
b=-A[0,0]/B[0,0]

m_d1=1/B[0,1]
b_d1=-A[0,0]/B[0,1]

m_d2=1/B[0,2]
b_d2=-A[0,0]/B[0,2]


m_d3=1/B[0,3]
b_d3=-A[0,0]/B[0,3]


m_d4=1/B[0,4]
b_d4=-A[0,0]/B[0,4]

m_d5=1/B[0,5]
b_d5=-A[0,0]/B[0,5]

m_sol=1861/0.00194583117912788
b_sol=1/0.00194583117912788

Ki_tm=natural_frequency**2 * m 
Kp_tm=(natural_frequency*damping_ratio*2*m)-b
Kd_tm=0

Ki_tm_d1=natural_frequency**2 * m_d1 
Kp_tm_d1=(natural_frequency*damping_ratio*2*m_d1)-b_d1
Kd_tm_d1=0

Ki_tm_d2=natural_frequency**2 * m_d2
Kp_tm_d2=(natural_frequency*damping_ratio*2*m_d2)-b_d2
Kd_tm_d2=0


Ki_tm_d3=natural_frequency**2 * m_d3 
Kp_tm_d3=(natural_frequency*damping_ratio*2*m_d3)-b_d3
Kd_tm_d3=0

Ki_tm_d4=natural_frequency**2 * m_d4 
Kp_tm_d4=(natural_frequency*damping_ratio*2*m_d4)-b_d4
Kd_tm_d4=0

Ki_tm_d5=natural_frequency**2 * m_d5 
Kp_tm_d5=(natural_frequency*damping_ratio*2*m_d5)-b_d5
Kd_tm_d5=0

Ki_tm_sol=natural_frequency**2 * m_sol
Kp_tm_sol=(natural_frequency*damping_ratio*2*m_sol)-b_sol
Kd_tm_sol=0


# controller designed for P_th
controller_transfer_function_tm=control.TransferFunction([Kd_tm,Kp_tm,Ki_tm], [1, 0])

closed_loop_transfer_function_tm=(controller_transfer_function_tm*tf_tm)/(1+(controller_transfer_function_tm*tf_tm))
poles_closedloop_tm=control.TransferFunction.pole(closed_loop_transfer_function_tm)
print("open loop tranfer function of t_m", tf_tm)
print("closed loop transfer function for t_m", closed_loop_transfer_function_tm)
print("poles of the closed loop transfer function t_m",poles_closedloop_tm)

timestep, tm_closed_loop =control.forced_response(closed_loop_transfer_function_tm,t,u)
plt.figure(3)
plt.plot(timestep, tm_closed_loop,label="closed loop step response tm")
plt.legend()


# controller design for t_amb 
controller_transfer_function_tm_d1=control.TransferFunction([Kd_tm_d1,Kp_tm_d1,Ki_tm_d1], [1, 0])
closed_loop_transfer_function_tm_d1=(controller_transfer_function_tm_d1*tf_tm_d1)/(1+(controller_transfer_function_tm_d1*tf_tm_d1))
poles_closedloop_tm_d1=control.TransferFunction.pole(closed_loop_transfer_function_tm_d1)


timestep, tm_closed_loop_d1 =control.forced_response(closed_loop_transfer_function_tm_d1,t,t_out)
plt.figure(4)
plt.plot(timestep, tm_closed_loop_d1,label="closed loop step response tm realtive to t_out")
plt.legend()

















"""transfer function t_air """

I=np.identity(A.shape[0]) # this is an identity matrix
Ad=inv(I-A)
Bd=Ad*B
X=Ad*0+Bd[0,0]*u
t_air_init=C[0,0]*X+D[0,0]*u
tf_tair=C[0,0]*tf_tm+D[0,0]
poles_openloop_tair=control.TransferFunction.pole(tf_tair)
zeros_openloop_tair=control.TransferFunction.zero(tf_tair)
# open loop step response 
timestep_tair_o, tair_o =control.forced_response(tf_tair,t,u)

#steady state value
tair_steady_state=tair_o[ns]

#value at 63.2%
tair_at_tau=t_air_init-0.632*(t_air_init-tair_steady_state)

#find time constant tau_p

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
tair_at_tau_tf_tair=find_nearest(tair_o, tair_at_tau)
for i in range(ns):
    if tair_o[i]==tair_at_tau_tf_tair:
        time_constant_tair=i  
        
# plot open loop step respone 
plt.figure(5)
plt.plot(timestep_tair_o, tair_o,label="Air Temperature open loop step response")
plt.legend()

""" Pole placemnt to control tair (PI)
Open loop transfer function form: a s + b / c s + d

"""
print("tranfer function of t_air", tf_tair)

a=D[0,0]
b=C[0,0]*B[0,0]+D[0,0]*-A[0,0]
c=1
d=-A[0,0]
Kp_tair=( (2*natural_frequency*damping_ratio*c) - (c*a*natural_frequency**2 /b) - d) / ( (a**2 * natural_frequency**2 /b) + b - (2*natural_frequency*damping_ratio*a))
Ki_tair= natural_frequency**2 *(c+a*Kp_tair)/b
Kd_tair=0
controller_transfer_function_tair=control.TransferFunction([Kd_tair,Kp_tair,Ki_tair], [1, 0])
closed_loop_transfer_function_tair=(controller_transfer_function_tair*tf_tair)/(1+(controller_transfer_function_tair*tf_tair))
poles_closedloop_tair=control.TransferFunction.pole(closed_loop_transfer_function_tair)

timestep, tair_closed_loop =control.forced_response(closed_loop_transfer_function_tair,t,u)
plt.figure(6)
plt.plot(timestep, tair_closed_loop,label="closed loop step response tair")
plt.legend()
