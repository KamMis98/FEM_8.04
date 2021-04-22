# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:37:28 2021

@author: Kamil
"""

#rozdział 2
2+4
a=2*3
a
print(a)

a = 5
b = 1.234
str = "napis"
a = 1 + 1/2 + 1/3 + \
1/4 + 1/5
print(a)
print("%.4f" % a)

a = 1 + 2 # + 4
print(a)
print("####################")


#rozdział 4
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
#4.1.1
A = np.array([[1, 2, 3], [7, 8, 9]])
print(A)
A = np.array([[1, 2, 3],
[7, 8, 9]])
print(A)
A = np.array([[1, 2, #po backslash’u nie moze byc zadnego znaku!
3],
[7, 8, 9]])
print(A)
#4.1.2
print("####################")
v = np.arange(1,7)
print(v,"\n")
v = np.arange(-2,7)
print(v,"\n")
v = np.arange(1,10,3)
print(v,"\n")
v = np.arange(1,10.1,3)
print(v,"\n")
v = np.arange(1,11,3)
print(v,"\n")
v = np.arange(1,2,0.1)
print(v,"\n")
v = np.linspace(1,3,4)
print(v)
v = np.linspace(1,10,4)
print(v)
X = np.ones((2,3))
Y = np.zeros((2,3,4))
Z = np.eye(2) # np.eye(2,2) # np.eye(2,3)
Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))
print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)
#4.1.3
print("####################")
#U = np.block([[A], [X,Z]])
#print(U)
#4.1.4
print("####################")
V = np.block([[
np.block([np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)

#4.2
print( V[0,2] )
print( V[3,0] )
print( V[3,3] )
print( V[-1,-1] )
print( V[-4,-3] )
print( V[3,:] )
print( V[:,2] )
print( V[3,0:3] )
print( V[np.ix_([0,2,3],[0,-1])] )
print( V[3] )
print("####################")
#4.3
Q = np.delete(V, 2, 0)
print(Q)
Q = np.delete(V, 2, 1)
print(Q)
v = np.arange(1,7)
print( np.delete(v, 3, 0) )
print("####################")
#4.4
np.size(v)
np.shape(v)
np.size(V)
np.shape(V)
print("####################")
#4.5
A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
print( A+B )
print( A-B )
print( A+2 )
print( 2*A )

MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)

MT1 = A*B
print(MT1)
MT2 = B*A
print(MT2)
DT1 = A/B
7
print(DT1)
print("####################")
#4.5.5
C = np.linalg.solve(A,MM1)
print(C) # porownaj z macierza B

x = np.ones((3,1))
b = A@x
y = np.linalg.solve(A,b)
print(y)

PM = np.linalg.matrix_power(A,2) # por. A@A
PT = A**2 # por. A*A

A.T # transpozycja
A.transpose()
A.conj().T # hermitowskie sprzezenie macierzy (dla m. zespolonych)
A.conj().transpose()

A == B
A != B
2 < A
A > B
A < B
A >= B
A <= B
np.logical_not(A)
np.logical_and(A, B)
np.logical_or(A, B)
np.logical_xor(A, B)
print( np.all(A) )
print( np.any(A) )
print( v > 4 )
print( np.logical_or(v>4, v<2))
print( np.nonzero(v>4) )
print( v[np.nonzero(v>4) ] )

print(np.max(A))
print(np.min(A))
print(np.max(A,0))
print(np.max(A,1))
print( A.flatten() )
print( A.flatten("F") )
print("####################")
#rozdział 5
#5.1
import matplotlib.pyplot as plt
x = [1,2,3]
y = [4,6,5]
plt.plot(x,y)
plt.show()
#5.1.1
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y)
plt.show()
#5.1.2
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y,"r:",linewidth=6)
plt.xlabel("Czas")
plt.ylabel("Pozycja")
plt.title("Nasz pierwszy wykres")
plt.grid(True)
plt.show()
print("####################")
#5.1.3
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
plt.plot(x,y1,"r:",x,y2,"g")
plt.legend(("dane y1","dane y2"))
plt.xlabel("Czas")
plt.ylabel("Pozycja")
plt.title("Wykres ")
plt.grid(True)
plt.show()
#5.1.4
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x,y,"b")
l2,l3 = plt.plot(x,y1,"r:",x,y2,"g")
plt.legend((l2,l3,l1),("dane y1","dane y2","y1*y2"))
plt.xlabel("Czas")
plt.ylabel("Pozycja")
plt.title("Wykres ")
plt.grid(True)
plt.show()

#rozdział 6
print("#####Zadanie 3######")
import numpy as np


A1 = np.array([np.linspace(1,5,5),
                np.linspace(5,1,5)])
A2=np.zeros((3,2))
A3=np.ones((2,3))
A3=A3*2
A4=np.linspace(-90,-70,3)
A5=np.ones((5,1))
A5=A5*10

A=np.block([[A3],[A4]])
A=np.block([[A2,A]])
A=np.block([[A1,A]])
A=np.block([[A,A5]])

print(A)

print("##########################################")
print("\n")
print("####Zadanie 4######")
import numpy as np
B1 = np.array([5,4,3,2,1,10])
B2 = np.array([0,0,2,2,2,10]) 
B = B1+B2
print(B)
print("##########################################")

print("\n")
print("####Zadanie 5#####")
C=np.max(A,0)
print(C)
print("##########################################")

print("\n")
print("####Zadanie 6####")
D=np.delete(B,0)
D=np.delete(D,len(D)-1)
print(D)
print("##########################################")

print("\n")
print("####Zadanie 7####")
D[D==4]=0
print(D)
print("##########################################")

print("####Zadanie 8####")
E=np.delete(C,(C==np.max(C)))
E=np.delete(E,(E==np.min(E)))
print(E)
print("##########################################")

print("####Zadanie 9####")
maks=np.max(A)
print(maks)
minim=np.min(A)
print(minim)
print('max')
for i in range(len(A)):
    zm=A[i]
    if np.max(zm)==maks:
        print(zm)
print('min')
for i in range(len(A)):
    zm=A[i]
    if np.min(zm)==minim:
        print(zm)
print("##########################################")
print("####Zadanie 10####")
print(D.flatten()*E.flatten())
print('\n')
n=len(D)
F1=[]
F2=[]
F=[]
for i in range (n):
    if i<(n-2):
        a=D[i+1]*E[i+2]
        F1.append(a)
    elif i<(n-1):
        a=D[i+1]*E[0]
        F1.append(a)
    else:
        a=D[0]*E[1]
        F1.append(a)
        
for i in range (n):
    if i<(n-3):
        a=D[n-1]*E[n-2]
        F2.append(a)
    elif i<(n-2):
        a=D[i-1]*E[n-1]
        F2.append(a)
    else:
        a=D[i-1]*E[i-2]
        F2.append(a)

for i in range (n):
    a=F1[i]-F2[i]
    F.append(a)
print(D)
print(E)
print (F)
print("##########################################")

print("####Zadanie 11####")
     def zadanie11():
    Q=np.round(10*np.random.rand(3,3))
    a=0
    for i in range(len(Q)):
        a=a+Q[i,i]
    return(Q,a)
print(zadanie11())   
print("##########################################")

print("####Zadanie 12####")
def zadanie12(n):
    Q= np.round(10*np.random.rand(n,n))
    for i in range(len(Q)):
        Q[i,i]=0
        Q[(len(Q)-1)-i,i]=0          
    return(Q)
print(zadanie12(3))
print("##########################################")

print("####Zadanie 13####")
def zadanie13(n):
    Q= np.round(10*np.random.rand(n,n))
    F=[]
    f=0
    for i in range(len(Q)):
        if (i%2)!=0:
         F=Q[i]
         for j in range(len(F)):
             f=f+F[j]
    return(Q,f)      
print(zadanie13(3))
print("##########################################")

print("####Zadanie 14####")
def lambada(x):
    x=np.sin(2.0*x)
    return(x)

x = np.arange(-10.0, 10.0, 0.01)
y = lambada(x)
plt.plot(x,y,'r--') 
print("##########################################")

print("###Zadanie 15####")
import zadanie15
x = np.arange(-10,10,0.1)
y = zadanie15.lambada_15(x)
plt.plot(x,y,'g+')
print("##########################################")

print("###Zadanie 16 - z gwiazdką ####")

print("###Zadanie 17####")
import zadanie15
x = np.arange(-10.0, 10.0, 0.01)
y= 3*lambada(x)+zad15.lambada_15(x)
plt.plot(x,y,'b*')
print("##########################################")

print("###Zadanie 18####")
m1=np.array([[10,5,1,7],
              [10,9,5,5],
              [1,6,7,3],
              [10,0,1,5]])
m2=np.array([[34],
              [44],
              [25],
              [27]])
m3=np.linalg.solve(m1,m2)
print(m3)
print("##########################################")

print("###Zadanie 19####")
a = 1000000
x = np.linspace(0, 2*np.pi, a)
y = np.sin(x)
cal = np.sum(2*np.pi/a*y)
print(cal)
print("##########################################")
