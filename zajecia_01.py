from scipy import fftpack
import math
# 0:
# print("hello world")

# # # 1:
# a = input("podaj 1 liczbę: ")
# b = input("podaj 2 liczbę: ")
# c = int(a) + int(b)
# print(c)

# 2
# a = input("Podaj liczbę")
# if (int(a) % 2 == 0):
#     print("Liczba jest parzysta")
# else:
#     print("Liczba jest nieparzysta")

# # 3
# a = input("Podaj 1 odcinek: ")
# b = input("Podaj 2 odcinek: ")
# c = input("Podaj 3 odcinek: ")

# if(float(a)+float(b) > float(c) and float(b)+float(c) > float(a) and float(a)+float(c) > float(b)):
#     print("Można zbudować trójkąt")
# else:
#     print("Nie można zbudować trójkątu")

# # 4
# n = int(input("Ile chcesz wpisać liczb? "))
# suma = 0.0
# i = 0
# # while(n > i):
# #     a = float(input("Podaj liczbę: "))
# #     suma = suma+a
# #     i = i+1
# # print(suma/n)

# for i in range(n):
#     a = float(input("Podaj liczbę: "))
#     suma = suma+a
#     i = i+1
# print(suma/n)

# 5
# lista[] = float(input("podaj liczby: "))

# suma = 0.0

# for i in range lista:
#     suma = suma + lista[]
#     suma/range(lista)

# # 6

# a = float(input("Podaj A: "))
# b = float(input("Podaj B: "))
# c = float(input("Podaj C: "))

# delta = b*b-4.0*a*c
# SqrDelta = math.sqrt(delta)

# x1 = (-b - SqrDelta)/2*a
# x2 = (-b + SqrDelta)/2*a

# print(x1)
# print(x2)

# 7
# lines = int(input("Jak duża ma być choinka? "))
# #znak = str(input("Podaj znak: "))
# lenght = lines*2-1
# spaces = (lenght - 1)/2
# i = 1
# while i <= lines:
#     print(" "*(spaces - i + 1), znak*(2*i-1))
#     i = i+1


# Zajęcia 2

# NumPy Vector
# import numpy as np
# list = [1, 2, 3, 4]
# arr = np.array(list)
# print(arr)

# #wypisanie macierzy zer 2x3
# import numpy as np
# print(np.zeros((2, 3)))

# # to samo dla jedynek
# import numpy as np
# print np.ones((2, 3))

# #wypisanie macierzy 1x7 z 7 liczbami od 0 do 6:
# import numpy as np
# print(np.arange(7))

# #wypisanie cyfr od 2 do 9 typu float
# import numpy as np
# arr = np.arange(2, 10, dtype=np.float)
# print(arr)
# print("Array Data Type :", arr.dtype)

# # wypisanie 6 wartości od 1 do 4 z takim samym skokiem
# import numpy as np
# print(np.linspace(1., 4., 6))

# #wypisanie macierzy z wartościami (jak matlab)
# import numpy as np
# print (np.matrix('1 2; 3 4'))

# #tak jak wyżej ale z podmianą zmiennej:
# import numpy as np
# mat = np.matrix('1 2; 3 4')
# print (mat.H)

# # wypisanie macierzy transponowanej
# import numpy as np
# mat = (np.matrix('1 2; 3 4'))
# print(mat.T)


# # generowanie losowych danych w macierzy
# from scipy.cluster.vq import kmeans, vq, whiten
# from numpy import vstack, array
# from numpy.random import rand

# # data generation with three features
# data = vstack((rand(100, 3) + array([.5, .5, .5]), rand(100, 3)))
# print(data)

# # # whitening of data
# # data = whiten(data)
# # print(data)

# # wypisanie tych wartości bez mnożenia przez e00:
# # computing K-Means with K = 3 (2 clusters)
# centroids, _ = kmeans(data, 3)

# # print(centroids)

# # assign each sample to a cluster
# clx, _ = vq(data, centroids)
# # check clusters of observation
# print(clx)


# Stałe

# # Import pi constant from both the packages
# from scipy.constants import pi as pia
# from math import pi as pib

# print("sciPy - pi = %.16f" % pia)
# print("math - pi = %.16f" % pib)

# import scipy.constants
# res = scipy.constants.physical_constants["alpha particle mass"]
# print(res)


# # Transformata Furiera:

# # Wykorzystanie losowych danych do sprawdzenia zmiany dziedziny
# # #Importing the fft and inverse fft functions from fftpackage
# from scipy.fftpack import fft
# import numpy as np

# # create an array with random n numbers
# x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

# # Applying the fft function
# y = fft(x)
# print(y)

# # #FFT is already in the workspace, using the same workspace to for inverse transform

# yinv = scipy.fft.ifft(y)

# print(yinv)

# import numpy as np
# time_step = 0.02
# period = 5.
# time_vec = np.arange(0, 20, time_step)
# sig = np.sin(2 * np.pi / period * time_vec) + \
#     0.5 * np.random.randn(time_vec.size)
# print(sig.size)

# sample_freq = fftpack.fftfreq(sig.size, d=time_step)
# sig_fft = fftpack.fft(sig)
# print(sig_fft)

# from scipy.fftpack import dct
# import numpy as np
# #print(dct(np.array([4., 3., 5., 10., 5., 3.])))

# from scipy.fftpack import idct

# print(idct(np.array([4., 3., 5., 10., 5., 3.])))


# INTEGRATE

# import scipy.integrate
# from numpy import exp
# def f(x): return exp(-x**2)


# i = scipy.integrate.quad(f, 0, 1)
# print(i)

# import scipy.integrate
# from numpy import exp
# from math import sqrt
# def f(x, y): return 16*x*y
# def g(x): return 0
# def h(y): return sqrt(1-4*y**2)


# i = scipy.integrate.dblquad(f, 0, 0.5, g, h)
# print(i)


# # INTERPOLACJA
# # wykres jak w matlabie
# import numpy as np
# import scipy
# from scipy import interpolate
# import matplotlib.pyplot as plt
# x = np.linspace(0, 4, 12)
# y = np.cos(x**2/3+4)
# #print(x, y)
# #plt.plot(x, y, 'o')
# # plt.show()

# f1 = scipy.interpolate.interp1d(x, y, kind='linear')

# f2 = scipy.interpolate.interp1d(x, y, kind='cubic')

# xnew = np.linspace(0, 4, 30)

# plt.plot(x, y, 'o', xnew, f1(xnew), '-', xnew, f2(xnew), '--')

# plt.legend(['data', 'linear', 'cubic', 'nearest'], loc='best')

# plt.show()
