import numpy as np
from scipy.sparse import dok_matrix
from numpy.polynomial.polynomial import polyval
import  matplotlib as mt
import random
import main as ma
import struct

def float_to_bin(num):
    return np.asarray(num, dtype=np.float32).view(np.int32).item()

def bin_to_float(binary):
    binary = format(binary, '032b')
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def chromosome_to_coefficient(chromosome, degree):
    coefficient = [0] * degree
    for x in range(degree):
        bin = (chromosome & (0b11111111111111111111111111111111 << (32)*x)) >> ((32)*x)
        coefficient[x] = bin_to_float(bin)
    return coefficient

def average(lst):
    return sum(lst) / len(lst)

def index_of_max(fit_result):
    m = 0
    for i in range(len(fit_result)):
        if fit_result[i] > m:
            m = i
    return m

def index_of_min(fit_result):
    m = 0
    for i in range(len(fit_result)):
        if fit_result[i] < m:
            m = i
    return m

def fit_function(x, y, label, coefficient):
    lenght = len(label)
    res = 0
    correct = 0;
    p1 = np.poly1d(coefficient)
    for i in range(lenght):
        res = p1(x[i])
        if y[i] >= res and label[i] == '1':

            correct += 1
        if y[i] < res  and label[i] == '2':
            correct += 1
    return correct/lenght

def evaluate(x_axis, y_axis, label, population, polynomial_degree):
    population_lenght = len(population)
    fit_result = [0]*population_lenght
    for i in range(population_lenght):
        coeficients = [0.0] * polynomial_degree
        coeficients = chromosome_to_coefficient(population[i], polynomial_degree)
        fit_result[i] = fit_function(x_axis, y_axis, label, coeficients)
    return fit_result


def selection(population, fit_result):
    population_lenght = len(population)
    calc = [0] * population_lenght
    cycle_whole = sum(fit_result)
    next_generation = [0b0]*population_lenght
    for i in range(population_lenght):
        if i == 0:
            calc[i] = fit_result[i]
        else:
            calc[i] = calc[i-1] + fit_result[i]
    for i in range(population_lenght):
        rul = random.random()*cycle_whole
        if rul <= calc[0] and rul > 0:
            next_generation[i] = population[0]
        else:
            for x in range(population_lenght):
                if rul <= calc[x] and rul > calc[x-1]:
                    next_generation[i] = population[x]

    return next_generation


def cross(population, prob, polynomial_degree):
    population_lenght = len(population)
    is_first_flag = 0
    par1_index = 0
    par1 = 0b0
    par2 = 0b0
    for i in range(population_lenght):
        rand = random.random()
        if rand <= prob:
            if(is_first_flag == 1):
                par2 = population[i]
                position = random.randint(0,polynomial_degree)
                a1 = (par1 << 32*position) >> 32*position
                b1 = (par2 << 32*position) >> 32*position
                a2 = (par2 >> 32*(polynomial_degree-position)) << 32*(polynomial_degree-position)
                b2 = (par2 >> 32*(polynomial_degree-position)) << 32*(polynomial_degree-position)
                population[i] = a1 | b2
                population[par1_index] = a2 | b1
                is_first_flag = 0
            elif (is_first_flag == 0):
                par1 = population[i]
                par1_index = i
                is_first_flag = 1
    return population

def set_bit(v, index, x):
    mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    v &= ~mask          # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask         # If x was True, set the bit indicated by the mask.
    return v

def mutate(population, prob, degree):

    for i in range(len(population)):
        rand = random.random()
        if rand <= prob:
            position = random.randint(0,96)
            var_plus = set_bit(population[i], position, 1)
            if var_plus == population[i]:
                set_bit(population[i], position, 0)
    return population

def generate_population(plynomial_degree, population_strenght, scope):
    population = [0b0] * population_strenght
    for i in range(population_strenght):
        coefficient = 0b0
        member = 0b0
        rand = 0
        for c in range(plynomial_degree):
            sign = random.random()
            if sign >= 0.5:
                rand = (random.random() * scope)*(-1)
            elif sign < 0.5:
                rand = (random.random() * scope)
            coefficient = float_to_bin(rand)
            member <<= 32
            member = member | coefficient
        population[i] = member
    return  population

def float_decimal_to_binary(numb):
    return int(numb)