import csv
from numpy.polynomial.polynomial import polyval
from array import array
import model as m
import numpy as np
import matplotlib.pyplot as plt
import random
import time


def extract_data1():
    with open('..\Book1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        x_axis = np.empty([245054], dtype=float)
        y_axis = np.empty([245054], dtype=float)
        label = [0]* 245054
        for row in csv_reader:
            x_axis[line_count] = int(row[0])/25
            y_axis[line_count] = int(row[2])/25
            label[line_count] = row[3]
            line_count += 1
    return np.concatenate((x_axis[1:50000:200], x_axis[150001:200000:200]),
            axis=0), np.concatenate((y_axis[1:50000:200], y_axis[150001:200000:200]),
            axis=0), np.concatenate((label[1:50000:200], label[150001:200000:200]), axis=0)

def genetic(population_no, degree, scope, cross_prob, mut_prob):
    x_axis, y_axis, label = extract_data1()
    polynomial_degree = degree + 1
    population = m.generate_population(polynomial_degree , population_no, scope)
    fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
    max = 0
    counter = 0
    iteration = 0
    while(1):
        population = m.selection(population, fit_result)
        population = m.cross(population, cross_prob, polynomial_degree)
        mutate = m.mutate(population, mut_prob, polynomial_degree)
        fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
        average = m.average(fit_result)
        if average > max:
            counter = 0
            max = average
        else:
            if counter >= 18:
                break
            else:
                counter += 1

        iteration += 1
    best = m.index_of_max(fit_result)
    return population[best], fit_result[best]

def genetic1(population_no, degree, scope, cross_prob, mut_prob):
    x_axis, y_axis, label = extract_data1()
    plt.plot(x_axis[1:250], y_axis[1:250], "go")
    plt.plot(x_axis[250:500], y_axis[250:500], "bo")
    polynomial_degree = degree + 1
    population = m.generate_population(polynomial_degree , population_no, scope)
    fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
    max = 0
    counter = 0
    iteration = 0
    print(m.average(fit_result))
    coefficents = [0] * polynomial_degree
    while(1):
        population = m.selection(population, fit_result)
        population = m.cross(population, cross_prob, polynomial_degree)
        mutate = m.mutate(population, mut_prob, polynomial_degree)
        fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
        average = m.average(fit_result)
        if average > max:
            counter = 0
            max = average
        else:
            if counter >= 8:
                break
            else:
                counter += 1
        ind = m.index_of_max(fit_result)
        coefficents[i] = m.chromosome_to_coefficient(population[ind], polynomial_degree);
        a, b = function(coefficents[i])
        plt.plot(a, b)
        iteration += 1
        print(average)
    plt.legend(loc='upper right')
    plt.show()
    best = m.index_of_max(fit_result)
    return population[best], fit_result[best]

def genetic2(population_no, degree, scope, cross_prob, mut_prob):
    x_axis, y_axis, label = extract_data1()
    polynomial_degree = degree + 1
    population = m.generate_population(polynomial_degree , population_no, scope)
    fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
    max = 0
    counter = 0
    iteration = 0
    print(m.average(fit_result))
    coefficents = [0] * polynomial_degree
    aa = [0] * 100
    bb = [0] * 100

    while(1):
        population = m.selection(population, fit_result)
        population = m.cross(population, cross_prob, polynomial_degree)
        mutate = m.mutate(population, mut_prob, polynomial_degree)
        fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
        average = m.average(fit_result)
        if average > max:
            counter = 0
            max = average
        else:
            if counter >= 8:
                break
            else:
                counter += 1
        ind = m.index_of_max(fit_result)
        aa[iteration] = fit_result[ind]
        bb[iteration] = iteration + 1
        iteration = iteration + 1
        print(average)
    print(aa[1: iteration])
    print(bb[1: iteration])
    plt.plot(bb[1: iteration],aa[1: iteration], "k")
    plt.xlabel("Generation")
    plt.ylabel("Best fitnes function result")
    plt.show()
    best = m.index_of_max(fit_result)
    return population[best], fit_result[best]

def genetic3(population_no, degree, scope, cross_prob, mut_prob):
    x_axis, y_axis, label = extract_data1()
    polynomial_degree = degree + 1
    population = m.generate_population(polynomial_degree , population_no, scope)
    fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
    max = 0
    counter = 0
    iteration = 0
    print(m.average(fit_result))
    coefficents = [0] * polynomial_degree
    aa = [0] * 100
    bb = [0] * 100

    while(1):
        population = m.selection(population, fit_result)
        population = m.cross(population, cross_prob, polynomial_degree)
        mutate = m.mutate(population, mut_prob, polynomial_degree)
        fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
        average = m.average(fit_result)
        if average > max:
            counter = 0
            max = average
        else:
            if counter >= 8:
                break
            else:
                counter += 1
        ind = m.index_of_min(fit_result)
        aa[iteration] = fit_result[ind]
        bb[iteration] = iteration + 1
        iteration = iteration + 1
        print(average)
    print(aa[1: iteration])
    print(bb[1: iteration])
    plt.plot(bb[1: iteration],aa[1: iteration], "k")
    plt.xlabel("Generation")
    plt.ylabel("Worst fitnes function result")
    plt.show()
    best = m.index_of_max(fit_result)
    return population[best], fit_result[best]


def genetic4(population_no, degree, scope, cross_prob, mut_prob):
    x_axis, y_axis, label = extract_data1()
    polynomial_degree = degree + 1
    population = m.generate_population(polynomial_degree , population_no, scope)
    fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
    max = 0
    counter = 0
    iteration = 0
    print(m.average(fit_result))
    coefficents = [0] * polynomial_degree
    aa = [0] * 100
    bb = [0] * 100

    while(1):
        population = m.selection(population, fit_result)
        population = m.cross(population, cross_prob, polynomial_degree)
        mutate = m.mutate(population, mut_prob, polynomial_degree)
        fit_result = m.evaluate(x_axis, y_axis, label, population, polynomial_degree)
        average = m.average(fit_result)
        if average > max:
            counter = 0
            max = average
        else:
            if counter >= 8:
                break
            else:
                counter += 1
        aa[iteration] = m.average(fit_result)
        bb[iteration] = iteration + 1
        iteration = iteration + 1
        print(average)
    print(aa[1: iteration])
    print(bb[1: iteration])
    plt.plot(bb[1: iteration],aa[1: iteration], "k")
    plt.xlabel("Generation")
    plt.ylabel("Average fitnes function result")
    plt.show()
    best = m.index_of_max(fit_result)
    return population[best], fit_result[best]





def function(coefficient):
    coff = [0]*len(coefficient)
    for i in range(len(coefficient)):
        coff[i] = coefficient[(len(coefficient)-i)-1]
    p1 = np.poly1d(coefficient)
    a = []
    b = []
    for x in np.arange(0, 10, 0.1):
        y = p1(x)
        if(y > 17.5):
            break
        if (y < -10):
            break

        a.append(x)
        b.append(y)
    return a, b

def plot1(bests, deg, fit):
    degree  = deg + 1
    x_axis, y_axis, label = extract_data1()
    coefficents = [0] * len(bests)
    plt.plot(x_axis[1:250], y_axis[1:250], "go")
    plt.plot(x_axis[250:500], y_axis[250:500], "bo")
    for i in range(len(bests)):
        coefficents[i] = m.chromosome_to_coefficient(bests[i], degree);
        a, b = function(coefficents[i])
        plt.plot(a, b, label = ('fit func - ' + str(fit[i])))
    plt.legend(loc='upper right')
    plt.show()

def run_hill():
    x_axis, y_axis, label = extract_data1()
    m.generate_hill(100, 2, x_axis, y_axis, label)

def fun1():
    bests = [0b0] * 10
    best_fits = [0] * 10
    population_no = 100
    degree = 5
    scope = 10
    prob_cross = 0.6
    prob_mut = 0.4
    for i in range(10):
        bests[i], best_fits[i] = genetic(population_no, degree, scope, prob_cross, prob_mut)
    print('all avg: ' + str(m.average(best_fits)))
    plot1(bests, degree, best_fits)
if __name__ == "__main__":
    bests = [0b0] * 1
    best_fits = [0] * 1
    population_no = 300
    degree = 3
    scope = 2
    prob_cross = 0.6
    prob_mut = 0.4
    #10 independent runs
    for i in range(1):
        bests[i], best_fits[i] = genetic(population_no, degree, scope, prob_cross, prob_mut)
    # 10 independent runs visualization
    plot1(bests, degree, best_fits)