Model: 

Genotype – consist of n IEEE 754 float points (32 bits) numbers in binary representation, where n – polynomial degree

Stop Condition – Genetic algorithm stops when highest average value of fit functions of 7 previous generations is greater 
than average value of fit function of current generation.

Selection – Selection is done base on roulette wheel.

Crossover - It’s done by selecting based on crossover probability n chromosomes then they are paired and crossover is being 
applied. Crossover swaps whole coefficients not a random selected bit from concatenated coefficients in binary form (I checked
both cases and this one is gives better results) . If n is odd number then simply n = n-1.

Mutation - It’s done by selecting based on mutation probability n chromosomes and then mutation is being applied on them.
Mutation change one random selected bit in chromosome.

Fit Function – fit function just simply return the ratio of correct classify point to all points. There are 5000 points: 
2500 point where label equal to 1 and 2500 point where label equal to 0. Because of efficiency some runs are made for 500 points.


In order to run genetic algorithm it’s enough to call genetic(population_no, degree, scope,
cross_prob, mut_prob) function with the following parameters population_no – strength of
population, scope – scope of coefficient where 2 means scope beetwen -2 and 2, cross_prob -
crossover probability, mut_prob –mutation probability

genetic returns to values: solution to the problem in a form of chromosome and value of 
fit function for that chromosome

In order to get the coefficiens it is necessary to clall chromosome_to_coefficient function
wich take two argumens chromosome and degree of considered polynomial.