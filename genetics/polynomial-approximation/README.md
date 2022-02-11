# Polynomial Approximation using Genetic Algorithms


```python
import time

from tqdm import tqdm
import numpy as np
import random as r
import math
```


```python
# constants
MIN = -20
MAX = 20
MIN_POWER = -5
MAX_POWER = 5
GENE_SIZE = 2
CHROMOSOME_LENGTH = 6
POPULATION_SIZE = 10
MUTATE_PROBABILITY = 0.5
CROSSOVER_PROBABILITY = 0.7
CROSSOVER_COUNT = 5
limit = 10
GENERATION = 10000
INF = 100000
TEST_RANGE = range(-30, 30)


```


```python
# Calculation
```


```python
def choose_chromosome(pop):
    index = r.randint(0, len(pop) - 1)
    first = pop[index]
    np.delete(pop, [index])
    return first


def expected_func(x):
    return x ** 2


def expected_plt_data():
    xs = []
    ys = []
    for q in TEST_RANGE:
        if q == 0:
            continue
        xs.append(q)
        ys.append(expected_func(q))
    return xs, ys


def actual_func(chromosome, x):
    sum = 0
    y = 0
    l = len(chromosome)
    while y < l:
        coefficient = chromosome[y]
        power = chromosome[y + 1]
        # print(f"c:{coefficient} pow:{power} x:{x}")
        y += GENE_SIZE
        sum += coefficient * np.float_power(x, power)

    return sum


def calculate_fitness(chromosome):
    xs = []
    ys = []
    diff = 0
    for q in TEST_RANGE:
        if q == 0:
            continue
        actual_val = actual_func(chromosome, q)
        xs.append(q)
        ys.append(actual_val)
        diff += math.fabs(expected_func(q) - actual_val)

    # print(f' diff {diff}')
    return min(INF, math.floor(diff)), xs, ys


def rand_gene():
    return r.randint(MIN, MAX), r.randint(MIN_POWER, MAX_POWER)


def generate_population(length, size):
    pop = []
    for i in range(0, size):
        ch = []
        for j in range(0, length):
            coeff, p = rand_gene()
            ch.append(coeff)
            ch.append(p)
        pop.append(ch)
    return np.array(pop)


def mutate(pop):
    count = 0
    for chromosome in pop:
        chance = r.random()
        if chance < MUTATE_PROBABILITY:
            which = r.randint(0, CHROMOSOME_LENGTH - 1)
            coeff, p = rand_gene()
            chromosome[which] = coeff
            chromosome[which + 1] = p
            count += 1
    return count


def crossover(fits, pop):
    offspring = []
    count = 0

    for _ in range(0, CROSSOVER_COUNT):
        chance = r.random()
        if chance < CROSSOVER_PROBABILITY:
            count += 1
            swap_index = r.randint(0, CHROMOSOME_LENGTH - 1) * GENE_SIZE

            first = pop[choose_chromosome(fits)[1]]
            second = pop[choose_chromosome(fits)[1]]
            off1 = np.append(first[0:swap_index], second[swap_index:])
            off2 = np.append(second[0:swap_index], first[swap_index:])
            offspring.append(off1)
            offspring.append(off2)
    return np.array(offspring), count


def select(pop):
    fitness = []
    XS = []
    YS = []
    fit = INF
    for p in range(0, len(pop)):
        n, xs, ys = calculate_fitness(pop[p])
        fitness.append([n, p])
        if n < fit:
            XS = xs
            YS = ys
            fit = n
    fitness.sort(key=lambda row: (row[0]), reverse=False)

    return np.array(fitness[0:POPULATION_SIZE // 2], dtype='uint64'), fit, XS, YS
```


```python
%matplotlib notebook

import matplotlib.pyplot as plt
import time

result_fig = plt.figure(figsize=(5, 5), dpi=80)
result_ax = result_fig.add_subplot(1, 1, 1)
result, = result_ax.plot([0], [0], color='green', label='actual')

expected_xs, expected_ys = expected_plt_data()
result_ax.plot(expected_xs, expected_ys, color='blue', label='expected')
result_ax.legend()

plt.show()

fig = plt.figure(figsize=(10, 3), dpi=80)
ax1 = fig.add_subplot(2, 1, 1)
xs1 = []
ys1 = []

ax2 = fig.add_subplot(2, 1, 2)
xs2 = []
ys2 = []

mutation_count = 0
crossover_count = 0
population = generate_population(CHROMOSOME_LENGTH, POPULATION_SIZE)
for g in tqdm(range(0, GENERATION)):
    # graph =========================
    xs2.append(g)
    ys2.append(len(population))
    # calculation ====================
    best, fit, best_xs, best_ys = select(population)
    indices = best[:, 1]
    offspring, cross_count = crossover(best, population)
    population = np.append(population[indices], offspring, axis=0) if offspring.ndim > 1 else population[indices]
    crossover_count += cross_count
    mutation_count += mutate(population)
    # graph ===================
    if g % 100 == 0:
        result.set_xdata(best_xs)
        result.set_ydata(best_ys)
        result_fig.canvas.draw()
        result_fig.canvas.flush_events()
        time.sleep(0.1)

    xs1.append(g)
    ys1.append(fit)
    # end condition ===============
    if fit < limit or g + 1 == GENERATION:
        print(
            f'generation:{g} best:{fit} mutations:{mutation_count} crossovers:{crossover_count}\nfitness:{best}\nchromosomes:{population[indices]}')
        break

ax1.clear()
ax1.plot(xs1, ys1)

ax2.clear()
ax2.plot(xs2, ys2)
```

![process](https://github.com/mehditeymorian/JNotes/blob/main/genetics/polynomial-approximation/assets/1.gif)

![final](https://github.com/mehditeymorian/JNotes/blob/main/genetics/polynomial-approximation/assets/2.png)

![fitness](https://github.com/mehditeymorian/JNotes/blob/main/genetics/polynomial-approximation/assets/3.png)


      6%|▌         | 552/10000 [00:16<04:38, 33.87it/s]
    

    generation:552 best:7 mutations:3392 crossovers:1945
    fitness:[[  7   0]
     [392   1]
     [392   8]
     [399   5]
     [399   6]]
    chromosomes:[[-7  0 -2 -1  1 19 -2 -1  3  0  4  0]
     [-2  2 -4 -1  1  2  3 -1  3  0  4  0]
     [-7 -2 -4 -1  1  2  3 -1  3  0  4  0]
     [-7  0 -2 -1  1  2  3 -1  3  0  4  0]
     [-7  0 -2 -1  1  2  3 -1  3  0  4  0]]
    




    [<matplotlib.lines.Line2D at 0x27c07e4b8e0>]


