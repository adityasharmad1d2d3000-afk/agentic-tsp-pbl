
import numpy as np
import random
import matplotlib.pyplot as plt
import time


# Generate cities

def generate_cities(n):
    return np.random.rand(n, 2) * 100

def distance(a, b):
    return np.linalg.norm(a - b)


# Safe distance calculation

def total_distance(route, cities):
    if len(set(route)) != len(route):
        return float('inf')

    dist = 0
    for i in range(len(route)):
        a = cities[route[i]]
        b = cities[route[(i+1) % len(route)]]
        dist += distance(a, b)

    return dist


# GREEDY

def greedy(cities):
    n = len(cities)
    visited = [False]*n
    route = [0]
    visited[0] = True

    for _ in range(n-1):
        last = route[-1]
        next_city = min(
            [i for i in range(n) if not visited[i]],
            key=lambda x: distance(cities[last], cities[x])
        )
        route.append(next_city)
        visited[next_city] = True

    return route


# GA

def crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = [-1]*len(p1)
    child[a:b] = p1[a:b]

    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1

    return child

def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]

def GA(cities, pop_size=50, generations=100, init_route=None):
    n = len(cities)
    population = []

    if init_route is not None:
        population.append(init_route)

    while len(population) < pop_size:
        population.append(random.sample(range(n), n))

    for _ in range(generations):
        population.sort(key=lambda r: total_distance(r, cities))
        new_pop = population[:10]

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:20], 2)
            child = crossover(p1, p2)

            if random.random() < 0.3:
                mutate(child)

            new_pop.append(child)

        population = new_pop

    return population[0]


# ACO

def ACO(cities, init_route=None, ants=20, iterations=50, alpha=1, beta=2, evaporation=0.5):
    n = len(cities)
    pheromone = np.ones((n, n))

    if init_route is not None:
        for i in range(n):
            a = init_route[i]
            b = init_route[(i+1) % n]
            pheromone[a][b] += 5

    best_route = None
    best_dist = float('inf')

    for _ in range(iterations):
        routes = []

        for _ in range(ants):
            route = [random.randint(0, n-1)]

            while len(route) < n:
                i = route[-1]
                probs = []

                for j in range(n):
                    if j not in route:
                        tau = pheromone[i][j]**alpha
                        eta = (1/(distance(cities[i], cities[j])+1e-6))**beta
                        probs.append((j, tau*eta))

                total = sum(p for _, p in probs)
                probs = [(j, p/total) for j,p in probs]

                r = random.random()
                s = 0
                for j,p in probs:
                    s += p
                    if r <= s:
                        route.append(j)
                        break

            d = total_distance(route, cities)
            routes.append((route, d))

            if d < best_dist:
                best_route, best_dist = route, d

        pheromone *= (1 - evaporation)

        for route, d in routes:
            for i in range(n):
                a, b = route[i], route[(i+1) % n]
                pheromone[a][b] += 1/d

    return best_route


# HYBRIDS

def greedy_ga(cities):
    g = greedy(cities)
    return GA(cities, init_route=g)

def ga_aco(cities):
    g = GA(cities)
    return ACO(cities, init_route=g)

def greedy_aco(cities):
    g = greedy(cities)
    return ACO(cities, init_route=g)

def full_hybrid(cities):
    g = greedy(cities)
    ga = GA(cities, init_route=g)
    return ACO(cities, init_route=ga)


# RUN MULTIPLE TIMES (AVERAGING)

runs = 5

methods = {
    "Greedy": greedy,
    "GA": GA,
    "ACO": ACO,
    "Greedy+GA": greedy_ga,
    "GA+ACO": ga_aco,
    "Greedy+ACO": greedy_aco,
    "Full Hybrid": full_hybrid
}

dist_results = {name: 0 for name in methods}
time_results = {name: 0 for name in methods}

for _ in range(runs):
    cities = generate_cities(20)

    for name, func in methods.items():
        start = time.time()
        route = func(cities)
        end = time.time()

        dist_results[name] += total_distance(route, cities)
        time_results[name] += (end - start)

# Average
for name in methods:
    dist_results[name] /= runs
    time_results[name] /= runs


# GRAPH 1: AVG TOUR LENGTH

plt.figure()
plt.bar(dist_results.keys(), dist_results.values())
plt.xticks(rotation=30)
plt.ylabel("Average Tour Length")
plt.title("Average Tour Length Comparison")
plt.show()


# GRAPH 2: AVG EXECUTION TIME

plt.figure()
plt.bar(time_results.keys(), time_results.values())
plt.xticks(rotation=30)
plt.ylabel("Average Execution Time (seconds)")
plt.title("Average Execution Time Comparison")
plt.show()
