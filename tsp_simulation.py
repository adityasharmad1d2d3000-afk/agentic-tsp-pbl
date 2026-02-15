import random
import math
import time
import matplotlib.pyplot as plt




def generate_cities(n, seed=42):
    random.seed(seed)
    return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]

def dist(a, b):
    return math.dist(a, b)

def tour_length(tour, cities):
    total = 0
    for i in range(len(tour)):
        total += dist(cities[tour[i]], cities[tour[(i+1) % len(tour)]])
    return total


# Greedy

def greedy_tsp(cities):
    n = len(cities)
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = tour[-1]
        nxt = min(
            [i for i in range(n) if not visited[i]],
            key=lambda i: dist(cities[last], cities[i])
        )
        visited[nxt] = True
        tour.append(nxt)
    return tour


# Genetic Algorithm

def genetic_tsp(cities, seed=None):
    n = len(cities)

    def random_tour():
        t = list(range(n))
        random.shuffle(t)
        return t

    population = []
    if seed:
        population += [seed[:] for _ in range(5)]
    population += [random_tour() for _ in range(45)]

    for _ in range(120):
        population.sort(key=lambda t: tour_length(t, cities))
        new_pop = population[:10]

        while len(new_pop) < 50:
            p1, p2 = random.sample(population[:20], 2)
            i, j = sorted(random.sample(range(n), 2))
            child = p1[:i] + [x for x in p2 if x not in p1[:i]]
            if random.random() < 0.1:
                a, b = random.sample(range(n), 2)
                child[a], child[b] = child[b], child[a]
            new_pop.append(child)

        population = new_pop

    return population[0]


# RUN EXPERIMENT

cities = generate_cities(25)

# Greedy
t0 = time.time()
greedy = greedy_tsp(cities)
greedy_time = time.time() - t0
greedy_cost = tour_length(greedy, cities)

# GA
t0 = time.time()
ga = genetic_tsp(cities)
ga_time = time.time() - t0
ga_cost = tour_length(ga, cities)

# Agentic Hybrid (Greedy → GA)
t0 = time.time()
hybrid = genetic_tsp(cities, seed=greedy)
hybrid_time = time.time() - t0
hybrid_cost = tour_length(hybrid, cities)

print("\nFINAL RESULTS")
print("Greedy          :", greedy_cost, greedy_time)
print("Genetic Algo    :", ga_cost, ga_time)
print("Agentic Hybrid  :", hybrid_cost, hybrid_time)


# GRAPH

algorithms = ["Greedy", "Genetic Algorithm", "Agentic Hybrid"]
distances = [greedy_cost, ga_cost, hybrid_cost]
times = [greedy_time, ga_time, hybrid_time]

plt.figure()
plt.bar(algorithms, distances)
plt.ylabel("Tour Length")
plt.title("Tour Length Comparison")
plt.show()

plt.figure()
plt.bar(algorithms, times)
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison")
plt.show()

input("Press ENTER to close graphs...")