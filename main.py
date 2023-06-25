import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the dataset, Breast Cancer Wisconsin (Diagnostic) Data Set
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters of the genetic algorithm
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
MUTATION_RATE = 0.1


def evaluate_fitness(individual):
    """
        This function evaluates the fitness of an individual by calculating the accuracy of the corresponding model
    :param individual:
    :return:
    """
    selected_features_indices = np.nonzero(individual)[0]
    selected_features = X_train[:, selected_features_indices]
    if len(selected_features_indices) == 0:
        return 0
    clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    clf.fit(selected_features, y_train)
    selected_test_features = X_test[:, selected_features_indices]
    y_pred = clf.predict(selected_test_features)
    return accuracy_score(y_test, y_pred)


def generate_population():
    """
        This function generates the initial population
    :return:
    """
    population = []
    for i in range(POPULATION_SIZE):
        individual = [bool(np.random.randint(0, 2)) for _ in range(X.shape[1])]
        population.append(individual)
    return population


def select_parents(population):
    """
        Selection: Tournament selection
        This function selects the parents from the population
    :param population:
    :return:
    """
    parents = []
    for _ in range(2):
        tournament = np.random.choice(np.arange(len(population)), size=5, replace=False)
        tournament_individuals = [population[i] for i in tournament]
        winner = max(tournament_individuals, key=evaluate_fitness)
        parents.append(winner)
    return parents

    # alternatively use Random selection
    # parents = []
    # for i in range(2):
    #     parents.append(population[np.random.randint(0, POPULATION_SIZE)])
    # return parents
    # other selection methods: https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)


def crossover(parent1, parent2):
    """
        Crossover: Single point crossover
        This function performs the crossover operation
    :param parents:
    :return:
    """
    parent1_list = np.array([parent1])
    crossover_point = np.random.randint(1, len(parent1_list))
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

    # alternatively you can use other crossover methods
    # read more here: https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)


def mutate(individual):
    """
        Mutation: Bit flip mutation
        This function performs the mutation operation
    :param individual:
    :return:
    """
    mutated_individual = []
    for gene in individual:
        if np.random.uniform(0, 1) < MUTATION_RATE:
            mutated_individual.append(not gene)
        else:
            mutated_individual.append(gene)
    return mutated_individual

    # alternatively you can use other mutation methods
    # read more here: https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)


def genetic_algorithm():
    """
        This function implements the genetic algorithm
    :return:
    """
    population = generate_population()
    best_fitness = 0
    best_individual = None
    for generation in range(NUM_GENERATIONS):
        for individual in population:
            fitness = evaluate_fitness(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        parent1, parent2 = select_parents(population)
        child = crossover(parent1, parent2)
        mutated_child = mutate(child)
        population = [mutated_child] + list(
            np.random.choice(np.arange(len(population)),
                             size=POPULATION_SIZE - 1,
                             replace=False))

    print("Best individual:", best_individual)
    print("Best fitness:", best_fitness)


# Run the genetic algorithm
genetic_algorithm()