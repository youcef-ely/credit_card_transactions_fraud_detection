import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.tuning import CrossValidator


class GeneticAlgorithm:
    """
    A class implementing a Genetic Algorithm for hyperparameter optimization of PySpark machine learning models.

    Attributes:
        estimator (object): The PySpark MLlib estimator (e.g., Regression, Classifier) to optimize.
        parameters_ranges (dict): A dictionary containing parameter names as keys and their possible values as lists.
        train_data (DataFrame): Training data as a PySpark DataFrame.
        evaluator (object): PySpark evaluator for model evaluation (e.g., RegressionEvaluator).
        size_of_population (int): The number of chromosomes (individuals) in the population.
        fitness_limit (float): The target fitness value to stop the algorithm.
        time_limit (float): The maximum runtime for the algorithm in minutes.
        probability (float): Probability of mutation for each gene.
        mutation_number (int): Number of mutations per chromosome.
    """

    def __init__(self, estimator, evaluator, parameters_ranges, train_data, size_of_population, 
                 fitness_limit, time_limit, probability=0.1, mutation_number=1):
        """
        Initializes the GeneticAlgorithm class with the given parameters.
        """
        self.estimator = estimator
        self.parameters_ranges = parameters_ranges
        self.train_data = train_data
        self.evaluator = evaluator
        self.size_of_population = size_of_population
        self.probability = probability
        self.mutation_number = mutation_number
        self.fitness_limit = fitness_limit
        self.time_limit = time_limit
        self.scores = []  # Tracks the best score per generation
        self.mean_scores = []  # Tracks the mean score per generation
        self.cached_fitness = {}  # Caches fitness evaluations for chromosomes

    def get_scores(self):
        """
        Returns the list of best scores per generation.
        """
        return self.scores

    def get_mean_scores(self):
        """
        Returns the list of mean scores per generation.
        """
        return self.mean_scores

    def generate_chromosome(self):
        """
        Generates a random chromosome (individual) with parameters sampled from their ranges.
        """
        return {key: random.choice(values) for key, values in self.parameters_ranges.items()}

    def generate_population(self):
        """
        Generates the initial population of chromosomes.
        """
        return [self.generate_chromosome() for _ in range(self.size_of_population)]

    def fitness_function(self, chromosome):
        """
        Evaluates the fitness of a chromosome by performing cross-validation on the estimator.

        Args:
            chromosome (dict): A dictionary representing parameter values for the estimator.

        Returns:
            float: The mean cross-validation score.
        """
        chromosome_tuple = tuple(chromosome.items())
        if chromosome_tuple not in self.cached_fitness:
            self.estimator(**chromosome)
            score = cross_val_score(self.estimator, self.X_train, self.y_train, 
                                    cv=KFold(5, shuffle=True, random_state=3)).mean()
            self.cached_fitness[chromosome_tuple] = score
        return self.cached_fitness[chromosome_tuple]

    def mean_score_population(self, population):
        """
        Calculates the mean fitness score of the population.

        Args:
            population (list): A list of chromosomes.

        Returns:
            float: The mean fitness score of the population.
        """
        return np.mean([self.fitness_function(chromosome) for chromosome in population])

    def sort_population(self, population):
        """
        Sorts the population by fitness in descending order.

        Args:
            population (list): A list of chromosomes.

        Returns:
            list: The sorted population.
        """
        return sorted(population, key=self.fitness_function, reverse=True)

    def selection_pair(self, population):
        """
        Selects the top two chromosomes from the population.

        Args:
            population (list): A list of chromosomes.

        Returns:
            tuple: The two selected chromosomes.
        """
        sorted_population = self.sort_population(population)
        return sorted_population[:2]

    def uniform_crossover(self, parent1, parent2):
        """
        Performs uniform crossover between two parent chromosomes.

        Args:
            parent1 (dict): The first parent chromosome.
            parent2 (dict): The second parent chromosome.

        Returns:
            tuple: Two offspring chromosomes.
        """
        child1, child2 = parent1.copy(), parent2.copy()
        for param in self.parameters_ranges.keys():
            if random.random() < 0.5:
                child1[param], child2[param] = child2[param], child1[param]
        return child1, child2

    def mutation(self, chromosome):
        """
        Applies mutation to a chromosome by randomly changing its parameters.

        Args:
            chromosome (dict): The chromosome to mutate.

        Returns:
            dict: The mutated chromosome.
        """
        mutant = chromosome.copy()
        for _ in range(self.mutation_number):
            if random.random() < self.probability:
                param = random.choice(list(self.parameters_ranges.keys()))
                mutant[param] = random.choice(self.parameters_ranges[param])
        return mutant

    def plot_generations_scores(self):
        """
        Plots the best scores per generation to visualize the optimization process.
        """
        plt.figure(figsize=(15, 10))
        plt.plot(self.scores, label='Max score per generation', color='seagreen')
        plt.xlabel('Generation')
        plt.ylabel('5-Fold CV Score')
        plt.legend()
        plt.show()

    def evolution(self):
        """
        Runs the genetic algorithm to optimize the estimator's hyperparameters.

        Returns:
            list: The final sorted population after optimization.
        """
        start_time = time.time()
        population = self.generate_population()
        population = self.sort_population(population)

        while (time.time() - start_time < self.time_limit * 60 and 
               self.fitness_function(population[0]) < self.fitness_limit):
            # Selection
            parent1, parent2 = self.selection_pair(population)

            # Crossover
            offspring1, offspring2 = self.uniform_crossover(parent1, parent2)

            # Mutation
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)

            # Next generation
            population = self.sort_population(population[:self.size_of_population - 2] + [offspring1, offspring2])

            # Track scores
            best_score = self.fitness_function(population[0])
            self.scores.append(best_score)
            self.mean_scores.append(self.mean_score_population(population))

        return population
