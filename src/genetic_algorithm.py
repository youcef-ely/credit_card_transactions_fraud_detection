import os
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

class GeneticAlgorithm:
    def __init__(self, pipeline, evaluator, parameters_ranges, train_data, size_of_population,
                 score_limit, time_limit, probability=0.1, mutation_number=1, num_folds=5):
        self.pipeline = pipeline.copy()
        self.evaluator = evaluator
        self.parameters_ranges = parameters_ranges
        self.train_data = train_data.cache()  # Cache data for repeated use
        self.size_of_population = size_of_population
        self.fitness_limit = score_limit
        self.time_limit = time_limit * 60  # Convert to seconds
        self.probability = probability
        self.mutation_number = mutation_number
        self.num_folds = num_folds
        self.crossval = self._initialize_crossvalidator()
        self.scores = []  # Tracks the best scores per generation
        self.best_solution = None

    def get_scores(self):
        return self.scores

    def _initialize_crossvalidator(self):
        param_grid = ParamGridBuilder().build()
        return CrossValidator(
            estimator=self.pipeline,
            evaluator=self.evaluator,
            numFolds=self.num_folds,
            estimatorParamMaps=param_grid
        )

    def generate_chromosome(self):
        """Generates a random chromosome."""
        return {key: random.choice(values) for key, values in self.parameters_ranges.items()}

    def generate_population(self):
        """Generates the initial population."""
        return [self.generate_chromosome() for _ in range(self.size_of_population)]

    def fitness_function(self, chromosome):
        """
        Evaluate the fitness of a chromosome by setting its parameters
        and evaluating the model using cross-validation.
        """
        try:
            estimator = self.pipeline.getStages()[-1]
            estimator.setParams(**chromosome)
            self.pipeline.setStages(self.pipeline.getStages()[:-1] + [estimator])
            cv_model = self.crossval.fit(self.train_data)
            return self.evaluator.evaluate(cv_model.bestModel.transform(self.train_data))
        except Exception as e:
            logging.error(f"Error in fitness function for chromosome {chromosome}: {e}")
            return float("-inf")

    def evaluate_population_parallel(self, population):
        """Evaluates the population in parallel to speed up computations."""
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:  # Adjust based on available resources
            results = list(executor.map(self.fitness_function, population))
        return sorted(zip(population, results), key=lambda x: x[1], reverse=True)

    def select_parents(self, population):
        """Selects the top N individuals as parents."""
        return [chrom for chrom, _ in population[:2]]  # Top 2 individuals

    def uniform_crossover(self, parent1, parent2):
        """Performs uniform crossover between two parents."""
        child1, child2 = parent1.copy(), parent2.copy()
        for param in self.parameters_ranges.keys():
            if random.random() < 0.5:
                child1[param], child2[param] = child2[param], child1[param]
        return child1, child2

    def mutate(self, chromosome):
        """Mutates a chromosome by randomly changing its genes."""
        mutant = chromosome.copy()
        for _ in range(self.mutation_number):
            if random.random() < self.probability:
                param = random.choice(list(self.parameters_ranges.keys()))
                mutant[param] = random.choice(self.parameters_ranges[param])
        return mutant

    def run(self):
        """Runs the genetic algorithm to optimize hyperparameters."""
        logging.info("Starting Genetic Algorithm...")
        population = self.generate_population()
        start_time = time.time()
        generation = 0

        while time.time() - start_time < self.time_limit:
            logging.info(f"Generation {generation} - Evaluating population...")
            population_with_scores = self.evaluate_population_parallel(population)
            best_chromosome, best_score = population_with_scores[0]
            self.scores.append(best_score)

            logging.info(f"Generation {generation} - Best score: {best_score:.4f}")

            # Update the best solution
            if self.best_solution is None or best_score > self.best_solution[1]:
                self.best_solution = (best_chromosome, best_score)

            # Check termination criteria
            if best_score >= self.fitness_limit:
                logging.info(f"Fitness limit reached: {best_score:.4f}")
                break

            # Selection
            parents = self.select_parents(population_with_scores)

            # Crossover and Mutation
            offspring = []
            for _ in range(self.size_of_population // 2):  # Create pairs of offspring
                child1, child2 = self.uniform_crossover(parents[0], parents[1])
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))

            # Elitism: Preserve top N parents
            elite = [chrom for chrom, _ in population_with_scores[:len(parents)]]
            population = elite + offspring

            generation += 1

        logging.info(f"Genetic Algorithm completed in {(time.time() - start_time) / 60:.2f} minutes.")
        logging.info(f"Best fitness achieved: {self.best_solution[1]:.4f}")
        return self.best_solution[0]