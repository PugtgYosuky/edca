import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from edca.evodata import *
from edca.model import *
import os
import json
from edca.evolutionary_algorithm import EvolutionarySearch
from edca.encoder import NpEncoder
from edca.ea import mutation_individuals, mutation_sampling_component, sample_class_balance_mutation, uniform_crossover, points_crossover, generate_sampling_component
from edca.estimator import PipelineEstimator, dataset_analysis
from edca.utils import evo_metric

class DataCentricAutoML(BaseEstimator):
    """ EDCA class """

    def __init__(self, 
                 metric='mcc', 
                 validation_size=0.25,
                 n_iterations=10,
                 time_budget=60,
                 binary_sampling_component=False,
                 automatic_data_optimization=True,
                 use_sampling=False,
                 use_feature_selection=False,
                 use_data_augmentation=False, 
                 prob_mutation=0.3,
                 prob_mutation_model=0.5,
                 prob_crossover=0.7,
                 tournament_size=3,
                 elitism_size=1,
                 population_size=25,
                 alpha=0.5, beta=0.5, gama=0.5,
                 time_norm=None,
                 log_folder_name=None,
                 class_balance_mutation=False,
                 mutation_factor=0.5,
                 uniform_crossover=True,
                 n_jobs=1, 
                 patience=5,
                 early_stop=None,
                 sampling_start=0,
                 mutation_size_neighborhood=20,
                 mutation_percentage_change=0.1,
                 config_models=None,
                 seed=42,
                 ):
        """
        Initialization of the class

        Parameters:
        ----------
        metric : string
            Name of the metric to use. Any of [mcc, f1, accuracy, precision, recall, roc_auc]. Otherwise, it will raise an error.

        validation_size : float
            Percentage of validation data to use on the internal division between train and validation

        n_iterations : integer
            Number of iterations to do of the optimization process. It will be ignored if the time_budget is not None

        binary_sampling_component : bool
            To use the binary sampling component or not

        automatic_data_optimization : bool
            To use the automatic data optimization or not. If False, it will use according use_sampling, use_feature_selection and use_data_augmentation

        use_sampling: bool
            Boolean to indicate of the sampling should be applied or not

        use_feature_selection: bool
            Boolean to indicate of the feature selection should be applied or not

        prob_mutation : float
            Probability of mutation

        prob_mutation_model : float
            Probability of mutation of the model

        prob_crossover : float
            Probability of crossover

        tournament_size : integer
            Tournament size

        elitism_size : integer
            Number of the best individuals to keep from one generation to the next

        population_size : integer
            Size of the population

        alpha, beta, gama: float
            Weights of the different components of the fitness function

        time_norm : integer
            Normalization of the time component of the fitness. Maximum value accepted

        log_folder_name : str
            Directory where to save the information / logs

        class_balance_mutation : bool
            To use balance mutation or not

        mutation_factor : float
            Mutation factor to add to the probabilities fot the class balance mutation

        uniform_crossover : bool
            To use uniform crossover (True) or point crossover (false)

        n_jobs : integer
            Number of parallel workers to use

        patience : integer or None
            Number of generations to wait until restart the population. If None, it will not restart

        early_stop : integer or None
            Number of generations to wait until finish the optimization process without improvement. If None, it will not finish
        
        sampling_start : integer
            Iteration where to start the sampling. By default, it starts in the first iteration

        mutation_size_neighborhood : integer
            Size of the neighborhood to change in the mutation (integer representation)

        mutation_percentage_change : float
            Percentage of the dimension to change in the mutation (integer representation)

        seed : integer
            Seed to use in the process

        config_models : dict or String or None:
            If none, it will use the default models. If string, it will use the path to the json file with the models. If dict, it will use the models in the dict

        Returns:
        -------
            -
        """
        super().__init__()
        self.seed = seed
        self.metric = evo_metric(metric)
        self.validation_size = validation_size
        self.n_iterations = n_iterations
        self.binary_sampling_component = binary_sampling_component
        self.automatic_data_optimization = automatic_data_optimization
        self.use_sampling = use_sampling
        self.use_feature_selection = use_feature_selection
        self.use_data_augmentation = use_data_augmentation
        self.sampling_start = sampling_start
        self.prob_mutation = prob_mutation
        self.prob_mutation_model = prob_mutation_model
        self.prob_crossover = prob_crossover
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.population_size = population_size
        self.time_budget = time_budget  # seconds
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.time_norm = time_norm
        self.class_balance_mutation = class_balance_mutation
        self.uniform_crossover = uniform_crossover
        self.mutation_factor = mutation_factor
        self.log_folder_name = log_folder_name
        self.n_jobs = n_jobs
        self.patience = patience
        self.early_stop = early_stop
        self.mutation_size_neighborhood = mutation_size_neighborhood
        self.mutation_percentage_change = mutation_percentage_change
        self.error_search = False
        # setup search space config
        if isinstance(config_models, str): # should use the defined path
            with open(config_models, 'r') as file:
                self.config_models = json.load(file)
        elif isinstance(config_models, dict): # should use the defined config dict
            self.config_models = config_models
        else:
            # not define -> should use default
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'models.json'), 'r') as file:
                self.config_models = json.load(file)
    def _save(self):
        """ Saves the configuration used, the best individuals found over the iterations and the class distributions of the sampling data"""
        # save json config

        # save best individuals and fitness components
        bests = pd.DataFrame()
        bests['config'] = [json.dumps(config, cls=NpEncoder)
                           for config in self.search_algo.bests_config]
        bests['fitness'] = self.search_algo.bests_fitness
        bests['average'] = self.search_algo.average_fitness
        bests['inv_metric'] = self.search_algo.bests_inv_metric
        bests['train_percentage'] = self.search_algo.bests_train_percentage
        bests['cpu_time'] = self.search_algo.bests_time_cpu
        bests['sample_percentage'] = self.search_algo.bests_sample_percentage
        bests['feature_percentage'] = self.search_algo.bests_feature_percentage
        bests.index = pd.Series(bests.index, name='Iteration') + 1
        bests.to_csv(os.path.join(self.log_folder_name, 'bests.csv'))

        # save target classes used in sampling / training
        self.train_y_sample.name = 'target_class'
        aux = self.train_y_sample.to_frame()
        aux.to_csv(os.path.join(self.log_folder_name, 'train_y.csv'))

        # save internal train and validation data
        aux = self.x_train.copy()
        aux['target_class'] = self.train_y_sample
        aux.to_csv(os.path.join(self.log_folder_name, 'internal_train_data.csv'))

        aux = self.x_val.copy()
        aux['target_class'] = self.val_y
        aux.to_csv(os.path.join(self.log_folder_name, 'internal_val_data.csv'))

        if self.error_search == False:
            # save best data selected
            aux = self.pipeline_estimator.X_train.copy()
            aux['target_class'] = self.pipeline_estimator.y_train
            aux.to_csv(os.path.join(self.log_folder_name, 'best_data.csv'))

            # save best samples data
            aux_x, aux_y = self.pipeline_estimator.get_best_sample_data()
            aux_x['target_class'] = aux_y
            aux_x.to_csv(os.path.join(self.log_folder_name, 'best_sample_data.csv'))

        # save pipeline config
        with open(os.path.join(self.log_folder_name, 'pipeline_config.json'), 'w') as file:
            json.dump(self.pipeline_config, file, cls=NpEncoder, indent=2)

    def fit(self, X_train, y_train):
        """ Function fit the AutoML framework. It divides the data and uses it to find the best pipeline.
        In the end, it retrains and fits the best pipeline
        """
        # save dataset
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()

        # split data into train and validation to search the best config
        self.x_train, self.x_val, self.train_y_sample, self.val_y = train_test_split(
            X_train, y_train, 
            shuffle=True, 
            stratify=y_train, 
            test_size=self.validation_size,
            random_state=self.seed
            )
        self.sampling_size = len(self.x_train)
        self.fs_size = self.x_train.shape[1]

        # analyse the dataset. Uses only the train data no analyse the results
        self.pipeline_config = dataset_analysis(self.x_train)
        # add the sampling config
        self.pipeline_config['sampling'] = self.use_sampling
        self.pipeline_config['feature_selection'] = self.use_feature_selection
        self.pipeline_config['data_augmentation'] = self.use_data_augmentation
        self.pipeline_config['alpha'] = self.alpha
        self.pipeline_config['beta'] = self.beta
        self.pipeline_config['gama'] = self.gama
        self.pipeline_config['time_norm'] = self.time_norm
        self.pipeline_config['sample-start'] = self.sampling_start
        self.pipeline_config['automatic_data_optimization'] = self.automatic_data_optimization
        self.pipeline_config['sampling_size'] = self.sampling_size
        self.pipeline_config['fs_size'] = self.fs_size
        # optimisation process
        self._search_algorithm(
                X_train=self.x_train,
                X_val=self.x_val,
                y_train=self.train_y_sample,
                y_val=self.val_y)

        if self.search_algo.best_individual is None:
            # if an error occurs
            self.error_search = True
        else:
            # gets the best individual and retrains it
            self.pipeline_estimator = PipelineEstimator(
                individual_config=self.search_algo.best_individual,
                pipeline_config=self.pipeline_config
            )

            self.pipeline_estimator.fit(self.x_train, self.train_y_sample)

            # saves the best individual found and other configs
            if self.log_folder_name is not None:
                self._save()
        return self

    def _search_algorithm(self, X_train, X_val, y_train, y_val):
        """ Optimisation process to found the best ML pipeline"""

        # select sample mutation
        if self.class_balance_mutation:
            class_balance_func = sample_class_balance_mutation(
                y_train, self.mutation_factor)
        else:
            class_balance_func = None

        sampling_mutation_operator = mutation_sampling_component(
            prob_mutation=self.prob_mutation,
            dimension=self.sampling_size,
            binary_representation=self.binary_sampling_component,
            size_neighborhood=self.mutation_size_neighborhood,
            max_number_changes=max(1, int(self.sampling_size * self.mutation_percentage_change)),
            class_balance_func=class_balance_func
        )

        fs_mutation_operator = mutation_sampling_component(
            prob_mutation=self.prob_mutation,
            dimension=self.fs_size,
            binary_representation=self.binary_sampling_component,
            size_neighborhood=self.mutation_size_neighborhood,
            max_number_changes=max(1, int(self.fs_size * self.mutation_percentage_change)),
            class_balance_func=None

        )

        # select crossover operator
        if self.uniform_crossover:
            crossover_operator = uniform_crossover(
                binary_representation=self.binary_sampling_component)
        else:
            crossover_operator = points_crossover(
                binary_representation=self.binary_sampling_component)

        sampling_generator = generate_sampling_component(
            binary_representation=self.binary_sampling_component)

        mutation_operator = mutation_individuals(
            prob_mutation=self.prob_mutation,
            prob_mutation_model=self.prob_mutation_model,
            config=self.config_models,
            sample_mutation_operator=sampling_mutation_operator,
            fs_mutation_operator=fs_mutation_operator,
            pipeline_config=self.pipeline_config,
            data_generator = sampling_generator
        )

        # search the best pipeline
        self.search_algo = EvolutionarySearch(
            config_models=self.config_models,
            pipeline_config=self.pipeline_config,
            mutation_operator=mutation_operator,
            crossover_operator=crossover_operator,
            sampling_generator=sampling_generator,
            prob_mutation=self.prob_mutation,
            prob_mutation_model=self.prob_mutation_model,
            prob_crossover=self.prob_crossover,
            population_size=self.population_size,
            tournament_size=self.tournament_size,
            elitism=self.elitism_size,
            num_iterations=self.n_iterations,
            time_budget=self.time_budget,
            filepath=self.log_folder_name,
            components={
                'alpha': self.alpha,
                'beta': self.beta,
                'gama': self.gama
            },
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            fitness_metric=self.metric,
            n_jobs=self.n_jobs,
            patience=self.patience,
            early_stop=self.early_stop
        )

        self.search_algo.evolutionary_algorithm()


    def predict(self, X):
        """ Predicts the test set using the best ML pipeline found during the optimisation process"""
        if self.error_search:
            return None
        preds = self.pipeline_estimator.predict(X)
        return preds

    def predict_proba(self, X):
        """ Predicts the probability of test sample with the best ML pipeline found """
        if self.error_search:
            return None
        preds_proba = self.pipeline_estimator.predict_proba(X)
        return preds_proba

    def get_data_size(self):
        """ Calculates and resturns the data size used """
        return self.pipeline_estimator.X_train.size

    def get_best_individual(self):
        """ Returns the best individual"""
        return self.search_algo.best_individual

    def get_best_data(self):
        """ The only the best samples selected from the optimization process"""
        return self.pipeline_estimator.X_train, self.pipeline_estimator.y_train

    def get_data_shape(self):
        return self.pipeline_estimator.X_train.shape

    def get_best_sample_data(self):
        return self.pipeline_estimator.get_best_sample_data()
