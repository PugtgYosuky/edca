import time
from edca.estimator import PipelineEstimator

from sklearn import set_config
set_config(transform_output='pandas')


def individual_fitness(
        X_train,
        X_val,
        y_train,
        y_val,
        pipeline_config,
        metric,
        individual):
    """
    Calculates the fitness of a given individual

    It instanciates the individual / pipeline based on its configurations and evaluates the predictions
    made.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        Data to train the individual

    X_val: pandas.DataFrame
        Validation data

    y_train: pandas.Series
        Train target

    y_val : pandas.Series
        Validation target (to evaluate the predictions)

    pipeline_config : dict
        Data characteristics by type and components of the fitness (weights)

    metric : function
        Metric used to evaluate the predictions

    individual : dict
        Configuration of the individual

    Returns:
    -------
        tuple
            (float, float, float, float)
            Fitness, Metric result, Train Percentage, CPU time
    """
    # make a copy
    try:
        pipeline_estimator = PipelineEstimator(
            individual_config=individual,
            pipeline_config=pipeline_config
        )
        # start counter
        init_time = time.time()
        # train and predict values
        pipeline_estimator.fit(X_train, y_train)
        preds = pipeline_estimator.predict(X_val)
        proba = pipeline_estimator.predict_proba(X_val)

        # if binary classification, get only the positive class
        if y_train.nunique() == 2:
            proba = proba[:, 1]
        # end counter
        end_time = time.time()

        # calculate metrics
        pred_metric = metric(y_test=y_val, y_pred=preds, y_proba=proba)
        # calculates the percentage of data used based on the instances and
        # features used from the original dataset
        train_percentage = pipeline_estimator.X_train.size / X_train.size
        # get percentage of data used by features and samples
        sample_percentage = pipeline_estimator.X_train.shape[0] / X_train.shape[0]
        feature_percentage = pipeline_estimator.X_train.shape[1] / X_train.shape[1]
        cpu_time = end_time - init_time
        # create fitness
        # alpha * metric + beta * train_percentage + gama * time_spent
        if pipeline_config.get('time_norm', None) is None:
            fit = sum_components(
                pipeline_config=pipeline_config,
                pred_metric=pred_metric,
                train_percentage=train_percentage,
                cpu_time=cpu_time)
        else:
            fit = sum_components_with_normalized_time(
                pipeline_config=pipeline_config,
                pred_metric=pred_metric,
                train_percentage=train_percentage,
                cpu_time=cpu_time)
        return fit, pred_metric, train_percentage, cpu_time, sample_percentage, feature_percentage
    except Exception as e:
        return 1, 1, 1, 1, 1, 1  # in case of an error


def sum_components(pipeline_config, pred_metric, train_percentage, cpu_time):
    """
    Calculates the fitness of the individual based on the different components and the weights associated

    Parameters:
    ----------
    pipeline_config : dict
        Dict with the weights for the components of the fitness

    pred_metric : float
        Prediction metric output, between 0 and 1

    train_percentage : float
        Percentage of train data used, between 0 and 1

    cpu_time : float
        Time the individual required to train, between 0 and infinite

    Returns:
    -------
        float
            Fitness
    """
    fitness = pipeline_config['alpha'] * pred_metric + pipeline_config['beta'] * \
        train_percentage + pipeline_config['gama'] * cpu_time
    return fitness


def sum_components_with_normalized_time(
        pipeline_config,
        pred_metric,
        train_percentage,
        cpu_time):
    """
    Calculates the fitness of the individual based on the different components and the weights associated,
    by normalising the time component of the fitness

    Parameters:
    ----------
    pipeline_config : dict
        Dict witht the weights for the components of the fitness

    pred_metric : float
        Prediction metric output, between 0 and 1

    train_percentage : float
        Percentage of train data used, between 0 and 1

    cpu_time : float
        Time the individual required to train, between 0 and infinite

    Returns:
    -------
        float
            Fitness
    """
    time_component = 1 - (pipeline_config['time_norm'] / (1 + cpu_time))
    fitness = pipeline_config['alpha'] * pred_metric + pipeline_config['beta'] * \
        train_percentage + pipeline_config['gama'] * time_component
    return fitness
