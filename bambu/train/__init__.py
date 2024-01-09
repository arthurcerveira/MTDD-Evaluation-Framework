from bambu.logo import logo
from bambu.models import DecisionTreeEstimator, SvmEstimator, LogisticRegressionEstimator, NeuralNetworkEstimator, GradientBoostingEstimator
from argparse import ArgumentParser
from flaml import AutoML
import pandas as pd
import pickle
import sys

CUSTOM_ESTIMATORS = {
    'decision_tree': DecisionTreeEstimator,
    'svm': SvmEstimator,
    'logistic_regression': LogisticRegressionEstimator,
    'gradient_boosting': GradientBoostingEstimator,
    'neural_network': NeuralNetworkEstimator,
}

AVAILABLE_ESTIMATORS = ['rf', 'extra_tree', *CUSTOM_ESTIMATORS.keys()]

def main():

    if "--list-estimators" in sys.argv:
        for estimator in AVAILABLE_ESTIMATORS:
            print(estimator)
        exit(0)

    print(logo)

    argument_parser = ArgumentParser(prog="bambu-train", description="trains a classification model based on the data generated by bambu-preprocess")
    argument_parser.add_argument('--input-train', required=True, help="path to CSV file containing training set generated by bambu-preprocess")
    argument_parser.add_argument('--output', required=True, help="path to output file containing the trained model")
    argument_parser.add_argument('--estimators', nargs='+', default=['rf'], help="list of estimators to be used by FLAML")
    argument_parser.add_argument('--list-estimators', default=False, action="store_true", help="list all estimators available when using --estimators")
    argument_parser.add_argument('--threads', type=int, default=-1, help="number of parallel jobs to be run by FLAML")
    argument_parser.add_argument('--time-budget', default=None, type=float, help="time limit for model / hiper-parameter optimization search")
    argument_parser.add_argument('--metric', default="roc_auc", choices=["accuracy", "roc_auc", "f1", "micro_f1", "macro_f1"], help="metric to be maximized during hyperparameter optimization")
    argument_parser.add_argument('--max-iter', default=None, type=int, help="max number of iteratins")
    argument_parser.add_argument('--eval-method', default='auto', choices=['auto', 'cv', 'holdout'], help='A string of resampling strategy')
    argument_parser.add_argument('--retrain-full', default=False, action='store_true', help="whether to retrain the selected model on the full training data when using holdout")
    argument_parser.add_argument('--model-history', default=False, action='store_true', help="A boolean of whether to keep the best model per estimator")
    arguments = argument_parser.parse_args()

    unavailable_estimator = False

    for estimator in arguments.estimators:
        if estimator not in AVAILABLE_ESTIMATORS:
            print(f"error: estimator '{estimator}' is not available in Bambu.")
            unavailable_estimator = True
    if unavailable_estimator:
        exit(1)

    train(
        arguments.input_train, 
        arguments.output,
        estimators=arguments.estimators,
        threads=arguments.threads,
        time_budget=arguments.time_budget,
        max_iter=arguments.max_iter,
        metric=arguments.metric,
        eval_method=arguments.eval_method,
        retrain_full=arguments.retrain_full,
        model_history=arguments.model_history
    )

def train(input_train, output, estimators=['rf'], threads=1, time_budget=None, max_iter=None, metric=None, eval_method='auto', retrain_full=False, model_history=False):

    df_train = pd.read_csv(input_train)

    X_train = df_train.drop(['activity'], axis=1)
    y_train = df_train['activity']

    automl = AutoML()

    for estimator_name, estimator_class in CUSTOM_ESTIMATORS.items():
       automl.add_learner(estimator_name, estimator_class)

    automl.fit(
        X_train, y_train, 
        task="classification", 
        estimator_list=estimators, 
        n_jobs=threads, 
        time_budget=time_budget, 
        max_iter=max_iter, 
        metric=metric,
        eval_method=eval_method, 
        retrain_full=retrain_full, 
        model_history=model_history
    )

    with open(output, 'wb') as model_writer:
        model_writer.write(pickle.dumps(automl, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
