from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


def plot_roc_curve(test_target, y_prob):
    """
    Plots the ROC curve for a given set of true labels and predicted probabilities.
    
    Args:
        test_target (array-like): True labels.
        y_prob (array-like): Predicted probabilities for the positive class.
    """
    # Calculate ROC curve values (False Positive Rate, True Positive Rate)
    fpr, tpr, _ = roc_curve(test_target, y_prob)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(test_target, y_prob))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(conf_matrix, params: dict = {}):
    """
    Plots a confusion matrix as a heatmap.
    
    Args:
        conf_matrix (array-like): The confusion matrix to visualize.
        params (dict, optional): Additional parameters for customization (default is {}).
    """
    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues', **params)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_feature_importance(feature_importances, columns, figsize=(10, 6), params={}):
    """
    Plots the feature importance from a model as a horizontal bar chart.
    
    Args:
        feature_importances (array-like): The importance values of the features.
        columns (list): The names of the features.
        figsize (tuple, optional): The size of the plot (default is (10, 6)).
        params (dict, optional): Additional parameters for customization (default is {}).
    """
    # Create a DataFrame with feature names and their importance values
    feature_importance_df = pd.DataFrame({
        'name': columns,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)
    
    # Plot feature importances
    feature_importance_df.plot(kind='barh', x='name', y='importance', figsize=figsize, **params)
    plt.show()


def evaluate_model(model, test_data, model_name):
    """
    Evaluates a given model on the test dataset and returns both the ROC score and classification report.
    
    Args:
        model (object): The trained Spark model to evaluate.
        test_data (DataFrame): The test dataset.
        model_name (str): The name of the model for reporting.
    
    Returns:
        tuple: A tuple containing the ROC score, classification report, and test scores as pandas DataFrame.
    """
    # Apply the model to the test data
    transformed_test = model.transform(test_data)
    test_scores = transformed_test.select('is_fraud', 'prediction').toPandas()

    # Generate the classification report
    class_report = classification_report(
        y_true=test_scores['is_fraud'],
        y_pred=test_scores['prediction'],
        zero_division=1 if model_name == 'RandomForest' else 'warn'
    )

    # Calculate the ROC score
    evaluator = BinaryClassificationEvaluator(labelCol='is_fraud', metricName='areaUnderROC')
    roc = evaluator.evaluate(transformed_test)

    return roc, class_report, test_scores


def generate_model_results(trained_models, test_data):
    """
    Evaluates all trained models and generates ROC scores and classification reports.
    
    Args:
        trained_models (dict): A dictionary of trained models (model names as keys, models as values).
        test_data (DataFrame): The test dataset to evaluate the models on.
    
    Returns:
        list: A list of dictionaries containing the results for each model.
    """
    model_results = []

    # Loop through each trained model
    for model_name, model in tqdm(trained_models.items(), desc='Generating models results'):
        # Evaluate the current model
        roc, class_report, test_scores = evaluate_model(model, test_data, model_name)

        # Store the evaluation results
        model_results.append({
            'model_name': model_name,
            'roc': roc,
            'classification_report': class_report,
            'test_scores': test_scores
        })

        # Print the results for this model
        print(f"\nArea under ROC curve for {model_name}: {roc:.4f}\n")
        print(f'\nClassification report for {model_name}: \n{class_report}')

    return model_results


def plot_all_roc_curves(model_results):
    """
    Plots the ROC curves for all models evaluated.
    
    Args:
        model_results (list): A list of dictionaries containing evaluation results for each model.
    """
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Loop through each model's results
    for result in model_results:
        model_name = result['model_name']
        y_true = result['test_scores']['is_fraud']
        y_prob = result['test_scores']['prediction']
        
        # Calculate and plot the ROC curve for this model
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_prob):.2f})')

    # Plot the diagonal line for random classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Final plot customization
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.show()