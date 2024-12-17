import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

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



def evaluate_model(model, test_data, model_name, label_col='is_fraud', prediction_col='prediction'):
    """
    Evaluates a given model on the test dataset and returns both the ROC score and classification report.
    
    Args:
        model (object): The trained Spark model to evaluate.
        test_data (DataFrame): The test dataset.
        label_col (str): The label column name.
        prediction_col (str): The prediction column name.
    
    Returns:
        dict: A dictionary containing the ROC score, classification report, and test scores as a pandas DataFrame.
    """
    try:
        # Apply the model to the test data and cache the result
        transformed_test = model.transform(test_data).cache()
        
        # Extract relevant columns and convert to Pandas DataFrame for sklearn evaluation
        test_scores = transformed_test.select(label_col, prediction_col).toPandas()

        # Generate the classification report
        y_true = test_scores[label_col].tolist()
        y_pred = test_scores[prediction_col].tolist()
        
        class_report = classification_report(y_true, y_pred, zero_division=1 if model_name == 'RandomForest' else 'warn')

        # Calculate the ROC score using Spark evaluator
        evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=prediction_col, metricName='areaUnderROC')
        roc = evaluator.evaluate(transformed_test)

        return {'roc': roc, 'classification_report': class_report, 'test_scores': test_scores}
    
    except Exception as e:
        raise RuntimeError(f"Error during model evaluation for {model_name}: {e}")

def generate_model_results(trained_models, test_data, max_workers=4):
    """
    Evaluates all trained models and generates ROC scores and classification reports using threading.
    
    Args:
        trained_models (dict): A dictionary of trained models (model names as keys, models as values).
        test_data (DataFrame): The test dataset to evaluate the models on.
        max_workers (int): The maximum number of threads to use.
    
    Returns:
        list: A list of dictionaries containing the results for each model.
    """
    model_results = []
    
    def evaluate_model_thread(model_name, model):
        result = evaluate_model(model, test_data, model_name)
        return model_name, result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_model_thread, model_name, model): model_name for model_name, model in trained_models.items()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Generating model results'):
            model_name = futures[future]
            try:
                model_name, result = future.result()
                model_results.append({
                    'model_name': model_name,
                    'roc': result['roc'],
                    'classification_report': result['classification_report']
                })
                
                # Print the results for this model
                print(f"\nArea under ROC curve for {model_name}: {result['roc']:.4f}\n")
                print(f'\nClassification report for {model_name}: \n{result["classification_report"]}')
            except Exception as e:
                print(f"Failed to evaluate model {model_name}: {e}")
    
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