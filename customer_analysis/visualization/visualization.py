import json
import matplotlib.pyplot as plt
import os

def plot_modeling_results(metrics):
    """
    Plot F1 scores of the models from the given metrics dictionary.
    """
    for model_name, model_metrics in metrics.items():
        classification_report = model_metrics['classification_report']
        f1_scores = [classification_report[str(label)]["f1-score"] for label in [0, 1]]
        labels = [f"Class {label}" for label in [0, 1]]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, f1_scores, color=['skyblue', 'orange'])
        plt.title(f"F1 Score Comparison for {model_name}", fontsize=14)
        plt.ylabel("F1 Score", fontsize=12)
        plt.ylim(0, 1.1)
        os.makedirs('./graphs', exist_ok=True)
        plt.savefig(f'./graphs/{model_name.replace(" ", "_").lower()}_f1_scores.png')
        plt.show()

def plot_model_comparison():
    """
    Plot a comparison of F1 scores and Accuracy for all models.
    """
    with open('./graphs/logistic_regression_metrics.json') as f:
        lr_metrics = json.load(f)

    with open('./graphs/pca_logistic_regression_metrics.json') as f:
        pca_lr_metrics = json.load(f)

    # Extract metrics
    metrics = {
        "Logistic Regression": {
            "f1_score": lr_metrics["classification_report"]["weighted avg"]["f1-score"],
            "accuracy": lr_metrics["accuracy"]
        },
        "PCA + Logistic Regression": {
            "f1_score": pca_lr_metrics["classification_report"]["weighted avg"]["f1-score"],
            "accuracy": pca_lr_metrics["accuracy"]
        }
    }

    # Data preparation
    categories = list(metrics.keys())
    f1_scores = [metrics[category]["f1_score"] for category in categories]
    accuracies = [metrics[category]["accuracy"] for category in categories]

    # Plot
    x = range(len(categories))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar([p - width/2 for p in x], f1_scores, width=width, label='F1 Score', color='skyblue')
    plt.bar([p + width/2 for p in x], accuracies, width=width, label='Accuracy', color='orange')
    plt.xticks(x, categories, fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Comparison: F1 Score vs Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.ylim(0, 1.1)
    os.makedirs('./graphs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./graphs/model_comparison_f1_accuracy.png')
    plt.show()

def plot_accuracy_comparison():
    """
    Plot the accuracy of Logistic Regression and PCA + Logistic Regression models.
    """
    with open('./graphs/logistic_regression_metrics.json') as f:
        lr_metrics = json.load(f)

    with open('./graphs/pca_logistic_regression_metrics.json') as f:
        pca_lr_metrics = json.load(f)

    # Extract accuracies
    metrics = {
        "Logistic Regression": lr_metrics["accuracy"],
        "PCA + Logistic Regression": pca_lr_metrics["accuracy"]
    }

    # Data preparation
    categories = list(metrics.keys())
    accuracies = list(metrics.values())

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, accuracies, color=['skyblue', 'orange'])
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=12)

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./graphs/model_accuracy_comparison.png')
    plt.show()


def visualize_results():
    """
    Orchestrate the visualization process by loading JSON metrics and calling the plotting functions.
    """
    with open('./graphs/logistic_regression_metrics.json') as f:
        lr_metrics = json.load(f)

    with open('./graphs/pca_logistic_regression_metrics.json') as f:
        pca_lr_metrics = json.load(f)

    metrics = {
        "Logistic Regression": lr_metrics,
        "PCA + Logistic Regression": pca_lr_metrics
    }

    # Generate individual F1 score plots
    plot_modeling_results(metrics)

    # Generate individual accuracy
    plot_accuracy_comparison()

    # Generate overall comparison plot
    plot_model_comparison()

