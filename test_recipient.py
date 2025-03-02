import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from recipient import train_model


def permutation_test(model, X, y, n_permutations=300, cv=5, random_state=42):
    """
    Performs a permutation test to assess the significance of the model's accuracy.
    
    1) Compute the actual accuracy using cross_val_score(model, X, y).
    2) Shuffle labels and compute accuracy again, multiple times.
    3) Calculate p-value as the fraction of permuted accuracies >= actual accuracy.
    """
    np.random.seed(random_state)

    # Actual accuracy (CV) with the pipeline
    actual_scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    actual_accuracy = np.mean(actual_scores)

    permuted_accuracies = []
    for i in range(n_permutations):
        # Shuffle the labels
        y_permuted = np.random.permutation(y)
        # Compute accuracy with shuffled labels
        scores = cross_val_score(model, X, y_permuted, cv=cv, n_jobs=-1)
        permuted_accuracies.append(np.mean(scores))

        # Log progress every 100 permutations
        if (i + 1) % 100 == 0:
            print(f"Permutation iteration {i + 1}/{n_permutations} complete")

    permuted_accuracies = np.array(permuted_accuracies)
    # Fraction of permuted runs that are >= the actual accuracy
    p_value = np.mean(permuted_accuracies >= actual_accuracy)
    return actual_accuracy, permuted_accuracies, p_value


def main():
    # Train/load the pipeline and get train data
    trained_pipeline, X_train, y_train = train_model()  # or load from file if desired
    
    print("Starting permutation test on training data (CV-based)...")
    actual_accuracy, permuted_accuracies, p_value = permutation_test(
        trained_pipeline,
        X_train,
        y_train,
        n_permutations=300,
        cv=5,
        random_state=42
    )

    print("\nPermutation Test Results:")
    print("Actual Accuracy:", actual_accuracy)
    print("p-value:", p_value)

    # Plot the distribution of accuracies from permutations
    plt.figure(figsize=(8,6))
    plt.hist(permuted_accuracies, bins=30, alpha=0.7, color='blue')
    plt.axvline(actual_accuracy, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title("Permutation Test: Distribution of Accuracy Scores")
    plt.show()


if __name__ == "__main__":
    main()
