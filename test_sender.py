import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sender import train_model


def permutation_test(model, X, y, n_permutations=300, cv=5, random_state=42):
    """
    Performs a permutation test to assess the significance of the model's accuracy.
    It computes the actual accuracy using cross-validation with true labels, then shuffles
    the labels n_permutations times and computes the accuracy each time.
    Returns the actual accuracy, an array of permuted accuracies, and the p-value.
    """
    np.random.seed(random_state)

    # Compute the actual accuracy using cross-validation
    actual_scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    actual_accuracy = np.mean(actual_scores)

    permuted_accuracies = []
    for i in range(n_permutations):
        # Shuffle the labels
        y_permuted = np.random.permutation(y)
        # Compute the accuracy with shuffled labels
        scores = cross_val_score(model, X, y_permuted, cv=cv, n_jobs=-1)
        permuted_accuracies.append(np.mean(scores))

        if (i + 1) % 100 == 0:
            print(f"Permutation iteration {i + 1}/{n_permutations} complete")

    permuted_accuracies = np.array(permuted_accuracies)
    p_value = np.mean(permuted_accuracies >= actual_accuracy)
    return actual_accuracy, permuted_accuracies, p_value


def main():
    # Train the sender model using train_model from sender.py
    trained_model, X_train, y_train = train_model()
    print("Starting permutation test for sender...")

    actual_accuracy, permuted_accuracies, p_value = permutation_test(
        trained_model, X_train, y_train, n_permutations=300, cv=5, random_state=42
    )

    print("Permutation Test Results:")
    print("Actual Accuracy:", actual_accuracy)
    print("p-value:", p_value)

    plt.hist(permuted_accuracies, bins=30, alpha=0.7, color='blue')
    plt.axvline(actual_accuracy, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title("Permutation Test: Distribution of Accuracy Scores (Sender)")
    plt.show()


if __name__ == "__main__":
    main()
