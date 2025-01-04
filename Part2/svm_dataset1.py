import pickle
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


dataset, labels = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))

model1 = SVC(kernel='linear', C=1)
model2 = SVC(kernel='linear', C=10)
model3 = SVC(kernel='rbf', C=1)
model4 = SVC(kernel='rbf', C=10)

fitted_1 = model1.fit(dataset, labels)
fitted_2 = model2.fit(dataset, labels)
fitted_3 = model3.fit(dataset, labels)
fitted_4 = model4.fit(dataset, labels)


# Function taken from https://scikit-learn.org/dev/auto_examples/svm/plot_svm_kernels.html
def plot_training_data_with_decision_boundary(kernel, clf, X, y, ax=None, long_title=True, support_vectors=True):
    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if long_title:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(kernel)

    plt.show()


# Call the plotting function with the training data
plot_training_data_with_decision_boundary(kernel="linear",  clf=fitted_1, X=dataset, y=labels)
plot_training_data_with_decision_boundary(kernel="linear",  clf=fitted_2, X=dataset, y=labels)
plot_training_data_with_decision_boundary(kernel="rbf", clf=fitted_3, X=dataset, y=labels)
plot_training_data_with_decision_boundary(kernel="rbf", clf=fitted_4, X=dataset, y=labels)
