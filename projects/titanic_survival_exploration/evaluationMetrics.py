# In this exercise we'll examine a learner which has high bias, and is incapable of
# learning the patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error. Use plt.plot() within the plot_curve function
# to create line graphs of the values.

from sklearn.linear_model import LinearRegression
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np

size = 1000
cv = KFold(size, shuffle=True)
score = make_scorer(explained_variance_score)

X = np.reshape(np.random.normal(scale=2, size=size), (-1, 1))
y = np.array([[1 - 2 * x[0] + x[0] ** 2] for x in X])

def plot_curve():
    reg = LinearRegression()
    reg.fit(X, y)
    print reg.score(X, y)

    plt.figure()

    train_sizes = np.linspace(.1, 1.0, 5)
    # TODO: Create the learning curve with the cv and score parameters defined above.
    train_sizes, train_scores, test_scores = learning_curve(
        reg, X, y, cv=cv, train_sizes=train_sizes)

    # TODO: Plot the training and testing curves.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    # Sizes the window for readability and displays the plot.
    plt.ylim(-.1, 1.1)
    plt.show()

plot_curve()
