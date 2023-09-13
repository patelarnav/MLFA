

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


dataset = pd.read_csv('Assignment4\data\car_evaluation.csv')

dataset



column_names = ['Price_Buying', 'Price_Maintenance', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Acceptability']
dataset.columns = column_names

enc = OrdinalEncoder()
enc.fit(dataset[column_names])
dataset[column_names] = enc.transform(dataset[column_names])
dataset

X = dataset.copy().drop(['Acceptability'],axis=1)
y = dataset['Acceptability']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class CustomDecisionTree:
    def __init__(self, entropy_threshold=0.0, min_samples_split=2, max_depth=100, n_features=None):
        self.entropy_threshold = entropy_threshold
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
            or self._calculate_entropy(y) <= self.entropy_threshold
        ):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feat_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._find_best_split(X, y, feat_indices)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        left_indices, right_indices = self._split_data(np.array(X)[:, best_feature], best_threshold)
        left = self._grow_tree(np.array(X)[left_indices, :], np.array(y)[left_indices], depth + 1)
        right = self._grow_tree(np.array(X)[right_indices, :], np.array(y)[right_indices], depth + 1)
        return TreeNode(best_feature, best_threshold, left, right)

    def _find_best_split(self, X, y, feat_indices):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feat_index in feat_indices:
            X_column = np.array(X)[:, feat_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._calculate_information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_information_gain(self, y, X_column, threshold):
        parent_entropy = self._calculate_entropy(y)
        left_indices, right_indices = self._split_data(X_column, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        entropy_left, entropy_right = self._calculate_entropy(np.array(y)[left_indices]), self._calculate_entropy(np.array(y)[right_indices])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split_data(self, X_column, threshold):
        left_indices = np.argwhere(X_column <= threshold).flatten()
        right_indices = np.argwhere(X_column > threshold).flatten()
        return left_indices, right_indices

    def _calculate_entropy(self, y):
        unique_labels, label_counts = np.unique(np.array(y), return_counts=True)
        probabilities = label_counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy


    def _most_common_label(self, y):
        unique_labels, label_counts = np.unique(y, return_counts=True)
        most_common_label = unique_labels[np.argmax(label_counts)]
        return most_common_label

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def get_tree_size(self):
        return self._calculate_tree_size(self.root)

    def _calculate_tree_size(self, node):
        if node is None:
            return 0

        left_size = self._calculate_tree_size(node.left)
        right_size = self._calculate_tree_size(node.right)

        return 1 + left_size + right_size

    def find_max_depth(self):
        return self._calculate_max_depth(self.root)

    def _calculate_max_depth(self, node):
        if node is None:
            return 0

        left_depth = self._calculate_max_depth(node.left)
        right_depth = self._calculate_max_depth(node.right)

        return max(left_depth, right_depth) + 1


    def _find_next_node_to_split(self,X,y):
        return self._find_next_node_to_split_recursive(self.root,X,y)

    def _find_next_node_to_split_recursive(self, node, X, y):
        if node is None:
            return None

        # Check if the current node is a leaf node
        if node.is_leaf_node():
            return node

        # Recursively check both left and right subtrees
        left_node = self._find_next_node_to_split_recursive(node.left, X, y)
        right_node = self._find_next_node_to_split_recursive(node.right, X, y)

        # Determine which node to split next based on information gain
        if left_node is None:
            return right_node
        elif right_node is None:
            return left_node
        else:
            # Calculate information gain for both nodes
            information_gain_left = self._calculate_information_gain(y, np.array(X)[:, node.feature], node.threshold)
            information_gain_right = self._calculate_information_gain(y,np.array(X)[:, node.feature], node.threshold)

            # Choose the node with the highest information gain
            return left_node if information_gain_left >= information_gain_right else right_node

    def _split_node(self, node):
        if node is None:
            return

        if node.is_leaf_node():
            # This is a leaf node; no further splitting required
            return

        # Split the node based on its feature and threshold
        feature = node.feature
        threshold = node.threshold

        left_indices, right_indices = self._split_data(X[:, feature], threshold)

        # Create left and right child nodes
        left_child = TreeNode()
        right_child = TreeNode()

        # Assign the left and right child nodes
        node.left = left_child
        node.right = right_child

        # Recursively fit the left and right child nodes
        self._fit_node(left_child, X[left_indices, :], y[left_indices])
        self._fit_node(right_child, X[right_indices, :], y[right_indices])


entropy_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
accuracies = []
tree_sizes = []
max_depths = []
for threshold in entropy_thresholds:
    custom_decision_tree = CustomDecisionTree(entropy_threshold=threshold, min_samples_split=2, max_depth=100)
    custom_decision_tree.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, custom_decision_tree.predict(X_train))
    val_accuracy = accuracy_score(y_val, custom_decision_tree.predict(X_val))

    accuracies.append((train_accuracy, val_accuracy))
    tree_sizes.append(custom_decision_tree.get_tree_size())

    max_depth = custom_decision_tree.find_max_depth()

    max_depth = custom_decision_tree.find_max_depth()
    max_depths.append(max_depth)

# Plot Percentage Accuracy vs. Threshold
train_accuracies = [acc[0] for acc in accuracies]
val_accuracies = [acc[1] for acc in accuracies]

plt.figure(figsize=(8, 5))
plt.bar(entropy_thresholds, train_accuracies, width=0.2, label='Train Accuracy', align='center')
plt.bar([x for x in entropy_thresholds], val_accuracies, width=0.2, label='Validation Accuracy', align='edge')
plt.xlabel('Entropy Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Entropy Threshold')
plt.xticks(entropy_thresholds)
plt.legend()
plt.grid(True)
plt.show()


# Plot Size of Decision Tree vs. Threshold
plt.figure(figsize=(8, 5))
plt.plot(entropy_thresholds, tree_sizes, marker='o')
plt.xlabel('Entropy Threshold')
plt.ylabel('Decision Tree Size')
plt.title('Decision Tree Size vs. Entropy Threshold')
plt.grid(True)
plt.show()

optimal_threshold = 0.5
print(f"We can see the best hyperparameter value of entropy threshold is {optimal_threshold} ")
ind = entropy_thresholds.index(0.5)
optimal_max_depth = max_depths[ind]
optimal_max_depth

optimal_custom_decision_tree = CustomDecisionTree(entropy_threshold=optimal_threshold, min_samples_split=2, max_depth=optimal_max_depth)
optimal_custom_decision_tree.fit(X_train, y_train)

train_accuracy_optimal = accuracy_score(y_train, optimal_custom_decision_tree.predict(X_train))
test_accuracy_optimal = accuracy_score(y_test, optimal_custom_decision_tree.predict(X_test))

print(f"Overall Training Accuracy with Optimal Hyperparameters: {train_accuracy_optimal:.2f}")
print(f"Overall Testing Accuracy with Optimal Hyperparameters: {test_accuracy_optimal:.2f}")



optimal_custom_decision_tree = CustomDecisionTree(entropy_threshold=optimal_threshold, min_samples_split=2, max_depth=optimal_max_depth)
optimal_custom_decision_tree.fit(X_train, y_train)
tree_exp_1 = optimal_custom_decision_tree

best_accuracy = 0.0
best_tree = None
stop_training = False  # Flag to stop training when validation accuracy decreases
num_nodes_when_val_decreases = 0  # Track the number of nodes when validation decreases

while not stop_training:
    # Create a new decision tree
    custom_decision_tree = CustomDecisionTree(entropy_threshold=optimal_threshold, min_samples_split=2, max_depth=optimal_max_depth)
    custom_decision_tree.fit(X_train, y_train)

    # Calculate training accuracy
    train_accuracy = accuracy_score(y_train, custom_decision_tree.predict(X_train))

    # Calculate validation accuracy
    val_accuracy = accuracy_score(y_val, custom_decision_tree.predict(X_val))

    # Check if validation accuracy decreases
    if val_accuracy < best_accuracy:
        stop_training = True  # Stop training if validation accuracy decreases
        num_nodes_when_val_decreases = custom_decision_tree.get_tree_size()  # Record the number of nodes

    else:
        # Update best accuracy and tree
        best_accuracy = val_accuracy
        best_tree = custom_decision_tree

# Analyze the overall testing accuracy for the best tree
test_accuracy = accuracy_score(y_test, best_tree.predict(X_test))
tree_exp_2 = best_tree

# Print results
print(f"Best Validation Accuracy: {best_accuracy}")
print(f"Test Accuracy (using best tree): {test_accuracy}")
print(f"Number of nodes when validation accuracy starts to decrease: {num_nodes_when_val_decreases}")

def print_classification_rules(tree, feature_names, class_labels, parent_rule=None):
    if tree is None:
        return

    if tree.is_leaf_node():
        class_index = int(tree.value)  # Assuming class labels are integers
        class_name = class_labels[class_index]
        rule = f"THEN {class_name}"
        if parent_rule:
            print(f"{parent_rule} AND {rule}")
        else:
            print(f"IF {rule}")
    else:
        feature_index = tree.feature
        threshold = tree.threshold
        feature_name = feature_names[feature_index]
        if parent_rule:
            current_rule = f"{parent_rule} AND {feature_name} <= {threshold:.2f}"
        else:
            current_rule = f"IF {feature_name} <= {threshold:.2f}"
        print_classification_rules(tree.left, feature_names, class_labels, current_rule)
        print_classification_rules(tree.right, feature_names, class_labels, current_rule)

class_labels = ["acc", "good", "unacc", "vgood"]
print("The classification tree from the decsion tree from experiment number 1 is")
print('\n')
print_classification_rules(tree_exp_1.root, column_names[:-1], class_labels)
print('\n')
print("The classification tree from the decsion tree from experiment number 2 is")
print('\n')
print_classification_rules(tree_exp_2.root, column_names[:-1], class_labels)

#Ans2b
optimal_custom_decision_tree = CustomDecisionTree(entropy_threshold=optimal_threshold, min_samples_split=2, max_depth=optimal_max_depth)
optimal_custom_decision_tree.fit(X_train, y_train)
tree_exp_1 = optimal_custom_decision_tree
train_accuracies_step_by_step = []
val_accuracies_step_by_step = []

# Initialize the tree with the root node
custom_decision_tree = CustomDecisionTree(entropy_threshold=optimal_threshold, min_samples_split=2, max_depth=optimal_max_depth)
custom_decision_tree.fit(X_train, y_train)

# Calculate initial accuracy
train_accuracy = accuracy_score(y_train, custom_decision_tree.predict(X_train))
val_accuracy = accuracy_score(y_val, custom_decision_tree.predict(X_val))
train_accuracies_step_by_step.append(train_accuracy)
val_accuracies_step_by_step.append(val_accuracy)

while custom_decision_tree.root is not None:  # Continue splitting until no more nodes to split
    # Find the next node to split
    next_node = custom_decision_tree._find_next_node_to_split(X_train, y_train)

    if next_node is None:
        break  # No more nodes to split, exit the loop

    # Split the node and update the tree
    custom_decision_tree._split_node(next_node)

    # Calculate training accuracy at this point
    train_accuracy = accuracy_score(y_train, custom_decision_tree.predict(X_train))
    train_accuracies_step_by_step.append(train_accuracy)

    # Calculate validation accuracy at this point
    val_accuracy = accuracy_score(y_val, custom_decision_tree.predict(X_val))
    val_accuracies_step_by_step.append(val_accuracy)

# Plot the training and validation accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_accuracies_step_by_step)), train_accuracies_step_by_step, label="Training Accuracy")
plt.plot(range(len(val_accuracies_step_by_step)), val_accuracies_step_by_step, label="Validation Accuracy")
plt.xlabel("Branch Formation")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs. Branch Formation")
plt.show()