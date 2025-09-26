import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split


class DecisionTreeNode:
    """
    Node class for Decision Tree
    """

    def __init__(self):
        self.feature_idx = None  # Index of feature to split on
        self.threshold = None  # Threshold value for split
        self.left = None  # Left child (feature <= threshold)
        self.right = None  # Right child (feature > threshold)
        self.value = None  # Prediction value (for leaf nodes)
        self.samples = 0  # Number of samples in node
        self.impurity = 0  # Impurity measure (gini/entropy)
        self.is_leaf = False  # Whether this is a leaf node


class DecisionTreeAdvanced:
    """
    Comprehensive Decision Tree implementation with:
    - Multiple splitting criteria (Gini, Entropy, MSE)
    - Pruning techniques (pre and post-pruning)
    - Feature importance calculation
    - Visualization capabilities
    """

    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0, max_features=None,
                 task='classification'):
        """
        Parameters:
        - criterion: 'gini', 'entropy' (classification) or 'mse' (regression)
        - max_depth: maximum depth of tree
        - min_samples_split: minimum samples required to split
        - min_samples_leaf: minimum samples required in leaf
        - min_impurity_decrease: minimum impurity decrease for split
        - max_features: number of features to consider per split
        - task: 'classification' or 'regression'
        """
        self.criterion = criterion
        self.max_depth = max_depth # độ sâu cây, cây quá nông thì bias cao, sâu thì variance cao
        self.min_samples_split = min_samples_split # số mẫu tối thiểu để tách 1 node
        self.min_samples_leaf = min_samples_leaf # số mẫu tối thiểu trong 1 lá
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.task = task

        # Fitted attributes
        self.tree_ = None
        self.feature_importances_ = None
        self.n_features_ = None
        self.classes_ = None
        self.n_classes_ = None

    def _calculate_gini(self, y):
        """Calculate Gini impurity: Gini = 1 - Σ(p_i²)"""
        # nhanh, thiên về chọn ra class phổ biến trong node
        # nhạy hơn 1 chút với các class phổ biến
        if len(y) == 0:
            return 0

        counts = np.bincount(y)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _calculate_entropy(self, y):
        # cân bằng, công bằng hơn với các class hiếm
        # chậm hơn gini do dùng log
        # đo lượng thông tin cần để mã hoá dữ liệu
        # nói cách khách là đo số lượng bit cần để mô tả dữ liệu
        """Calculate entropy: H(S) = -Σ(p_i * log2(p_i))"""
        if len(y) == 0:
            return 0

        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)  # Only non-zero probabilities
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _calculate_mse(self, y):
        """Calculate MSE for regression: MSE = (1/n) * Σ(y_i - ȳ)²"""
        # dùng cho bài toán hồi quy
        if len(y) == 0:
            return 0

        mean_y = np.mean(y)
        mse = np.mean((y - mean_y) ** 2)
        return mse

    def _calculate_impurity(self, y):
        """Calculate impurity based on criterion"""
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        elif self.criterion == 'entropy':
            return self._calculate_entropy(y)
        elif self.criterion == 'mse':
            return self._calculate_mse(y)
        else:
            raise ValueError("Invalid criterion")

    def _information_gain(self, y_parent, y_left, y_right):
        """Calculate information gain from a split"""
        # gain giúp làm giảm impurity ở node con
        # gain càng lớn càng tốt
        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n_parent == 0:
            return 0

        # Weighted average of children impurities
        impurity_parent = self._calculate_impurity(y_parent)
        impurity_left = self._calculate_impurity(y_left)
        impurity_right = self._calculate_impurity(y_right)

        weighted_impurity = (n_left / n_parent) * impurity_left + (n_right / n_parent) * impurity_right

        # Information gain
        gain = impurity_parent - weighted_impurity
        return gain

    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        # tìm cấu trúc node tốt nhất
        n_samples, n_features = X.shape

        if self.max_features:
            # nếu có max feature, chọn ngẫu nhiên 1 subset, vì nếu quá nhiều feature thì thời gian tính toán tăng tuyến tính
            feature_indices = np.random.choice(n_features,
                                               size=min(self.max_features, n_features),
                                               replace=False)
        else:
            feature_indices = np.arange(n_features)

        best_gain = -1
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]

            # Try different thresholds (unique values)
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                # Split samples
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Skip if split doesn't meet minimum samples requirement
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calculate information gain
                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = self._information_gain(y, y_left, y_right)

                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = np.where(left_mask)[0]
                    best_right_indices = np.where(right_mask)[0]

        return best_feature, best_threshold, best_gain, best_left_indices, best_right_indices

    def _create_leaf(self, y):
        """Create a leaf node with prediction value"""
        node = DecisionTreeNode()
        node.is_leaf = True
        node.samples = len(y)
        node.impurity = self._calculate_impurity(y)

        if self.task == 'classification':
            # Most common class
            node.value = Counter(y).most_common(1)[0][0]
        else:
            # Mean for regression
            node.value = np.mean(y)

        return node

    def _build_tree(self, X, y, depth=0, sample_indices=None):
        """Recursively build the decision tree"""
        if sample_indices is None:
            sample_indices = np.arange(len(X))

        X_subset = X[sample_indices]
        y_subset = y[sample_indices]
        n_samples = len(y_subset)

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
                (n_samples < self.min_samples_split) or \
                (len(np.unique(y_subset)) == 1):  # Pure node
            return self._create_leaf(y_subset)

        # Find best split
        best_feature, best_threshold, best_gain, left_indices, right_indices = \
            self._find_best_split(X_subset, y_subset)

        # If no valid split found or gain too small
        if best_feature is None or best_gain < self.min_impurity_decrease:
            return self._create_leaf(y_subset)

        # Create internal node
        node = DecisionTreeNode()
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.samples = n_samples
        node.impurity = self._calculate_impurity(y_subset)

        # Convert local indices to global indices
        global_left_indices = sample_indices[left_indices]
        global_right_indices = sample_indices[right_indices]

        # Recursively build children
        node.left = self._build_tree(X, y, depth + 1, global_left_indices)
        node.right = self._build_tree(X, y, depth + 1, global_right_indices)

        return node

    def fit(self, X, y):
        """Fit the decision tree"""
        self.n_features_ = X.shape[1]

        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            # Encode labels as integers
            y_encoded = np.searchsorted(self.classes_, y)
        else:
            y_encoded = y.copy()

        # Build tree
        self.tree_ = self._build_tree(X, y_encoded)

        # Calculate feature importances
        self._calculate_feature_importance(X, y_encoded)

        return self

    def _predict_sample(self, x, node):
        """Predict a single sample by traversing the tree"""
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        """Make predictions for samples"""
        predictions = []

        for x in X:
            pred = self._predict_sample(x, self.tree_)

            if self.task == 'classification':
                # Convert back to original labels
                pred = self.classes_[pred]

            predictions.append(pred)

        return np.array(predictions)

    def _calculate_feature_importance(self, X, y):
        """Calculate feature importance based on impurity decrease"""
        importances = np.zeros(self.n_features_)

        def traverse_tree(node, n_total_samples):
            if node.is_leaf:
                return

            # Calculate weighted impurity decrease
            impurity_decrease = node.samples / n_total_samples * node.impurity

            if node.left:
                impurity_decrease -= (node.left.samples / n_total_samples) * node.left.impurity
            if node.right:
                impurity_decrease -= (node.right.samples / n_total_samples) * node.right.impurity

            importances[node.feature_idx] += impurity_decrease

            # Recursively process children
            if node.left:
                traverse_tree(node.left, n_total_samples)
            if node.right:
                traverse_tree(node.right, n_total_samples)

        traverse_tree(self.tree_, len(X))

        # Normalize importances
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)

        self.feature_importances_ = importances

        return importances

    def print_tree(self, node=None, depth=0, prefix=""):
        """Print tree structure"""
        if node is None:
            node = self.tree_

        if node.is_leaf:
            print(f"{prefix}├── Leaf: value={node.value:.3f}, samples={node.samples}, impurity={node.impurity:.3f}")
        else:
            print(f"{prefix}├── Feature_{node.feature_idx} <= {node.threshold:.3f}")
            print(f"{prefix}│   samples={node.samples}, impurity={node.impurity:.3f}")

            # Print children
            if node.left:
                print(f"{prefix}│   Left:")
                self.print_tree(node.left, depth + 1, prefix + "│   ")
            if node.right:
                print(f"{prefix}│   Right:")
                self.print_tree(node.right, depth + 1, prefix + "│   ")

    def get_depth(self, node=None):
        """Get depth of the tree"""
        if node is None:
            node = self.tree_

        if node.is_leaf:
            return 1

        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0

        return 1 + max(left_depth, right_depth)

    def get_n_leaves(self, node=None):
        """Get number of leaves in the tree"""
        if node is None:
            node = self.tree_

        if node.is_leaf:
            return 1

        n_leaves = 0
        if node.left:
            n_leaves += self.get_n_leaves(node.left)
        if node.right:
            n_leaves += self.get_n_leaves(node.right)

        return n_leaves


class RandomForest:
    """
    Random Forest implementation from scratch
    """

    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, bootstrap=True, oob_score=False,
                 random_state=None, task='classification'):
        """
        Parameters:
        - n_estimators: number of trees
        - max_features: number of features to consider per split
        - bootstrap: whether to use bootstrap sampling
        - oob_score: whether to calculate out-of-bag score
        """
        self.n_estimators = n_estimators
        self.max_features = max_features # số feature được xét khi split -> giảm variance
        self.max_depth = max_depth # depth nhỏ -> bias cao, lớn -> variance cao
        self.min_samples_split = min_samples_split # số mẫu tối thiểu để tách 1 node
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.task = task

        # Fitted attributes
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_predictions_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _get_max_features(self, n_features):
        """Calculate number of features to use per tree"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features

    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample of the data"""
        n_samples = X.shape[0]

        if self.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        else:
            # Use all samples
            indices = np.arange(n_samples)
            oob_indices = np.array([])

        return X[indices], y[indices], indices, oob_indices

    def fit(self, X, y):
        """Fit Random Forest"""
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        self.estimators_ = []
        feature_importances = np.zeros(n_features)

        # For OOB score calculation
        if self.oob_score:
            oob_predictions_sum = np.zeros(n_samples)
            oob_counts = np.zeros(n_samples)

        print(f" Training {self.n_estimators} trees with max_features={max_features}")

        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_boot, y_boot, boot_indices, oob_indices = self._bootstrap_sample(X, y)

            # Create and train decision tree
            tree = DecisionTreeAdvanced(
                criterion='gini' if self.task == 'classification' else 'mse',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                task=self.task
            )

            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

            # Accumulate feature importances
            feature_importances += tree.feature_importances_

            # Calculate OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions_sum[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1

            if (i + 1) % 20 == 0 or i == 0:
                print(f"   Trained {i + 1}/{self.n_estimators} trees")

        # Average feature importances
        self.feature_importances_ = feature_importances / self.n_estimators

        # Calculate OOB score
        if self.oob_score:
            oob_mask = oob_counts > 0
            if np.sum(oob_mask) > 0:
                self.oob_predictions_ = np.zeros(n_samples)
                self.oob_predictions_[oob_mask] = oob_predictions_sum[oob_mask] / oob_counts[oob_mask]

                if self.task == 'classification':
                    self.oob_score_ = np.mean(y[oob_mask] == self.oob_predictions_[oob_mask])
                else:
                    self.oob_score_ = -np.mean((y[oob_mask] - self.oob_predictions_[oob_mask]) ** 2)

        return self

    def predict(self, X):
        """Make predictions using all trees"""
        if self.task == 'classification':
            # Majority voting
            predictions = np.array([tree.predict(X) for tree in self.estimators_])

            # Get mode for each sample
            final_predictions = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                final_predictions.append(Counter(votes).most_common(1)[0][0])

            return np.array(final_predictions)

        else:
            # Average predictions for regression
            predictions = np.array([tree.predict(X) for tree in self.estimators_])
            return np.mean(predictions, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities (classification only)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        # Get all unique classes from first tree
        classes = self.estimators_[0].classes_
        n_classes = len(classes)
        n_samples = X.shape[0]

        # Initialize probability matrix
        probabilities = np.zeros((n_samples, n_classes))

        # Collect votes from all trees
        for tree in self.estimators_:
            predictions = tree.predict(X)
            for i, pred in enumerate(predictions):
                class_idx = np.where(classes == pred)[0][0]
                probabilities[i, class_idx] += 1

        # Normalize to get probabilities
        probabilities = probabilities / self.n_estimators

        return probabilities


def generate_classification_dataset(n_samples=1000, n_features=10, n_informative=5,
                                    n_redundant=2, n_classes=3, random_state=42):
    """Generate synthetic classification dataset"""
    np.random.seed(random_state)

    # Generate informative features
    X_informative = np.random.randn(n_samples, n_informative)

    # Create class centers
    centers = np.random.randn(n_classes, n_informative) * 2

    # Assign samples to classes based on distance to centers
    distances = np.array([np.sum((X_informative - center) ** 2, axis=1) for center in centers])
    y = np.argmin(distances, axis=0)

    # Add some noise to make it more realistic
    noise_factor = 0.1
    for class_idx in range(n_classes):
        mask = y == class_idx
        X_informative[mask] += noise_factor * np.random.randn(np.sum(mask), n_informative)

    # Add redundant features (linear combinations)
    X_redundant = np.zeros((n_samples, n_redundant))
    for i in range(n_redundant):
        # Random linear combination of informative features
        weights = np.random.randn(n_informative)
        X_redundant[:, i] = X_informative @ weights + 0.1 * np.random.randn(n_samples)

    # Add random noise features
    n_noise = n_features - n_informative - n_redundant
    X_noise = np.random.randn(n_samples, n_noise)

    # Combine all features
    X = np.hstack([X_informative, X_redundant, X_noise])

    return X, y


def plot_decision_tree_results(X, y, tree, title="Decision Tree Results"):
    """Visualize decision tree results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Feature importances
    if tree.feature_importances_ is not None:
        feature_names = [f'Feature_{i}' for i in range(len(tree.feature_importances_))]
        importance_indices = np.argsort(tree.feature_importances_)[::-1][:10]  # Top 10

        axes[0, 0].barh(range(len(importance_indices)),
                        tree.feature_importances_[importance_indices])
        axes[0, 0].set_yticks(range(len(importance_indices)))
        axes[0, 0].set_yticklabels([feature_names[i] for i in importance_indices])
        axes[0, 0].set_xlabel('Feature Importance')
        axes[0, 0].set_title('Top 10 Feature Importances')
        axes[0, 0].invert_yaxis()

    # Plot 2: Tree statistics
    tree_stats = {
        'Depth': tree.get_depth(),
        'Leaves': tree.get_n_leaves(),
        'Features Used': np.sum(tree.feature_importances_ > 0)
    }

    axes[0, 1].bar(tree_stats.keys(), tree_stats.values())
    axes[0, 1].set_title('Tree Statistics')
    axes[0, 1].set_ylabel('Count')

    # Plot 3: Class distribution (if classification)
    if tree.task == 'classification':
        unique_classes, counts = np.unique(y, return_counts=True)
        axes[1, 0].pie(counts, labels=[f'Class {c}' for c in unique_classes], autopct='%1.1f%%')
        axes[1, 0].set_title('Class Distribution')
    else:
        axes[1, 0].hist(y, bins=20, alpha=0.7)
        axes[1, 0].set_xlabel('Target Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Target Distribution')

    # Plot 4: Predictions vs Actual (if 2D data for visualization)
    if X.shape[1] >= 2:
        predictions = tree.predict(X)
        if tree.task == 'classification':
            # Use first two features for visualization
            scatter = axes[1, 1].scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis', alpha=0.7)
            axes[1, 1].set_xlabel('Feature 0')
            axes[1, 1].set_ylabel('Feature 1')
            axes[1, 1].set_title('Predictions (2D Projection)')
            plt.colorbar(scatter, ax=axes[1, 1])
        else:
            axes[1, 1].scatter(y, predictions, alpha=0.7)
            axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Actual')
            axes[1, 1].set_ylabel('Predicted')
            axes[1, 1].set_title('Predictions vs Actual')

    plt.tight_layout()
    plt.show()


def compare_tree_vs_forest(X_train, X_test, y_train, y_test):
    """Compare single decision tree vs random forest"""
    print("\n DECISION TREE vs RANDOM FOREST COMPARISON")
    print("=" * 60)

    # Single Decision Tree
    single_tree = DecisionTreeAdvanced(
        criterion='gini',
        max_depth=None,  # No limit
        min_samples_split=2,
        task='classification'
    )
    single_tree.fit(X_train, y_train)

    tree_train_pred = single_tree.predict(X_train)
    tree_test_pred = single_tree.predict(X_test)
    tree_train_acc = np.mean(tree_train_pred == y_train)
    tree_test_acc = np.mean(tree_test_pred == y_test)

    # Random Forest
    forest = RandomForest(
        n_estimators=100,
        max_features='sqrt',
        max_depth=None,
        oob_score=True,
        task='classification',
        random_state=42
    )
    forest.fit(X_train, y_train)

    forest_train_pred = forest.predict(X_train)
    forest_test_pred = forest.predict(X_test)
    forest_train_acc = np.mean(forest_train_pred == y_train)
    forest_test_acc = np.mean(forest_test_pred == y_test)

    # Results comparison
    results = {
        'Model': ['Single Tree', 'Random Forest'],
        'Train Accuracy': [tree_train_acc, forest_train_acc],
        'Test Accuracy': [tree_test_acc, forest_test_acc],
        'Overfitting': [tree_train_acc - tree_test_acc, forest_train_acc - forest_test_acc]
    }

    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Overfitting':<12}")
    print("-" * 55)
    for i in range(len(results['Model'])):
        print(f"{results['Model'][i]:<15} {results['Train Accuracy'][i]:<12.4f} "
              f"{results['Test Accuracy'][i]:<12.4f} {results['Overfitting'][i]:<12.4f}")

    print(f"\n Random Forest OOB Score: {forest.oob_score_:.4f}")

    # Feature importance comparison
    print(f"\n Feature Importance Comparison (Top 5):")
    print(f"{'Feature':<12} {'Single Tree':<15} {'Random Forest':<15}")
    print("-" * 45)

    top_features = np.argsort(forest.feature_importances_)[::-1][:5]
    for feat_idx in top_features:
        tree_importance = single_tree.feature_importances_[feat_idx]
        forest_importance = forest.feature_importances_[feat_idx]
        print(f"Feature_{feat_idx:<5} {tree_importance:<15.4f} {forest_importance:<15.4f}")

    return single_tree, forest


def demonstrate_overfitting_control():
    """Demonstrate overfitting control with tree parameters"""
    print("\n  OVERFITTING CONTROL DEMONSTRATION")
    print("=" * 60)

    # Generate dataset prone to overfitting
    X, y = generate_classification_dataset(n_samples=200, n_features=20,
                                           n_informative=5, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test different parameters
    parameters = [
        {'name': 'No Constraints', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'name': 'Max Depth = 5', 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'name': 'Min Split = 10', 'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 1},
        {'name': 'Min Leaf = 5', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 5},
        {'name': 'Combined', 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 3},
    ]

    print(f"{'Configuration':<20} {'Train Acc':<12} {'Test Acc':<12} {'Tree Depth':<12} {'# Leaves':<12}")
    print("-" * 75)

    for params in parameters:
        tree = DecisionTreeAdvanced(
            criterion='gini',
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            task='classification'
        )

        tree.fit(X_train, y_train)

        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)
        depth = tree.get_depth()
        n_leaves = tree.get_n_leaves()

        print(f"{params['name']:<20} {train_acc:<12.4f} {test_acc:<12.4f} {depth:<12} {n_leaves:<12}")

    print("\n Key Insights:")
    print("- No constraints → High train accuracy, poor test (overfitting)")
    print("- Max depth limit → Prevents deep, overspecialized trees")
    print("- Min samples constraints → Ensures statistical significance")
    print("- Combined constraints → Best generalization")


def analyze_splitting_criteria():
    """Compare different splitting criteria"""
    print("\n SPLITTING CRITERIA COMPARISON")
    print("=" * 60)

    # Generate dataset
    X, y = generate_classification_dataset(n_samples=500, n_features=10, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    criteria = ['gini', 'entropy']

    print(f"{'Criterion':<12} {'Train Acc':<12} {'Test Acc':<12} {'Tree Depth':<12} {'Training Time':<15}")
    print("-" * 70)

    for criterion in criteria:
        import time
        start_time = time.time()

        tree = DecisionTreeAdvanced(
            criterion=criterion,
            max_depth=10,
            min_samples_split=5,
            task='classification'
        )

        tree.fit(X_train, y_train)
        training_time = time.time() - start_time

        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)
        depth = tree.get_depth()

        print(f"{criterion:<12} {train_acc:<12.4f} {test_acc:<12.4f} {depth:<12} {training_time:<15.4f}s")

    print("\n Splitting Criteria Explained:")
    print(" Gini Impurity: Gini = 1 - Σ(p_i²)")
    print("   - Faster to compute (no logarithms)")
    print("   - Range: [0, 0.5] for binary classification")
    print("   - Tends to isolate most frequent class")

    print("\n Entropy: H = -Σ(p_i * log₂(p_i))")
    print("   - More expensive (logarithm computation)")
    print("   - Range: [0, 1] for binary classification")
    print("   - Tends to create more balanced splits")


class GradientBoosting:
    """
    Simple Gradient Boosting implementation for regression
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        # Fitted attributes
        self.estimators_ = []
        self.initial_prediction_ = None

    def fit(self, X, y):
        """Fit gradient boosting model"""
        # Initialize with mean prediction
        self.initial_prediction_ = np.mean(y)
        current_prediction = np.full(len(y), self.initial_prediction_)

        print(f" Gradient Boosting: Training {self.n_estimators} weak learners")

        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for MSE)
            residuals = y - current_prediction

            # Fit weak learner to residuals
            weak_learner = DecisionTreeAdvanced(
                criterion='mse',
                max_depth=self.max_depth,
                min_samples_split=5,
                task='regression'
            )

            weak_learner.fit(X, residuals)
            self.estimators_.append(weak_learner)

            # Update predictions
            residual_predictions = weak_learner.predict(X)
            current_prediction += self.learning_rate * residual_predictions

            if (i + 1) % 20 == 0:
                mse = np.mean((y - current_prediction) ** 2)
                print(f"   Iteration {i + 1}: MSE = {mse:.4f}")

        return self

    def predict(self, X):
        """Make predictions"""
        predictions = np.full(len(X), self.initial_prediction_)

        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)

        return predictions


def demonstrate_ensemble_methods():
    """Demonstrate different ensemble methods"""
    print("\n ENSEMBLE METHODS COMPARISON")
    print("=" * 60)

    # Generate regression dataset
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 5)
    # True function with non-linear relationship
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] - 0.3 * X[:, 2] * X[:, 3] + 0.1 * np.random.randn(n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Compare different methods
    models = {
        'Single Tree': DecisionTreeAdvanced(criterion='mse', max_depth=6, task='regression'),
        'Random Forest': RandomForest(n_estimators=50, max_depth=6, task='regression', random_state=42),
        'Gradient Boosting': GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
    }

    results = {}

    for name, model in models.items():
        print(f"\n Training {name}...")
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse = np.mean((y_train - train_pred) ** 2)
        test_mse = np.mean((y_test - test_pred) ** 2)
        train_r2 = 1 - train_mse / np.var(y_train)
        test_r2 = 1 - test_mse / np.var(y_test)

        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

    # Display results
    print(f"\n ENSEMBLE COMPARISON RESULTS:")
    print(f"{'Model':<18} {'Train MSE':<12} {'Test MSE':<12} {'Train R²':<12} {'Test R²':<12}")
    print("-" * 75)

    for name, result in results.items():
        print(f"{name:<18} {result['train_mse']:<12.4f} {result['test_mse']:<12.4f} "
              f"{result['train_r2']:<12.4f} {result['test_r2']:<12.4f}")

    print(f"\n Ensemble Benefits:")
    print("• Random Forest: Reduces overfitting through bagging")
    print("• Gradient Boosting: Sequentially corrects errors")
    print("• Both typically outperform single trees")


def main():
    """Main function demonstrating decision trees and ensemble methods"""
    print(" Day 4: Decision Trees & Ensemble Methods")
    print("=" * 80)

    # Generate dataset
    X, y = generate_classification_dataset(n_samples=800, n_features=15,
                                           n_informative=8, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f" Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")

    print("\n" + "=" * 80)
    print(" BASIC DECISION TREE ANALYSIS")
    print("=" * 80)

    # Train basic decision tree
    tree = DecisionTreeAdvanced(
        criterion='gini',
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        task='classification'
    )

    tree.fit(X_train, y_train)

    # Make predictions
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_accuracy = np.mean(train_pred == y_train)
    test_accuracy = np.mean(test_pred == y_test)

    print(f" Decision Tree Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Tree Depth: {tree.get_depth()}")
    print(f"   Number of Leaves: {tree.get_n_leaves()}")
    print(f"   Features Used: {np.sum(tree.feature_importances_ > 0)}/{X.shape[1]}")

    # Print tree structure (first few levels)
    print(f"\n Tree Structure (partial):")
    tree.print_tree()

    # Plot results
    plot_decision_tree_results(X_train, y_train, tree, "Decision Tree Analysis")

    print("\n" + "=" * 80)
    print(" TREE vs FOREST COMPARISON")
    print("=" * 80)

    # Compare single tree vs random forest
    single_tree, forest = compare_tree_vs_forest(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 80)
    print(" OVERFITTING CONTROL")
    print("=" * 80)

    # Demonstrate overfitting control
    demonstrate_overfitting_control()

    print("\n" + "=" * 80)
    print(" SPLITTING CRITERIA")
    print("=" * 80)

    # Compare splitting criteria
    analyze_splitting_criteria()

    print("\n" + "=" * 80)
    print(" ENSEMBLE METHODS")
    print("=" * 80)

    # Demonstrate ensemble methods
    demonstrate_ensemble_methods()

    print("\n" + "=" * 80)
    print(" PRACTICAL INSIGHTS & INTERVIEW PREP")
    print("=" * 80)

    print(" KEY INTERVIEW CONCEPTS:")
    print("\n DECISION TREE ADVANTAGES:")
    print("    Interpretable and explainable")
    print("    Handles both numerical and categorical features")
    print("    No assumptions about data distribution")
    print("    Automatic feature selection")
    print("    Handles missing values naturally")

    print("\n DECISION TREE DISADVANTAGES:")
    print("    Prone to overfitting (high variance)")
    print("    Biased toward features with more levels")
    print("    Unstable (small data changes → different tree)")
    print("    Difficulty with linear relationships")
    print("    Greedy algorithm (local optima)")

    print("\n ENSEMBLE SOLUTIONS:")
    print("    Random Forest (Bagging):")
    print("      • Bootstrap sampling + feature randomness")
    print("      • Reduces variance, prevents overfitting")
    print("      • Parallel training, fast prediction")
    print("      • OOB error estimation")

    print("\n    Gradient Boosting:")
    print("      • Sequential error correction")
    print("      • Reduces bias and variance")
    print("      • Higher accuracy but slower training")
    print("      • Risk of overfitting with too many iterations")

    print("\n FEATURE IMPORTANCE INTERPRETATION:")
    print("   • Based on impurity decrease at each split")
    print("   • Higher importance ≠ causation")
    print("   • Can be biased toward high-cardinality features")
    print("   • Use permutation importance for unbiased measure")

    print("\n️ HYPERPARAMETER TUNING GUIDE:")
    print("    max_depth: Start with √(n_features), tune based on validation")
    print("    min_samples_split: 2-20, higher for noisy data")
    print("    min_samples_leaf: 1-10, higher prevents overfitting")
    print("    max_features: sqrt(n) for classification, n/3 for regression")
    print("    n_estimators: More is better until plateau (50-500)")


if __name__ == "__main__":
    main()