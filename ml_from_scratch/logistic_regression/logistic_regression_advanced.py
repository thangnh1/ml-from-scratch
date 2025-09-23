import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


class LogisticRegressionAdvanced:
    """
    Comprehensive Logistic Regression implementation with:
    - Binary and Multiclass classification
    - Class imbalance handling
    - Regularization (L1/L2)
    - Comprehensive evaluation metrics
    """

    def __init__(self, class_weight=None, regularization=None, lambda_reg=0.01,
                 solver='gradient_descent', max_iter=1000, learning_rate=0.01):
        """
        Parameters:
        - class_weight: None, 'balanced', or dict
        - regularization: None, 'l1', 'l2', 'elastic'
        - lambda_reg: regularization strength
        - solver: 'gradient_descent', 'newton', 'lbfgs'
        """
        self.class_weight = class_weight
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # Fitted parameters
        self.weights_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.cost_history_ = []
        self.sample_weights_ = None

    def _sigmoid(self, z):
        """Stable sigmoid function to avoid overflow"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500) # giới hạn z [-500, 500] để vẫn biểu diễn được bằng float64
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        """Stable softmax function for multiclass"""
        # Subtract max for numerical stability
        z_stable = z - np.max(z, axis=1, keepdims=True) # lấy z-m_max để z luôn bé hơn hoặc bằng 0 -> không còn giá trị cực lớn
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_sample_weights(self, y):
        """Compute sample weights for class imbalance"""
        if self.class_weight is None:
            return np.ones(len(y)) # default w = 1 nếu không định nghĩa class_weight

        elif self.class_weight == 'balanced':
            # Weight inversely proportional to class frequency
            unique_classes, counts = np.unique(y, return_counts=True) # tính tổng record của từng class
            n_samples = len(y)
            n_classes = len(unique_classes)

            # balanced weight = n_samples / (n_classes * class_count)
            class_weights = n_samples / (n_classes * counts) # w_class = tổng sample / (tổng class * count record class)
            weight_dict = dict(zip(unique_classes, class_weights))

            return np.array([weight_dict[class_] for class_ in y])

        elif isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(class_, 1.0) for class_ in y]) # đọc w từ khai báo dict ban đầu

    def _add_regularization(self, cost, weights):
        """Add regularization term to cost"""
        if self.regularization == 'l1':
            return cost + self.lambda_reg * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            return cost + self.lambda_reg * 0.5 * np.sum(weights ** 2)
        elif self.regularization == 'elastic':
            l1_term = self.lambda_reg * 0.5 * np.sum(np.abs(weights))
            l2_term = self.lambda_reg * 0.5 * np.sum(weights ** 2)
            return cost + l1_term + l2_term
        return cost

    def _compute_cost_binary(self, X, y, weights):
        """Compute logistic loss for binary classification"""
        z = X @ weights # @ == np.dot
        h = self._sigmoid(z)

        # Clip predictions to prevent log(0)
        h = np.clip(h, 1e-15, 1 - 1e-15)

        # Weighted logistic loss (log loss/cross entropy loss)
        cost = -np.average(
            y * np.log(h) + (1 - y) * np.log(1 - h),
            weights=self.sample_weights_
        )

        # Add regularization
        cost = self._add_regularization(cost, weights[1:])  # Don't regularize bias

        return cost

    def _compute_gradient_binary(self, X, y, weights):
        """Compute gradient for binary classification"""
        z = X @ weights
        h = self._sigmoid(z)

        # Weighted gradient
        error = h - y
        gradient = (1 / len(X)) * X.T @ (error * self.sample_weights_)

        # Add regularization gradient
        if self.regularization == 'l1':
            # L1: gradient = λ * sign(w), but don't regularize bias
            reg_grad = np.zeros_like(weights)
            reg_grad[1:] = self.lambda_reg * np.sign(weights[1:])
            gradient += reg_grad
        elif self.regularization == 'l2':
            # L2: gradient = λ * w, but don't regularize bias
            reg_grad = np.zeros_like(weights)
            reg_grad[1:] = self.lambda_reg * weights[1:]
            gradient += reg_grad

        return gradient

    def _fit_binary(self, X, y):
        """Fit binary logistic regression"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X_with_intercept.shape[1]

        # Initialize weights
        weights = np.random.normal(0, 0.01, n_features)

        # Compute sample weights for class imbalance
        self.sample_weights_ = self._compute_sample_weights(y)

        if self.solver == 'gradient_descent':
            # Gradient descent optimization
            for iteration in range(self.max_iter):
                cost = self._compute_cost_binary(X_with_intercept, y, weights)
                gradient = self._compute_gradient_binary(X_with_intercept, y, weights)

                # Update weights
                weights -= self.learning_rate * gradient

                self.cost_history_.append(cost)

                # Check convergence
                if len(self.cost_history_) > 1:
                    if abs(self.cost_history_[-2] - self.cost_history_[-1]) < 1e-6:
                        break

        elif self.solver == 'lbfgs':
            # Use scipy optimization
            def cost_function(w):
                return self._compute_cost_binary(X_with_intercept, y, w)

            def gradient_function(w):
                return self._compute_gradient_binary(X_with_intercept, y, w)

            result = minimize(
                cost_function, weights, method='L-BFGS-B',
                jac=gradient_function, options={'maxiter': self.max_iter}
            )
            weights = result.x
            self.cost_history_ = [result.fun]

        self.intercept_ = weights[0]
        self.weights_ = weights[1:]

        return self

    def _one_hot_encode(self, y):
        """Convert labels to one-hot encoding"""
        n_samples = len(y)
        n_classes = len(self.classes_)
        y_onehot = np.zeros((n_samples, n_classes))
        for i, class_ in enumerate(self.classes_):
            y_onehot[y == class_, i] = 1
        return y_onehot

    def _fit_multiclass(self, X, y):
        """Fit multiclass logistic regression (softmax)"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_samples, n_features = X_with_intercept.shape

        # Initialize weights matrix: (n_features, n_classes)
        weights_matrix = np.random.normal(0, 0.01, (n_features, self.n_classes_))

        # One-hot encode labels
        y_onehot = self._one_hot_encode(y)

        # Compute sample weights
        self.sample_weights_ = self._compute_sample_weights(y)

        for iteration in range(self.max_iter):
            # Forward pass
            z = X_with_intercept @ weights_matrix  # (n_samples, n_classes)
            probabilities = self._softmax(z)

            # Compute cost (cross-entropy)
            probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
            cost = -np.average(
                np.sum(y_onehot * np.log(probabilities), axis=1),
                weights=self.sample_weights_
            )

            # Add regularization
            if self.regularization == 'l2':
                cost += self.lambda_reg * 0.5 * np.sum(weights_matrix[1:] ** 2)

            self.cost_history_.append(cost)

            # Compute gradient
            error = probabilities - y_onehot  # (n_samples, n_classes)
            gradient = (1 / n_samples) * X_with_intercept.T @ (error * self.sample_weights_[:, np.newaxis])

            # Add regularization gradient
            if self.regularization == 'l2':
                reg_grad = np.zeros_like(weights_matrix)
                reg_grad[1:] = self.lambda_reg * weights_matrix[1:]
                gradient += reg_grad

            # Update weights
            weights_matrix -= self.learning_rate * gradient

            # Check convergence
            if len(self.cost_history_) > 1:
                if abs(self.cost_history_[-2] - self.cost_history_[-1]) < 1e-6:
                    break

        self.intercept_ = weights_matrix[0]
        self.weights_ = weights_matrix[1:]

        return self

    def fit(self, X, y):
        """Fit logistic regression model"""
        # Encode labels and determine classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Handle class encoding
        if self.n_classes_ == 2:
            # Binary classification: encode as 0/1
            y_encoded = (y == self.classes_[1]).astype(int)
            return self._fit_binary(X, y_encoded)
        else:
            # Multiclass classification
            return self._fit_multiclass(X, y)

    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.n_classes_ == 2:
            # Binary classification
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            weights = np.concatenate([[self.intercept_], self.weights_])
            z = X_with_intercept @ weights
            prob_positive = self._sigmoid(z)
            return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Multiclass classification
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            weights_matrix = np.vstack([self.intercept_, self.weights_])
            z = X_with_intercept @ weights_matrix
            return self._softmax(z)

    def predict(self, X, threshold=0.5):
        """Make predictions"""
        probabilities = self.predict_proba(X)

        if self.n_classes_ == 2:
            predictions = (probabilities[:, 1] >= threshold).astype(int)
            return np.where(predictions == 1, self.classes_[1], self.classes_[0])
        else:
            predictions = np.argmax(probabilities, axis=1)
            return self.classes_[predictions]


class ClassificationMetrics:
    """
    Comprehensive classification evaluation metrics
    """

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        """Compute confusion matrix"""
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        n_labels = len(labels)
        cm = np.zeros((n_labels, n_labels), dtype=int)

        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1

        return cm, labels

    @staticmethod
    def precision_recall_f1(y_true, y_pred, average='binary'):
        """Compute precision, recall, and F1-score"""
        if average == 'binary':
            # Assume positive class is 1 or the second unique value
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            pos_label = unique_labels[-1] if len(unique_labels) == 2 else 1

            tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
            fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
            fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return precision, recall, f1

        else:  # macro average
            labels = np.unique(np.concatenate([y_true, y_pred]))
            precisions, recalls, f1s = [], [], []

            for label in labels:
                tp = np.sum((y_true == label) & (y_pred == label))
                fp = np.sum((y_true != label) & (y_pred == label))
                fn = np.sum((y_true == label) & (y_pred != label))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            return np.mean(precisions), np.mean(recalls), np.mean(f1s)

    @staticmethod
    def roc_curve(y_true, y_scores):
        """Compute ROC curve points"""
        # Sort by descending score
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores_sorted = y_scores[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]

        # Get unique thresholds
        thresholds = np.concatenate([[y_scores_sorted[0] + 1], y_scores_sorted, [0]])

        tpr_list, fpr_list = [], []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return np.array(fpr_list), np.array(tpr_list), thresholds

    @staticmethod
    def auc_score(fpr, tpr):
        """Compute Area Under Curve using trapezoidal rule"""
        # Sort by fpr
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # Trapezoidal integration
        auc = np.trapz(tpr_sorted, fpr_sorted)
        return auc

    @staticmethod
    def precision_recall_curve(y_true, y_scores):
        """Compute Precision-Recall curve"""
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores_sorted = y_scores[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]

        thresholds = np.concatenate([[y_scores_sorted[0] + 1], y_scores_sorted])

        precision_list, recall_list = [], []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)

        return np.array(precision_list), np.array(recall_list), thresholds

    @staticmethod
    def classification_report(y_true, y_pred, y_proba=None):
        """Generate comprehensive classification report"""
        print("🔍 CLASSIFICATION REPORT")
        print("=" * 60)

        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Confusion Matrix
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"Labels: {labels}")
        print(cm)

        # Per-class metrics
        print(f"\nPer-class Metrics:")
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 55)

        overall_precision, overall_recall, overall_f1 = [], [], []

        for i, label in enumerate(labels):
            # Binary classification for each class
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)

            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            support = np.sum(y_true == label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            overall_precision.append(precision)
            overall_recall.append(recall)
            overall_f1.append(f1)

            print(f"{str(label):<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")

        # Overall metrics
        print("-" * 55)
        macro_precision = np.mean(overall_precision)
        macro_recall = np.mean(overall_recall)
        macro_f1 = np.mean(overall_f1)

        print(f"{'Macro avg':<10} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f} {len(y_true):<10}")

        # ROC-AUC for binary classification
        if len(labels) == 2 and y_proba is not None:
            y_true_binary = (y_true == labels[1]).astype(int)
            y_scores = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba

            fpr, tpr, _ = ClassificationMetrics.roc_curve(y_true_binary, y_scores)
            auc = ClassificationMetrics.auc_score(fpr, tpr)
            print(f"\nROC-AUC Score: {auc:.4f}")

        return cm, labels


def generate_classification_datasets():
    """Generate different types of classification datasets"""
    np.random.seed(42)
    datasets = {}

    # 1. Balanced binary classification
    n_samples = 1000
    X1 = np.random.randn(n_samples, 2)
    y1 = ((X1[:, 0] + X1[:, 1]) > 0).astype(int)
    datasets['balanced_binary'] = (X1, y1, "Balanced Binary Classification")

    # 2. Imbalanced binary classification (90% class 0, 10% class 1)
    n_majority = 900
    n_minority = 100
    X2_majority = np.random.normal(0, 1, (n_majority, 2))
    X2_minority = np.random.normal(2, 0.5, (n_minority, 2))
    X2 = np.vstack([X2_majority, X2_minority])
    y2 = np.hstack([np.zeros(n_majority), np.ones(n_minority)])

    # Shuffle
    indices = np.random.permutation(len(X2))
    X2, y2 = X2[indices], y2[indices]
    datasets['imbalanced_binary'] = (X2, y2, "Imbalanced Binary Classification")

    # 3. Multiclass classification (3 classes)
    n_per_class = 200
    X3_class0 = np.random.normal([-2, -2], 0.8, (n_per_class, 2))
    X3_class1 = np.random.normal([2, -2], 0.8, (n_per_class, 2))
    X3_class2 = np.random.normal([0, 2], 0.8, (n_per_class, 2))

    X3 = np.vstack([X3_class0, X3_class1, X3_class2])
    y3 = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])

    # Shuffle
    indices = np.random.permutation(len(X3))
    X3, y3 = X3[indices], y3[indices]
    datasets['multiclass'] = (X3, y3, "Multiclass Classification")

    return datasets


def plot_classification_results(X, y, model, title="Classification Results"):
    """Plot classification results and decision boundary"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Data and Decision Boundary
    if X.shape[1] == 2:  # Only for 2D data
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)[:, 1] if hasattr(model, 'predict_proba') else model.predict(mesh_points)
        Z = Z.reshape(xx.shape)

        axes[0].contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdBu')
        scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black')
        axes[0].set_title(f'{title}\nDecision Boundary')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=axes[0])

    # Plot 2: Cost/Loss History
    if hasattr(model, 'cost_history_') and len(model.cost_history_) > 1:
        axes[1].plot(model.cost_history_, linewidth=2)
        axes[1].set_title('Training Loss')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Cost')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

    # Plot 3: Class Distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    axes[2].bar(unique_classes, counts, color=['skyblue', 'salmon', 'lightgreen'][:len(unique_classes)])
    axes[2].set_title('Class Distribution')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Count')

    # Add count labels on bars
    for i, count in enumerate(counts):
        axes[2].text(unique_classes[i], count + 0.01 * max(counts), str(count),
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate Logistic Regression implementations"""
    print("🚀 Day 2: Logistic Regression & Classification Metrics")
    print("=" * 80)

    datasets = generate_classification_datasets()

    for dataset_name, (X, y, description) in datasets.items():
        print(f"\n📊 DATASET: {description}")
        print("-" * 60)
        print(f"Shape: {X.shape}, Classes: {np.unique(y)}")

        if dataset_name == 'balanced_binary':
            # Standard logistic regression
            model = LogisticRegressionAdvanced()
            model.fit(X, y)

            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            print("\n✅ Standard Logistic Regression")
            ClassificationMetrics.classification_report(y, y_pred, y_proba)

            plot_classification_results(X, y, model, "Standard Logistic Regression")

        elif dataset_name == 'imbalanced_binary':
            # Compare standard vs balanced class weights
            models = {
                'Standard': LogisticRegressionAdvanced(),
                'Balanced': LogisticRegressionAdvanced(class_weight='balanced'),
                'L2 Regularized': LogisticRegressionAdvanced(regularization='l2', lambda_reg=0.1)
            }

            for model_name, model in models.items():
                print(f"\n✅ {model_name}")
                model.fit(X, y)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)
                ClassificationMetrics.classification_report(y, y_pred, y_proba)

                if model_name == 'Balanced':
                    plot_classification_results(X, y, model, f"{model_name} - Imbalanced Data")

        elif dataset_name == 'multiclass':
            # Multiclass classification
            model = LogisticRegressionAdvanced()
            model.fit(X, y)

            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            print("\n✅ Multiclass Logistic Regression")
            ClassificationMetrics.classification_report(y, y_pred, y_proba)

            plot_classification_results(X, y, model, "Multiclass Classification")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

# =============================================================================
# INTERVIEW QUESTIONS FOR DAY 2:
# =============================================================================

"""
🎯 INTERVIEW QUESTIONS FOR DAY 2:

1. SIGMOID FUNCTION INTUITION:
   Q: "Why use sigmoid instead of linear function for classification?"
   A: Sigmoid maps any real number to (0,1), representing probability.
      Linear function can output values outside [0,1].

2. ODDS RATIO INTERPRETATION:
   Q: "Coefficient β=0.693. How do you interpret this?"
   A: exp(0.693) ≈ 2. One unit increase in feature doubles the odds of positive class.

3. IMBALANCED DATASETS:
   Q: "Dataset has 95% negative, 5% positive class. Model achieves 95% accuracy. Good?"
   A: No! Model could predict all negative. Check precision, recall, F1, ROC-AUC.

4. PRECISION vs RECALL TRADEOFF:
   Q: "Medical diagnosis: prefer high precision or recall?"
   A: High recall (minimize false negatives - don't miss diseases)
      Fraud detection: High precision (minimize false positives - don't block good users)

5. ROC vs PR CURVES:
   Q: "When to use ROC-AUC vs PR-AUC?"
   A: ROC-AUC for balanced datasets. PR-AUC for imbalanced (focuses on positive class).

6. MULTICLASS EXTENSION:
   Q: "How does logistic regression handle multiclass classification?"
   A: Two approaches: 
      - One-vs-Rest: Train binary classifier for each class
      - Softmax: Generalize sigmoid to multiple classes simultaneously

7. GRADIENT COMPUTATION:
   Q: "Derive gradient for logistic regression"
   A: ∇J = (1/m) * X^T * (h - y), where h = sigmoid(X*θ)
      Same form as linear regression, but h is sigmoid instead of linear

8. REGULARIZATION NECESSITY:
   Q: "Why might you need regularization in logistic regression?"
   A: High-dimensional data, multicollinearity, overfitting
      L1 (Lasso): Feature selection, sparse solutions
      L2 (Ridge): Shrinkage, handles multicollinearity

9. CLASS IMBALANCE SOLUTIONS:
   Q: "5 ways to handle imbalanced datasets"
   A: 1) Class weights, 2) SMOTE/oversampling, 3) Undersampling majority
      4) Cost-sensitive learning, 5) Threshold tuning

10. CONVERGENCE ISSUES:
    Q: "Logistic regression not converging. Possible causes?"
    A: Perfect separation, multicollinearity, learning rate too high,
       outliers, need feature scaling

PRACTICAL SCENARIOS:

Scenario 1: "Email spam detection - 99% not spam, 1% spam"
- Use class_weight='balanced' or custom weights
- Focus on recall (don't miss spam) but maintain reasonable precision
- Consider cost of false positives vs false negatives

Scenario 2: "Medical diagnosis model shows 95% accuracy but doctors complain"
- Likely high specificity, low sensitivity
- Check confusion matrix, precision/recall per class
- Adjust threshold or use balanced datasets

Scenario 3: "Model works in training but fails in production"
- Data drift: feature distributions changed
- Class distribution shift
- Need monitoring and retraining pipeline

CODING CHALLENGES:
- Implement L1 regularization (coordinate descent)
- Add momentum to gradient descent
- Implement one-vs-rest multiclass manually
- Handle categorical features with target encoding
- Build ROC curve from scratch without sklearn

MATHEMATICAL UNDERSTANDING:
- Derive why cross-entropy is natural loss for classification
- Understand connection between logistic regression and neural networks
- Explain why we use log-odds (logits) instead of direct probabilities
"""