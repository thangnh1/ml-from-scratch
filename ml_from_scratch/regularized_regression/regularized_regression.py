import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class RegularizedRegression:
    """
    Comprehensive implementation of regularized regression methods:
    1. Ridge Regression (L2 regularization)
    2. Lasso Regression (L1 regularization)
    3. Elastic Net (L1 + L2 combination)
    4. Coordinate Descent for Lasso
    5. Cross-validation for hyperparameter tuning
    """

    def __init__(self, regularization='ridge', alpha=1.0, l1_ratio=0.5,
                 max_iter=1000, tol=1e-4, normalize=True):
        """
        Parameters:
        - regularization: 'ridge', 'lasso', 'elastic_net'
        - alpha: regularization strength (λ)
        - l1_ratio: ratio of L1 penalty in elastic net (0=ridge, 1=lasso)
        - normalize: whether to standardize features
        """
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize

        # Fitted parameters
        self.coef_ = None
        self.intercept_ = None
        self.path_ = []  # Coefficient path during training
        self.scaler_ = None

    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator for Lasso"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _ridge_solution(self, X, y):
        """
        Ridge Regression closed-form solution:
        β = (X^T X + αI)^(-1) X^T y
        """
        n_features = X.shape[1]

        # Add regularization to diagonal
        XTX_reg = X.T @ X + self.alpha * np.eye(n_features)
        XTy = X.T @ y

        # Solve regularized normal equation
        self.coef_ = np.linalg.solve(XTX_reg, XTy)

        return self

    def _lasso_coordinate_descent(self, X, y):
        """
        Lasso using Coordinate Descent Algorithm
        Cyclically optimize one coordinate at a time
        """
        n_samples, n_features = X.shape

        # Initialize coefficients
        self.coef_ = np.zeros(n_features)

        # Precompute X^T X diagonal for efficiency
        XTX_diag = np.sum(X ** 2, axis=0)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y - X @ self.coef_ + self.coef_[j] * X[:, j]

                # Compute coordinate update
                rho_j = X[:, j] @ residual

                # Soft thresholding
                if XTX_diag[j] > 0:
                    self.coef_[j] = self._soft_threshold(rho_j, self.alpha) / XTX_diag[j]
                else:
                    self.coef_[j] = 0

            # Store coefficient path
            self.path_.append(self.coef_.copy())

            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        return self

    def _elastic_net_coordinate_descent(self, X, y):
        """
        Elastic Net using Coordinate Descent
        Combines L1 and L2 penalties
        """
        n_samples, n_features = X.shape

        # Initialize coefficients
        self.coef_ = np.zeros(n_features)

        # Elastic net parameters
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1 - self.l1_ratio)

        # Precompute for efficiency
        XTX_diag = np.sum(X ** 2, axis=0)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y - X @ self.coef_ + self.coef_[j] * X[:, j]
                rho_j = X[:, j] @ residual

                # Elastic net update with L2 modification
                denominator = XTX_diag[j] + l2_reg
                if denominator > 0:
                    self.coef_[j] = self._soft_threshold(rho_j, l1_reg) / denominator
                else:
                    self.coef_[j] = 0

            # Store path
            self.path_.append(self.coef_.copy())

            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        return self

    def fit(self, X, y):
        """Fit regularized regression model"""
        # Normalize features if requested
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Center target variable
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_

        # Choose fitting method based on regularization type
        if self.regularization == 'ridge':
            self._ridge_solution(X, y_centered)
        elif self.regularization == 'lasso':
            self._lasso_coordinate_descent(X, y_centered)
        elif self.regularization == 'elastic_net':
            self._elastic_net_coordinate_descent(X, y_centered)
        else:
            raise ValueError("regularization must be 'ridge', 'lasso', or 'elastic_net'")

        return self

    def predict(self, X):
        """Make predictions"""
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return X @ self.coef_ + self.intercept_

    def get_feature_importance(self):
        """Get feature importance (absolute coefficients)"""
        return np.abs(self.coef_)

    def get_selected_features(self, threshold=1e-5):
        """Get features selected by L1 regularization"""
        return np.where(np.abs(self.coef_) > threshold)[0]


class CrossValidator:
    """
    Cross-validation framework for model selection
    """

    @staticmethod
    def k_fold_split(X, y, k=5, shuffle=True, random_state=42):
        """Generate k-fold cross-validation indices"""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)

        fold_size = n_samples // k
        folds = []

        for i in range(k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < k - 1 else n_samples

            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

            folds.append((train_indices, test_indices))

        return folds

    @staticmethod
    def cross_val_score(model_class, X, y, param_dict, cv=5, scoring='mse'):
        """Perform cross-validation with given parameters"""
        folds = CrossValidator.k_fold_split(X, y, k=cv)
        scores = []

        for train_idx, val_idx in folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and fit model
            model = model_class(**param_dict)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_val)

            # Calculate score
            if scoring == 'mse':
                score = np.mean((y_val - y_pred) ** 2)
            elif scoring == 'mae':
                score = np.mean(np.abs(y_val - y_pred))
            elif scoring == 'r2':
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                score = 1 - (ss_res / ss_tot)

            scores.append(score)

        return np.array(scores)

    @staticmethod
    def grid_search(model_class, X, y, param_grid, cv=5, scoring='mse', verbose=True):
        """Grid search for hyperparameter tuning"""
        best_score = float('inf') if scoring in ['mse', 'mae'] else -float('inf')
        best_params = None
        results = []

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        from itertools import product
        param_combinations = list(product(*param_values))

        if verbose:
            print(f" Grid Search: Testing {len(param_combinations)} parameter combinations")
            print("-" * 60)

        for i, param_combo in enumerate(param_combinations):
            param_dict = dict(zip(param_names, param_combo))

            # Perform cross-validation
            scores = CrossValidator.cross_val_score(
                model_class, X, y, param_dict, cv=cv, scoring=scoring
            )

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            results.append({
                'params': param_dict,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            })

            # Update best parameters
            is_better = (mean_score < best_score) if scoring in ['mse', 'mae'] else (mean_score > best_score)
            if is_better:
                best_score = mean_score
                best_params = param_dict

            if verbose:
                print(f"Params: {param_dict}")
                print(f"CV Score: {mean_score:.4f} (+/- {2 * std_score:.4f})")
                print()

        if verbose:
            print(f" Best Parameters: {best_params}")
            print(f" Best CV Score: {best_score:.4f}")

        return best_params, best_score, results

    @staticmethod
    def learning_curves(model_class, X, y, param_dict, train_sizes=None, cv=5):
        """Generate learning curves to diagnose bias-variance tradeoff"""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        n_samples = len(X)
        train_scores = []
        val_scores = []

        for train_size in train_sizes:
            n_train_samples = int(train_size * n_samples)

            # Get cross-validation folds
            folds = CrossValidator.k_fold_split(X, y, k=cv)
            fold_train_scores = []
            fold_val_scores = []

            for train_idx, val_idx in folds:
                # Use subset of training data
                train_subset = train_idx[:n_train_samples]

                X_train, X_val = X[train_subset], X[val_idx]
                y_train, y_val = y[train_subset], y[val_idx]

                # Fit model
                model = model_class(**param_dict)
                model.fit(X_train, y_train)

                # Calculate scores
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                train_mse = np.mean((y_train - train_pred) ** 2)
                val_mse = np.mean((y_val - val_pred) ** 2)

                fold_train_scores.append(train_mse)
                fold_val_scores.append(val_mse)

            train_scores.append(fold_train_scores)
            val_scores.append(fold_val_scores)

        return train_sizes, np.array(train_scores), np.array(val_scores)


class BiasVarianceAnalyzer:
    """
    Analyze bias-variance tradeoff with regularization
    """

    @staticmethod
    def regularization_path(model_class, X, y, alphas, regularization_type='ridge'):
        """Compute regularization path (coefficients vs alpha)"""
        coefficients = []
        mse_scores = []

        for alpha in alphas:
            if regularization_type == 'ridge':
                model = model_class(regularization='ridge', alpha=alpha)
            elif regularization_type == 'lasso':
                model = model_class(regularization='lasso', alpha=alpha)
            elif regularization_type == 'elastic_net':
                model = model_class(regularization='elastic_net', alpha=alpha, l1_ratio=0.5)

            model.fit(X, y)
            coefficients.append(model.coef_.copy())

            # Calculate training MSE
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            mse_scores.append(mse)

        return np.array(coefficients), np.array(mse_scores)

    @staticmethod
    def bias_variance_decomposition(model_class, X_train, y_train, X_test, y_test,
                                    param_dict, n_bootstrap=100):
        """
        Perform bias-variance decomposition using bootstrap sampling

        Total Error = Bias² + Variance + Noise
        """
        n_samples = len(X_train)
        predictions = []

        # Bootstrap sampling
        np.random.seed(42)
        for _ in range(n_bootstrap):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train[bootstrap_indices]
            y_boot = y_train[bootstrap_indices]

            # Fit model on bootstrap sample
            model = model_class(**param_dict)
            model.fit(X_boot, y_boot)

            # Predict on test set
            y_pred = model.predict(X_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)  # Shape: (n_bootstrap, n_test)

        # Calculate main prediction (average across bootstrap samples)
        main_prediction = np.mean(predictions, axis=0)

        # Calculate bias squared: (f_bar - y_true)²
        bias_squared = np.mean((main_prediction - y_test) ** 2)

        # Calculate variance: E[(f - f_bar)²]
        variance = np.mean(np.var(predictions, axis=0))

        # Calculate total error: E[(f - y_true)²]
        total_error = np.mean([np.mean((pred - y_test) ** 2) for pred in predictions])

        # Noise is the remainder
        noise = total_error - bias_squared - variance

        return {
            'bias_squared': bias_squared,
            'variance': variance,
            'noise': max(0, noise),  # Ensure non-negative
            'total_error': total_error
        }


def generate_polynomial_data(n_samples=100, noise_std=0.3, degree_true=3, seed=42):
    """Generate polynomial dataset for testing regularization"""
    np.random.seed(seed)

    # Generate true polynomial function
    X = np.linspace(-1, 1, n_samples).reshape(-1, 1)

    # True function: y = x + 0.5*x² - 0.3*x³ + noise
    y_true = X.flatten() + 0.5 * X.flatten() ** 2 - 0.3 * X.flatten() ** 3
    y = y_true + noise_std * np.random.randn(n_samples)

    return X, y, y_true


def plot_regularization_comparison(X, y, y_true, alphas):
    """Compare different regularization methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Generate polynomial features for fitting
    poly_features = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    regularization_types = ['ridge', 'lasso', 'elastic_net']
    colors = ['blue', 'red', 'green']

    # Plot 1: Original data and true function
    X_plot = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_plot_true = X_plot.flatten() + 0.5 * X_plot.flatten() ** 2 - 0.3 * X_plot.flatten() ** 3

    axes[0, 0].scatter(X, y, alpha=0.5, color='gray', label='Data')
    axes[0, 0].plot(X_plot, y_plot_true, 'k--', linewidth=2, label='True Function')
    axes[0, 0].set_title('Original Data and True Function')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Regularization paths
    for reg_type, color in zip(regularization_types, colors):
        coefficients, _ = BiasVarianceAnalyzer.regularization_path(
            RegularizedRegression, X_poly, y, alphas, reg_type
        )

        for i in range(coefficients.shape[1]):
            axes[0, 1].plot(alphas, coefficients[:, i], color=color, alpha=0.7)

    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Alpha (Regularization Strength)')
    axes[0, 1].set_ylabel('Coefficient Values')
    axes[0, 1].set_title('Regularization Paths')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Predictions with different alphas
    X_plot_poly = poly_features.transform(X_plot)

    test_alphas = [0.0, 0.01, 0.1, 1.0]
    for i, alpha in enumerate(test_alphas):
        model = RegularizedRegression(regularization='ridge', alpha=alpha, normalize=True)
        model.fit(X_poly, y)
        y_pred = model.predict(X_plot_poly)

        axes[1, 0].plot(X_plot, y_pred, label=f'Ridge α={alpha}', alpha=0.8)

    axes[1, 0].scatter(X, y, alpha=0.5, color='gray', s=20)
    axes[1, 0].plot(X_plot, y_plot_true, 'k--', linewidth=2, label='True Function')
    axes[1, 0].set_title('Ridge Regression with Different α')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cross-validation scores
    cv_scores_ridge = []
    cv_scores_lasso = []

    for alpha in alphas:
        # Ridge CV score
        ridge_scores = CrossValidator.cross_val_score(
            RegularizedRegression, X_poly, y,
            {'regularization': 'ridge', 'alpha': alpha}, cv=5
        )
        cv_scores_ridge.append(np.mean(ridge_scores))

        # Lasso CV score
        lasso_scores = CrossValidator.cross_val_score(
            RegularizedRegression, X_poly, y,
            {'regularization': 'lasso', 'alpha': alpha}, cv=5
        )
        cv_scores_lasso.append(np.mean(lasso_scores))

    axes[1, 1].plot(alphas, cv_scores_ridge, 'o-', label='Ridge', color='blue')
    axes[1, 1].plot(alphas, cv_scores_lasso, 'o-', label='Lasso', color='red')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Alpha')
    axes[1, 1].set_ylabel('Cross-Validation MSE')
    axes[1, 1].set_title('CV Scores vs Regularization')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_bias_variance_tradeoff():
    """Demonstrate bias-variance tradeoff with different regularization strengths"""
    print("\n BIAS-VARIANCE TRADEOFF ANALYSIS")
    print("=" * 60)

    # Generate data
    X, y, y_true = generate_polynomial_data(n_samples=50, noise_std=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create polynomial features
    poly = PolynomialFeatures(degree=10, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Test different regularization strengths
    alphas = [0.0, 0.01, 0.1, 1.0, 10.0]

    print(f"{'Alpha':<8} {'Bias²':<10} {'Variance':<10} {'Noise':<10} {'Total Error':<12}")
    print("-" * 55)

    for alpha in alphas:
        decomposition = BiasVarianceAnalyzer.bias_variance_decomposition(
            RegularizedRegression, X_train_poly, y_train, X_test_poly, y_test,
            {'regularization': 'ridge', 'alpha': alpha}, n_bootstrap=50
        )

        print(f"{alpha:<8.2f} {decomposition['bias_squared']:<10.4f} "
              f"{decomposition['variance']:<10.4f} {decomposition['noise']:<10.4f} "
              f"{decomposition['total_error']:<12.4f}")

    print("\n Interpretation:")
    print("- Low α: Low bias, high variance (overfitting)")
    print("- High α: High bias, low variance (underfitting)")
    print("- Optimal α: Balance bias and variance")


def main():
    """Main function demonstrating regularization and model selection"""
    print(" Day 3: Regularization & Model Selection")
    print("=" * 80)

    # Generate polynomial dataset
    X, y, y_true = generate_polynomial_data(n_samples=100, noise_std=0.3)

    print(f" Dataset: {X.shape[0]} samples with polynomial relationship + noise")

    # Create high-dimensional polynomial features (potential for overfitting)
    poly_features = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    print(f" Polynomial Features: {X_poly.shape[1]} features (degree 10)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

    print("\n" + "=" * 60)
    print(" COMPARING REGULARIZATION METHODS")
    print("=" * 60)

    # Compare different regularization methods
    methods = {
        'No Regularization': RegularizedRegression(regularization='ridge', alpha=0.0),
        'Ridge (α=0.1)': RegularizedRegression(regularization='ridge', alpha=0.1),
        'Lasso (α=0.01)': RegularizedRegression(regularization='lasso', alpha=0.01),
        'Elastic Net': RegularizedRegression(regularization='elastic_net', alpha=0.01, l1_ratio=0.5)
    }

    results = {}
    for name, model in methods.items():
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse = np.mean((y_train - train_pred) ** 2)
        test_mse = np.mean((y_test - test_pred) ** 2)

        # Count non-zero features (for sparsity analysis)
        non_zero_features = np.sum(np.abs(model.coef_) > 1e-5)

        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'non_zero_features': non_zero_features,
            'model': model
        }

        print(f" {name:<20}")
        print(f"   Train MSE: {train_mse:.4f}")
        print(f"   Test MSE:  {test_mse:.4f}")
        print(f"   Non-zero features: {non_zero_features}/{X_poly.shape[1]}")
        print(f"   Overfitting: {'Yes' if train_mse < 0.5 * test_mse else 'No'}")
        print()

    print("\n" + "=" * 60)
    print(" HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("=" * 60)

    # Grid search for optimal hyperparameters
    param_grids = {
        'Ridge': {
            'regularization': ['ridge'],
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Lasso': {
            'regularization': ['lasso'],
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'Elastic Net': {
            'regularization': ['elastic_net'],
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
    }

    best_models = {}
    for method_name, param_grid in param_grids.items():
        print(f"\n {method_name} Grid Search:")

        best_params, best_score, _ = CrossValidator.grid_search(
            RegularizedRegression, X_train, y_train, param_grid, cv=5, verbose=False
        )

        print(f" Best params: {best_params}")
        print(f" Best CV score: {best_score:.4f}")

        # Fit best model on full training set
        best_model = RegularizedRegression(**best_params)
        best_model.fit(X_train, y_train)

        test_pred = best_model.predict(X_test)
        test_mse = np.mean((y_test - test_pred) ** 2)

        print(f" Test MSE: {test_mse:.4f}")
        best_models[method_name] = best_model

    print("\n" + "=" * 60)
    print(" LEARNING CURVES ANALYSIS")
    print("=" * 60)

    # Generate learning curves for best Ridge model
    best_ridge_params = best_models['Ridge'].__dict__.copy()
    train_sizes, train_scores, val_scores = CrossValidator.learning_curves(
        RegularizedRegression, X_train, y_train,
        {'regularization': 'ridge', 'alpha': best_ridge_params['alpha']},
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )

    # Plot learning curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    color='blue', alpha=0.2)

    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    color='red', alpha=0.2)

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Learning Curves - Ridge Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Interpret learning curves
    final_gap = val_mean[-1] - train_mean[-1]
    print(f" Learning Curve Analysis:")
    print(f"   Final train-validation gap: {final_gap:.4f}")
    if final_gap > 0.05:
        print("   → Model shows overfitting. Consider more regularization.")
    elif train_mean[-1] > 0.5:
        print("   → Model shows underfitting. Consider less regularization.")
    else:
        print("   → Model shows good bias-variance balance.")

    print("\n" + "=" * 60)
    print(" FEATURE SELECTION WITH LASSO")
    print("=" * 60)

    # Use Lasso for feature selection
    lasso_model = best_models['Lasso']
    selected_features = lasso_model.get_selected_features()
    feature_importance = lasso_model.get_feature_importance()

    print(f" Lasso Feature Selection:")
    print(f"   Original features: {X_poly.shape[1]}")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   Sparsity: {(1 - len(selected_features) / X_poly.shape[1]) * 100:.1f}%")

    # Show top important features
    top_features = np.argsort(feature_importance)[-10:][::-1]
    print(f"\n Top 10 Important Features:")
    for i, feat_idx in enumerate(top_features):
        if feature_importance[feat_idx] > 1e-5:
            print(f"   {i + 1}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")

    print("\n" + "=" * 60)
    print(" REGULARIZATION PATH VISUALIZATION")
    print("=" * 60)

    # Plot regularization comparison
    alphas = np.logspace(-3, 2, 50)
    plot_regularization_comparison(X, y, y_true, alphas)

    # Bias-variance tradeoff analysis
    demonstrate_bias_variance_tradeoff()

    print("\n" + "=" * 60)
    print(" PRACTICAL RECOMMENDATIONS")
    print("=" * 60)

    print(" WHEN TO USE WHICH REGULARIZATION:")
    print()
    print(" RIDGE REGRESSION (L2):")
    print("    Use when: All features potentially relevant")
    print("    Use when: Multicollinearity present")
    print("    Use when: Stable, shrunk coefficients desired")
    print("    Avoid when: Need explicit feature selection")
    print()

    print(" LASSO REGRESSION (L1):")
    print("    Use when: Feature selection needed")
    print("    Use when: Interpretable sparse model desired")
    print("    Use when: High-dimensional data (p >> n)")
    print("    Avoid when: Groups of correlated features (picks one randomly)")
    print()

    print(" ELASTIC NET (L1 + L2):")
    print("    Use when: Want both feature selection AND stability")
    print("    Use when: Grouped features should be selected together")
    print("    Use when: n < p and features are correlated")
    print("    Avoid when: Clear preference for Ridge or Lasso exists")

    # Summary table of best models
    print("\n FINAL MODEL COMPARISON:")
    print("-" * 70)
    print(f"{'Method':<15} {'Best Alpha':<12} {'CV Score':<12} {'Test MSE':<12} {'Features':<10}")
    print("-" * 70)

    for method_name, model in best_models.items():
        test_pred = model.predict(X_test)
        test_mse = np.mean((y_test - test_pred) ** 2)
        n_features = len(model.get_selected_features())

        alpha_str = f"{model.alpha:.3f}"
        if hasattr(model, 'l1_ratio'):
            alpha_str += f" (r={model.l1_ratio:.1f})"

        print(f"{method_name:<15} {alpha_str:<12} {'N/A':<12} {test_mse:<12.4f} {n_features:<10}")


def feature_selection_comparison():
    """Compare different feature selection methods"""
    print("\n BONUS: FEATURE SELECTION METHODS COMPARISON")
    print("=" * 70)

    # Generate high-dimensional dataset with irrelevant features
    np.random.seed(42)
    n_samples, n_relevant, n_irrelevant = 100, 5, 45

    X_relevant = np.random.randn(n_samples, n_relevant)
    X_irrelevant = np.random.randn(n_samples, n_irrelevant)
    X = np.hstack([X_relevant, X_irrelevant])

    # True coefficients: only first 5 features are relevant
    true_coef = np.zeros(n_relevant + n_irrelevant)
    true_coef[:n_relevant] = [2, -1.5, 1, -0.5, 0.8]

    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    print(f" Dataset: {n_samples} samples, {X.shape[1]} features")
    print(f"   Relevant features: {n_relevant} (indices 0-{n_relevant - 1})")
    print(f"   Irrelevant features: {n_irrelevant}")

    # Compare different selection methods
    methods = {
        'Lasso (α=0.01)': RegularizedRegression('lasso', alpha=0.01),
        'Lasso (α=0.1)': RegularizedRegression('lasso', alpha=0.1),
        'Elastic Net': RegularizedRegression('elastic_net', alpha=0.1, l1_ratio=0.5)
    }

    print(f"\n{'Method':<20} {'Selected':<10} {'Relevant Found':<15} {'Precision':<12} {'Recall':<10}")
    print("-" * 75)

    for name, model in methods.items():
        model.fit(X, y)
        selected = model.get_selected_features()

        # Calculate precision and recall for feature selection
        true_relevant = set(range(n_relevant))
        selected_relevant = set(selected) & true_relevant

        precision = len(selected_relevant) / len(selected) if len(selected) > 0 else 0
        recall = len(selected_relevant) / len(true_relevant)

        print(f"{name:<20} {len(selected):<10} {len(selected_relevant):<15} "
              f"{precision:<12.3f} {recall:<10.3f}")


if __name__ == "__main__":
    main()
    feature_selection_comparison()