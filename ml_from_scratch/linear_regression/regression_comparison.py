import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time


class RegressionComparison:
    """
    Compare different regression methods:
    1. OLS (Ordinary Least Squares) - Normal Equation
    2. Gradient Descent
    3. Ridge Regression (L2 regularization)
    4. Lasso Regression (L1 regularization)
    5. Robust Regression (Huber loss)
    """

    def __init__(self):
        self.methods = {}
        self.results = {}

    def ols_normal_equation(self, X, y):
        """
        OLS using Normal Equation: θ = (X^T X)^(-1) X^T y

        PROS:
        - Analytical solution (exact)
        - Fast for small datasets
        - No hyperparameters to tune

        CONS:
        - Requires matrix inversion O(n³) - expensive for large data
        - Fails if X^T X is singular (multicollinearity)
        - Sensitive to outliers
        - No regularization
        """
        print("OLS Normal Equation")
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # Check for singularity
        XTX = X_with_intercept.T @ X_with_intercept
        if np.linalg.det(XTX) < 1e-12:
            print("⚠️  Matrix is singular! Adding small regularization...")
            XTX += 1e-6 * np.eye(XTX.shape[0])

        coefficients = np.linalg.solve(XTX, X_with_intercept.T @ y)

        # Calculate metrics
        y_pred = X_with_intercept @ coefficients
        mse = np.mean((y - y_pred) ** 2)

        training_time = time.time() - start_time

        return {
            'method': 'OLS Normal Equation',
            'coefficients': coefficients,
            'predictions': y_pred,
            'mse': mse,
            'training_time': training_time,
            'pros': 'Exact solution, fast for small data',
            'cons': 'Matrix inversion expensive, sensitive to outliers'
        }

    def gradient_descent(self, X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        OLS using Gradient Descent: θ = θ - α * ∇J(θ)

        PROS:
        - Works with large datasets (no matrix inversion)
        - Memory efficient
        - Can handle streaming data

        CONS:
        - Requires hyperparameter tuning (learning rate)
        - May not converge or converge slowly
        - Approximate solution
        """
        print("Gradient Descent")
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_samples = X.shape[0]

        # Initialize coefficients
        coefficients = np.random.normal(0, 0.01, X_with_intercept.shape[1])

        cost_history = []

        for iteration in range(max_iterations):
            # Forward pass
            y_pred = X_with_intercept @ coefficients
            cost = np.mean((y - y_pred) ** 2)
            cost_history.append(cost)

            # Gradient: ∇J = (2/n) * X^T * (X*θ - y)
            gradient = (2 / n_samples) * X_with_intercept.T @ (y_pred - y)

            # Update coefficients
            new_coefficients = coefficients - learning_rate * gradient

            # Check convergence
            if np.linalg.norm(new_coefficients - coefficients) < tolerance:
                print(f"   Converged after {iteration} iterations")
                break

            coefficients = new_coefficients

        y_pred = X_with_intercept @ coefficients
        mse = np.mean((y - y_pred) ** 2)

        training_time = time.time() - start_time

        return {
            'method': 'Gradient Descent',
            'coefficients': coefficients,
            'predictions': y_pred,
            'mse': mse,
            'training_time': training_time,
            'cost_history': cost_history,
            'iterations': iteration + 1,
            'pros': 'Scalable to large data, memory efficient',
            'cons': 'Requires tuning, approximate solution'
        }

    def ridge_regression(self, X, y, alpha=1.0):
        """
        Ridge Regression: θ = (X^T X + αI)^(-1) X^T y

        PROS:
        - Handles multicollinearity well
        - Always has unique solution (even if X^T X singular)
        - Reduces overfitting

        CONS:
        - Doesn't perform feature selection
        - Need to tune α hyperparameter
        - Biased estimates
        """
        print(f"Ridge Regression (alpha={alpha})")
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # Ridge: θ = (X^T X + αI)^(-1) X^T y
        # Note: Don't regularize intercept (first column)
        I = np.eye(X_with_intercept.shape[1])
        I[0, 0] = 0  # Don't penalize intercept

        XTX_ridge = X_with_intercept.T @ X_with_intercept + alpha * I
        coefficients = np.linalg.solve(XTX_ridge, X_with_intercept.T @ y)

        y_pred = X_with_intercept @ coefficients
        mse = np.mean((y - y_pred) ** 2)

        # L2 penalty
        l2_penalty = alpha * np.sum(coefficients[1:] ** 2)  # Exclude intercept
        total_loss = mse + l2_penalty

        training_time = time.time() - start_time

        return {
            'method': f'Ridge (α={alpha})',
            'coefficients': coefficients,
            'predictions': y_pred,
            'mse': mse,
            'l2_penalty': l2_penalty,
            'total_loss': total_loss,
            'training_time': training_time,
            'pros': 'Handles multicollinearity, prevents overfitting',
            'cons': 'Biased estimates, no feature selection'
        }

    def lasso_regression(self, X, y, alpha=1.0, max_iterations=1000):
        """
        Lasso Regression using Coordinate Descent
        Loss: ||y - Xθ||² + α||θ||₁

        PROS:
        - Automatic feature selection (sets coefficients to 0)
        - Good for high-dimensional data
        - Interpretable models

        CONS:
        - Can be unstable with correlated features
        - Need to tune α
        - More complex optimization
        """
        print(f"🟡 Lasso Regression (α={alpha})")
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_samples = X.shape[0]

        # Initialize coefficients
        coefficients = np.zeros(X_with_intercept.shape[1])

        for iteration in range(max_iterations):
            coefficients_old = coefficients.copy()

            # Coordinate descent: update each coefficient one by one
            for j in range(len(coefficients)):
                # Calculate residual without j-th feature
                X_j = X_with_intercept[:, j]
                y_residual = y - X_with_intercept @ coefficients + coefficients[j] * X_j

                # Coordinate update with soft thresholding
                rho = X_j @ y_residual
                z = np.sum(X_j ** 2)

                if j == 0:  # Don't regularize intercept
                    coefficients[j] = rho / z
                else:
                    # Soft thresholding operator
                    if rho > alpha:
                        coefficients[j] = (rho - alpha) / z
                    elif rho < -alpha:
                        coefficients[j] = (rho + alpha) / z
                    else:
                        coefficients[j] = 0

            # Check convergence
            if np.linalg.norm(coefficients - coefficients_old) < 1e-6:
                print(f"   Converged after {iteration} iterations")
                break

        y_pred = X_with_intercept @ coefficients
        mse = np.mean((y - y_pred) ** 2)

        # L1 penalty
        l1_penalty = alpha * np.sum(np.abs(coefficients[1:]))  # Exclude intercept
        total_loss = mse + l1_penalty

        # Count non-zero coefficients (selected features)
        selected_features = np.sum(np.abs(coefficients[1:]) > 1e-6)

        training_time = time.time() - start_time

        return {
            'method': f'Lasso (α={alpha})',
            'coefficients': coefficients,
            'predictions': y_pred,
            'mse': mse,
            'l1_penalty': l1_penalty,
            'total_loss': total_loss,
            'selected_features': selected_features,
            'training_time': training_time,
            'iterations': iteration + 1,
            'pros': 'Feature selection, sparse models',
            'cons': 'Unstable with correlated features'
        }

    def huber_regression(self, X, y, delta=1.35, max_iterations=100):
        """
        Robust Regression using Huber Loss
        Huber loss: smooth for small errors, linear for large errors

        PROS:
        - Robust to outliers
        - Smoother than absolute loss
        - Good for noisy data

        CONS:
        - More complex optimization
        - Need to tune δ parameter
        - Iterative solution required
        """
        print(f"🟠 Huber Regression (δ={delta})")
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # Initialize with OLS solution
        coefficients = np.linalg.solve(X_with_intercept.T @ X_with_intercept,
                                       X_with_intercept.T @ y)

        for iteration in range(max_iterations):
            coefficients_old = coefficients.copy()

            # Calculate residuals
            y_pred = X_with_intercept @ coefficients
            residuals = y - y_pred

            # Calculate weights based on residuals
            weights = np.ones_like(residuals)
            large_residuals = np.abs(residuals) > delta
            weights[large_residuals] = delta / np.abs(residuals[large_residuals])

            # Weighted least squares update
            W = np.diag(weights)
            XTW = X_with_intercept.T @ W
            coefficients = np.linalg.solve(XTW @ X_with_intercept, XTW @ y)

            # Check convergence
            if np.linalg.norm(coefficients - coefficients_old) < 1e-6:
                print(f"   Converged after {iteration} iterations")
                break

        y_pred = X_with_intercept @ coefficients

        # Calculate Huber loss
        residuals = y - y_pred
        huber_loss = np.sum(np.where(np.abs(residuals) <= delta,
                                     0.5 * residuals ** 2,
                                     delta * (np.abs(residuals) - 0.5 * delta)))

        mse = np.mean(residuals ** 2)  # For comparison

        training_time = time.time() - start_time

        return {
            'method': f'Huber (δ={delta})',
            'coefficients': coefficients,
            'predictions': y_pred,
            'mse': mse,
            'huber_loss': huber_loss,
            'training_time': training_time,
            'iterations': iteration + 1,
            'pros': 'Robust to outliers, smooth loss',
            'cons': 'Complex optimization, parameter tuning'
        }

    def compare_all_methods(self, X, y, test_X=None, test_y=None, add_outliers=False):
        """Compare all regression methods"""

        # Add outliers for robustness testing
        if add_outliers:
            print("🔥 Adding outliers to test robustness...")
            outlier_indices = np.random.choice(len(y), size=int(0.05 * len(y)), replace=False)
            y_with_outliers = y.copy()
            y_with_outliers[outlier_indices] += np.random.normal(0, 5 * np.std(y), len(outlier_indices))
            y = y_with_outliers

        print("\n" + "=" * 70)
        print("REGRESSION METHODS COMPARISON")
        print("=" * 70)

        results = {}

        # 1. OLS Normal Equation
        results['ols'] = self.ols_normal_equation(X, y)

        # 2. Gradient Descent
        results['gd'] = self.gradient_descent(X, y, learning_rate=0.01)

        # 3. Ridge Regression (different alphas)
        results['ridge_01'] = self.ridge_regression(X, y, alpha=0.1)
        results['ridge_10'] = self.ridge_regression(X, y, alpha=10.0)

        # 4. Lasso Regression (different alphas)
        results['lasso_01'] = self.lasso_regression(X, y, alpha=0.1)
        results['lasso_10'] = self.lasso_regression(X, y, alpha=1.0)

        # 5. Huber Regression
        results['huber'] = self.huber_regression(X, y, delta=1.35)

        # Test set evaluation if provided
        if test_X is not None and test_y is not None:
            print(f"\n📊 Test Set Evaluation:")
            test_X_intercept = np.column_stack([np.ones(test_X.shape[0]), test_X])

            for name, result in results.items():
                test_pred = test_X_intercept @ result['coefficients']
                test_mse = np.mean((test_y - test_pred) ** 2)
                results[name]['test_mse'] = test_mse

        self._print_comparison_table(results)
        self._plot_comparison(X, y, results)

        return results

    def _print_comparison_table(self, results):
        """Print detailed comparison table"""
        print(f"\n📋 DETAILED COMPARISON:")
        print("-" * 100)
        print(f"{'Method':<20} {'Train MSE':<12} {'Test MSE':<12} {'Time (s)':<12} {'Features':<10} {'Comments'}")
        print("-" * 100)

        for name, result in results.items():
            test_mse = result.get('test_mse', 'N/A')
            test_mse_str = f"{test_mse:.4f}" if test_mse != 'N/A' else 'N/A'

            selected_features = result.get('selected_features', len(result['coefficients']) - 1)

            print(f"{result['method']:<20} {result['mse']:<12.4f} {test_mse_str:<12} "
                  f"{result['training_time']:<12.4f} {selected_features:<10} "
                  f"{result.get('pros', '')}")

        print("\n🎯 WHEN TO USE EACH METHOD:")
        print("-" * 50)
        print("OLS Normal Equation: Small datasets, no multicollinearity")
        print("Gradient Descent: Large datasets, streaming data")
        print("Ridge: Multicollinearity, prevent overfitting")
        print("Lasso: Feature selection, sparse models")
        print("Huber: Noisy data with outliers")

    def _plot_comparison(self, X, y, results):
        """Visualize comparison results"""
        n_methods = len(results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Regression Methods Comparison', fontsize=16)

        # Plot 1: Coefficients comparison
        ax = axes[0, 0]
        methods = list(results.keys())
        coefficients_matrix = np.array([results[method]['coefficients'] for method in methods])

        im = ax.imshow(coefficients_matrix, cmap='RdBu', aspect='auto')
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([results[method]['method'] for method in methods])
        ax.set_xlabel('Coefficient Index')
        ax.set_title('Coefficients Heatmap')
        plt.colorbar(im, ax=ax)

        # Plot 2: Training MSE comparison
        ax = axes[0, 1]
        mse_values = [results[method]['mse'] for method in methods]
        method_names = [results[method]['method'] for method in methods]

        bars = ax.bar(range(len(methods)), mse_values)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel('Training MSE')
        ax.set_title('Training MSE Comparison')

        # Highlight best method
        best_idx = np.argmin(mse_values)
        bars[best_idx].set_color('green')

        # Plot 3: Training time comparison
        ax = axes[0, 2]
        time_values = [results[method]['training_time'] for method in methods]

        ax.bar(range(len(methods)), time_values, color='orange')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')

        # Plot 4: Test MSE (if available)
        ax = axes[1, 0]
        test_mse_values = []
        valid_methods = []

        for method in methods:
            if 'test_mse' in results[method]:
                test_mse_values.append(results[method]['test_mse'])
                valid_methods.append(results[method]['method'])

        if test_mse_values:
            bars = ax.bar(range(len(valid_methods)), test_mse_values, color='red', alpha=0.7)
            ax.set_xticks(range(len(valid_methods)))
            ax.set_xticklabels(valid_methods, rotation=45, ha='right')
            ax.set_ylabel('Test MSE')
            ax.set_title('Test MSE Comparison')

            best_test_idx = np.argmin(test_mse_values)
            bars[best_test_idx].set_color('darkgreen')
        else:
            ax.text(0.5, 0.5, 'No test data provided', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Test MSE (No Data)')

        # Plot 5: Predictions scatter (for first method vs true)
        ax = axes[1, 1]
        first_method = list(results.keys())[0]
        y_pred = results[first_method]['predictions']

        ax.scatter(y, y_pred, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'Predictions vs True ({results[first_method]["method"]})')

        # Plot 6: Convergence (if available)
        ax = axes[1, 2]
        convergence_methods = ['gd']  # Methods with convergence history

        for method in convergence_methods:
            if method in results and 'cost_history' in results[method]:
                ax.plot(results[method]['cost_history'], label=results[method]['method'])

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Convergence History')
        ax.legend()

        plt.tight_layout()
        plt.show()


def generate_test_scenarios():
    """Generate different test scenarios to compare methods"""
    scenarios = {}

    # Scenario 1: Clean data, no multicollinearity
    np.random.seed(42)
    X1 = np.random.randn(200, 3)
    y1 = 1.5 + 2 * X1[:, 0] - 1 * X1[:, 1] + 0.5 * X1[:, 2] + 0.1 * np.random.randn(200)
    scenarios['clean'] = (X1, y1, "Clean data")

    # Scenario 2: Multicollinear data
    X2 = np.random.randn(200, 5)
    X2[:, 3] = 0.9 * X2[:, 0] + 0.1 * np.random.randn(200)  # X3 highly correlated with X0
    X2[:, 4] = 0.8 * X2[:, 1] + 0.2 * np.random.randn(200)  # X4 correlated with X1
    y2 = 1 + X2[:, 0] + X2[:, 1] + 0.5 * X2[:, 2] + 0.1 * np.random.randn(200)
    scenarios['multicollinear'] = (X2, y2, "Multicollinear features")

    # Scenario 3: High-dimensional, sparse
    X3 = np.random.randn(100, 20)
    true_coeffs = np.zeros(20)
    true_coeffs[[2, 7, 15]] = [2, -1.5, 1]  # Only 3 features are relevant
    y3 = 0.5 + X3 @ true_coeffs + 0.1 * np.random.randn(100)
    scenarios['sparse'] = (X3, y3, "High-dim sparse")

    return scenarios


def main():
    """Main comparison function"""
    print("🧪 COMPREHENSIVE REGRESSION METHODS COMPARISON")
    print("=" * 60)

    # Generate test scenarios
    scenarios = generate_test_scenarios()

    for scenario_name, (X, y, description) in scenarios.items():
        print(f"\n🔍 SCENARIO: {description}")
        print("=" * 60)

        # Split train/test
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Compare methods
        comparer = RegressionComparison()

        # Test without outliers
        print(f"\n📊 Without outliers:")
        results = comparer.compare_all_methods(X_train, y_train, X_test, y_test)

        # Test with outliers (for robustness)
        if scenario_name == 'clean':
            print(f"\n🔥 With outliers (robustness test):")
            results_outliers = comparer.compare_all_methods(X_train, y_train, X_test, y_test, add_outliers=True)

        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()