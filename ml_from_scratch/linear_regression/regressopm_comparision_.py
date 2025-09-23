import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


class LinearRegressionComparison:
    """
    Compare different methods to solve Linear Regression:
    1. OLS (Normal Equation) - Analytical solution
    2. Gradient Descent - Iterative optimization
    3. SGD (Stochastic Gradient Descent) - Online learning
    4. SVD (Singular Value Decomposition) - Numerically stable
    5. QR Decomposition - Numerically stable alternative
    6. Ridge Regression - Regularized solution
    """

    def __init__(self):
        self.methods = {}
        self.results = {}

    def ols_normal_equation(self, X, y):
        """
        Method 1: OLS using Normal Equation
        θ = (X^T X)^(-1) X^T y

        Pros: Fast, exact solution, no hyperparameters
        Cons: Requires matrix inversion (expensive for large datasets)
              Fails when X^T X is singular
        """
        start_time = time.time()

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        try:
            # Normal equation
            XTX = np.dot(X_with_intercept.T, X_with_intercept)
            XTy = np.dot(X_with_intercept.T, y)
            theta = np.linalg.solve(XTX, XTy)  # More stable than inv(XTX) @ XTy

            # Calculate MSE
            predictions = X_with_intercept @ theta
            mse = np.mean((y - predictions) ** 2)

            training_time = time.time() - start_time

            return {
                'method': 'OLS (Normal Equation)',
                'coefficients': theta,
                'mse': mse,
                'training_time': training_time,
                'converged': True
            }
        except np.linalg.LinAlgError:
            return {
                'method': 'OLS (Normal Equation)',
                'coefficients': None,
                'mse': np.inf,
                'training_time': time.time() - start_time,
                'converged': False,
                'error': 'Singular matrix - use regularization or other methods'
            }

    def gradient_descent(self, X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Method 2: Gradient Descent
        θ = θ - α * ∇J(θ)

        Pros: Works with large datasets, memory efficient
        Cons: Requires tuning learning_rate, slower convergence
        """
        start_time = time.time()

        # Add intercept and normalize features
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_samples, n_features = X_with_intercept.shape

        # Initialize parameters
        theta = np.random.normal(0, 0.01, n_features)
        costs = []

        for iteration in range(max_iterations):
            # Forward pass
            predictions = X_with_intercept @ theta
            cost = np.mean((predictions - y) ** 2) / 2
            costs.append(cost)

            # Gradient calculation: ∇J = (1/m) * X^T * (h - y)
            gradient = (1 / n_samples) * X_with_intercept.T @ (predictions - y)

            # Update parameters
            theta_new = theta - learning_rate * gradient

            # Check convergence
            if np.linalg.norm(theta_new - theta) < tolerance:
                theta = theta_new
                break

            theta = theta_new

        final_predictions = X_with_intercept @ theta
        mse = np.mean((y - final_predictions) ** 2)

        return {
            'method': 'Gradient Descent',
            'coefficients': theta,
            'mse': mse,
            'training_time': time.time() - start_time,
            'iterations': iteration + 1,
            'converged': iteration < max_iterations - 1,
            'cost_history': costs
        }

    def stochastic_gd(self, X, y, learning_rate=0.01, max_epochs=100, batch_size=32):
        """
        Method 3: Stochastic Gradient Descent (Mini-batch)
        Update parameters using small batches of data

        Pros: Very memory efficient, works with streaming data
        Cons: Noisy convergence, requires more tuning
        """
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_samples, n_features = X_with_intercept.shape

        # Initialize parameters
        theta = np.random.normal(0, 0.01, n_features)
        costs = []

        for epoch in range(max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]

            epoch_cost = 0
            n_batches = 0

            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Mini-batch gradient
                predictions = X_batch @ theta
                batch_cost = np.mean((predictions - y_batch) ** 2) / 2
                gradient = (1 / len(X_batch)) * X_batch.T @ (predictions - y_batch)

                # Update parameters
                theta -= learning_rate * gradient

                epoch_cost += batch_cost
                n_batches += 1

            costs.append(epoch_cost / n_batches)

        final_predictions = X_with_intercept @ theta
        mse = np.mean((y - final_predictions) ** 2)

        return {
            'method': 'Stochastic GD',
            'coefficients': theta,
            'mse': mse,
            'training_time': time.time() - start_time,
            'epochs': max_epochs,
            'converged': True,  # SGD doesn't have clear convergence criterion
            'cost_history': costs
        }

    def svd_solution(self, X, y):
        """
        Method 4: SVD (Singular Value Decomposition)
        More numerically stable than normal equation

        X = U * Σ * V^T
        θ = V * Σ^(-1) * U^T * y

        Pros: Handles singular/near-singular matrices better
        Cons: More computationally expensive than normal equation
        """
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # SVD decomposition
        U, sigma, Vt = np.linalg.svd(X_with_intercept, full_matrices=False)

        # Handle small singular values (numerical stability)
        sigma_inv = np.where(sigma > 1e-10, 1 / sigma, 0)

        # Calculate coefficients: θ = V * Σ^(-1) * U^T * y
        theta = Vt.T @ np.diag(sigma_inv) @ U.T @ y

        predictions = X_with_intercept @ theta
        mse = np.mean((y - predictions) ** 2)

        return {
            'method': 'SVD Solution',
            'coefficients': theta,
            'mse': mse,
            'training_time': time.time() - start_time,
            'converged': True,
            'condition_number': sigma[0] / sigma[-1] if sigma[-1] > 1e-10 else np.inf
        }

    def qr_decomposition(self, X, y):
        """
        Method 5: QR Decomposition
        X = Q * R, then solve R * θ = Q^T * y

        Pros: Numerically stable, good for tall matrices
        Cons: Slightly slower than normal equation
        """
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # QR decomposition
        Q, R = np.linalg.qr(X_with_intercept)

        # Solve R * θ = Q^T * y
        QTy = Q.T @ y
        theta = np.linalg.solve(R, QTy)

        predictions = X_with_intercept @ theta
        mse = np.mean((y - predictions) ** 2)

        return {
            'method': 'QR Decomposition',
            'coefficients': theta,
            'mse': mse,
            'training_time': time.time() - start_time,
            'converged': True
        }

    def ridge_regression(self, X, y, alpha=1.0):
        """
        Method 6: Ridge Regression (L2 Regularization)
        θ = (X^T X + αI)^(-1) X^T y

        Pros: Handles multicollinearity, prevents overfitting
        Cons: Biased estimates, requires tuning α
        """
        start_time = time.time()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X_with_intercept.shape[1]

        # Ridge modification: add α to diagonal (except intercept)
        XTX = X_with_intercept.T @ X_with_intercept
        regularization_matrix = alpha * np.eye(n_features)
        regularization_matrix[0, 0] = 0  # Don't regularize intercept

        # Ridge solution
        theta = np.linalg.solve(XTX + regularization_matrix, X_with_intercept.T @ y)

        predictions = X_with_intercept @ theta
        mse = np.mean((y - predictions) ** 2)

        return {
            'method': f'Ridge Regression (α={alpha})',
            'coefficients': theta,
            'mse': mse,
            'training_time': time.time() - start_time,
            'converged': True,
            'regularization': alpha
        }

    def compare_all_methods(self, X, y, verbose=True):
        """Compare all methods and return results"""
        methods_to_run = [
            ('ols', lambda: self.ols_normal_equation(X, y)),
            ('gradient_descent', lambda: self.gradient_descent(X, y)),
            ('sgd', lambda: self.stochastic_gd(X, y)),
            ('svd', lambda: self.svd_solution(X, y)),
            ('qr', lambda: self.qr_decomposition(X, y)),
            ('ridge', lambda: self.ridge_regression(X, y, alpha=1.0))
        ]

        results = {}

        if verbose:
            print("🔍 COMPARING LINEAR REGRESSION METHODS")
            print("=" * 80)
            print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print()

        for method_name, method_func in methods_to_run:
            try:
                result = method_func()
                results[method_name] = result

                if verbose:
                    print(f"✅ {result['method']:<25}")
                    print(f"   MSE: {result['mse']:.6f}")
                    print(f"   Training Time: {result['training_time']:.4f}s")
                    print(f"   Coefficients: {result['coefficients']}")
                    if 'converged' in result:
                        print(f"   Converged: {result['converged']}")
                    print()

            except Exception as e:
                if verbose:
                    print(f"❌ {method_name} failed: {str(e)}")
                    print()

        return results

    def plot_convergence(self, results):
        """Plot convergence for iterative methods"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Gradient Descent convergence
        if 'gradient_descent' in results and 'cost_history' in results['gradient_descent']:
            gd_costs = results['gradient_descent']['cost_history']
            axes[0].plot(gd_costs, label='Gradient Descent', linewidth=2)
            axes[0].set_title('Gradient Descent Convergence')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Cost (MSE/2)')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)

        # SGD convergence
        if 'sgd' in results and 'cost_history' in results['sgd']:
            sgd_costs = results['sgd']['cost_history']
            axes[1].plot(sgd_costs, label='Stochastic GD', color='orange', linewidth=2)
            axes[1].set_title('Stochastic GD Convergence')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Average Cost per Epoch')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def analyze_when_to_use_what(self, X, y):
        """Analyze which method to use in different scenarios"""
        print("\n🧠 WHEN TO USE WHICH METHOD:")
        print("=" * 60)

        n_samples, n_features = X.shape
        condition_number = np.linalg.cond(X)

        print(f"Dataset characteristics:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Condition number: {condition_number:.2f}")
        print()

        recommendations = []

        # Size-based recommendations
        if n_samples > 10000 and n_features > 1000:
            recommendations.append("🚀 Large dataset → Use Stochastic GD or Mini-batch GD")
        elif n_samples < 1000:
            recommendations.append("📊 Small dataset → OLS Normal Equation is fine")

        # Condition number based
        if condition_number > 1000:
            recommendations.append("⚠️  High condition number → Use SVD or Ridge Regression")

        # Memory based
        if n_samples > 100000:
            recommendations.append("💾 Memory concerns → Use SGD (online learning)")

        # Multicollinearity based
        correlations = np.corrcoef(X.T)
        max_corr = np.max(np.abs(correlations - np.eye(n_features)))
        if max_corr > 0.8:
            recommendations.append("🔗 High multicollinearity → Use Ridge or SVD")

        for rec in recommendations:
            print(rec)

        if not recommendations:
            print("✅ Standard dataset → OLS Normal Equation works well")


def create_test_scenarios():
    """Create different scenarios to test methods"""
    np.random.seed(42)
    scenarios = {}

    # Scenario 1: Normal case
    X1 = np.random.randn(100, 3)
    y1 = 1 + 2 * X1[:, 0] - 1.5 * X1[:, 1] + 0.8 * X1[:, 2] + 0.1 * np.random.randn(100)
    scenarios['normal'] = (X1, y1, "Normal Case")

    # Scenario 2: Multicollinear features
    X2 = np.random.randn(100, 3)
    X2[:, 2] = 0.9 * X2[:, 0] + 0.1 * np.random.randn(100)  # X3 ≈ X1
    y2 = 1 + 2 * X2[:, 0] - 1.5 * X2[:, 1] + 0.8 * X2[:, 2] + 0.1 * np.random.randn(100)
    scenarios['multicollinear'] = (X2, y2, "Multicollinear Features")

    # Scenario 3: Large dataset
    X3 = np.random.randn(5000, 10)
    true_coef = np.random.randn(10)
    y3 = X3 @ true_coef + 0.5 * np.random.randn(5000)
    scenarios['large'] = (X3, y3, "Large Dataset")

    return scenarios


def main():
    """Main comparison function"""
    print("🎯 LINEAR REGRESSION METHODS COMPARISON")
    print("=" * 80)

    scenarios = create_test_scenarios()

    for scenario_name, (X, y, description) in scenarios.items():
        print(f"\n📊 SCENARIO: {description}")
        print("-" * 50)

        comparator = LinearRegressionComparison()
        results = comparator.compare_all_methods(X, y)

        # Plot convergence for iterative methods
        if scenario_name == 'normal':  # Only plot for first scenario
            comparator.plot_convergence(results)

        # Analysis and recommendations
        comparator.analyze_when_to_use_what(X, y)

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

# =============================================================================
# DEEP DIVE: WHY OLS vs ALTERNATIVES?
# =============================================================================

"""
🤔 TẠI SAO DÙNG OLS (ORDINARY LEAST SQUARES)?

1. MATHEMATICAL ELEGANCE:
   - Có closed-form solution: θ = (X^T X)^(-1) X^T y
   - Không cần hyperparameter tuning
   - Exact solution (not approximation)

2. STATISTICAL PROPERTIES (GAUSS-MARKOV THEOREM):
   - BLUE: Best Linear Unbiased Estimator
   - Minimum variance among all unbiased linear estimators
   - If assumptions met → optimal solution

3. COMPUTATIONAL EFFICIENCY:
   - O(n³) for matrix inversion, but very fast for small-medium datasets
   - No iterative process needed

KHI NÀO KHÔNG DÙNG OLS?

❌ CASE 1: Large datasets (n > 100K)
   → Matrix inversion becomes expensive O(n³)
   → Use: Gradient Descent, SGD

❌ CASE 2: Singular/near-singular X^T X  
   → Normal equation fails
   → Use: SVD, Ridge Regression

❌ CASE 3: Online/streaming data
   → Can't store all data in memory
   → Use: SGD, online algorithms

❌ CASE 4: Multicollinearity
   → Unstable coefficient estimates
   → Use: Ridge, Lasso, PCA

❌ CASE 5: Outliers present
   → OLS sensitive to outliers (squared loss)
   → Use: Robust regression, Huber loss

INTERVIEW INSIGHT:
"OLS is optimal WHEN assumptions are met, but real-world data often violates assumptions.
That's why we need alternatives!"
"""