import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


class LinearRegressionStats:
    def __init__(self):
        self.weights_ = None
        self.intercept_ = None
        self.coefficients_ = None  # [intercept, weights]
        self.std_errors_ = None
        self.t_stats_ = None
        self.p_values_ = None
        self.r_squared_ = None
        self.adjusted_r_squared_ = None
        self.residuals_ = None
        self.fitted_values_ = None
        self.mse_ = None
        self.rmse_ = None

    def fit(self, X, y):
        # Thêm intercept vào x (bias) => [[1][x]]
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_samples, n_features = X_with_intercept.shape

        # công thức OLS để ước lượng các weight: θ = (X^T X)^(-1) X^T y
        # XTX là Gram matrix, ma trận hiệp phương sai chưa chuẩn hoá giữa các biến độc lập, thể hiển quan hệ biến thiên giữa các biến
        # XTy là vector tổng các tích cho từng biến (hàng)
        XTX = np.dot(X_with_intercept.T, X_with_intercept)
        XTy = np.dot(X_with_intercept.T, y)

        # Kiểm tra đa cộng tuyến, det = 0 -> ma trận suy biến, khả năng đa cộng tuyến cao
        if np.linalg.det(XTX) < 1e-10:
            print("Warning: Matrix is near singular. Possible multicollinearity!")

        self.coefficients_ = np.linalg.solve(XTX, XTy)  # tìm nghiệm θ
        self.intercept_ = self.coefficients_[0] # lấy hằng số là phần từ đầu tiên
        self.weights_ = self.coefficients_[1:]  # còn lại là weight

        # tính giá trị dự đoán và phần dư
        self.fitted_values_ = np.dot(X_with_intercept, self.coefficients_) # tìm y với X và coefficients đã tính ra
        self.residuals_ = y - self.fitted_values_ # lấy giá trị thực tế trừ đi giá trị mới dự đoán để có phần dư

        # tính MSE và RSE
        self.mse_ = np.mean(self.residuals_ ** 2) # trung bình bình phương sai số
        self.rmse_ = np.sqrt(self.mse_) # căn bậc 2 của MSE

        # tính R-squared: cho biết mô hình giải thích được bao nhiêu % biến thiên của dữ liệu
        ss_res = np.sum(self.residuals_ ** 2) # tổng bình phương sai số
        ss_tot = np.sum((y - np.mean(y)) ** 2) # tổng bình phương biến thiên
        self.r_squared_ = 1 - (ss_res / ss_tot)

        # khi thêm bất kỳ biến độc lập nào, R-squared sẽ không bao giờ giảm
        # có thể tăng lên chút ít ngay cả khi biến đó vô nghĩa
        # tính Adjusted R-squared để phạt biến dư thừa, chỉ giữ lại những biến cải thiện mô hình
        n = len(y)
        p = n_features - 1
        self.adjusted_r_squared_ = 1 - (1 - self.r_squared_) * (n - 1) / (n - p - 1)
        # nếu p vô dụng -> SSE giảm rất ít -> n - p - 1  giảm -> SSE/(n - p - 1) tăng -> Adjusted R-squared giảm
        # nếu p hữu dụng -> SSE giảm mạnh hơn mức giảm mẫu số  -> SSE/(n - p - 1) giảm -> Adjusted R-squared tăng

        # Tính độ lệch chuẩn của lỗi
        # phương sai của sai số ước lượng bằng MSE
        # SE(β) = √(MSE * diag((X^T X)^(-1)))
        XTX_inv = np.linalg.inv(XTX) # tính ma trận nghịch đảo của XTX
        var_coef = self.mse_ * np.diag(XTX_inv) # phương sai từng hệ số
        self.std_errors_ = np.sqrt(var_coef) # căn bậc 2 ra độ lệch chuẩn sai số

        # Tính t-statistics: t = β / SE(β)
        # abs(t) càng lớn -> hệ số β càng có khả năng khác 0 -> có ý nghĩa thống kê
        self.t_stats_ = self.coefficients_ / self.std_errors_

        # Tính p-values (kiểm định 2 phía)
        # p-values = xác suất quan sát được abs(t) lớn như vậy nếu H0: β = 0 là đúng
        # hay p-values cho biết giả sử β = 0 thì xác suất có được abs(t) lớn như thế này là bao nhiêu
        # Ý tưởng:
        # - Giả sử H0: β = 0, thì t-stat ~ phân phối Student-t với df = n - p - 1
        # - p-value = P(|T| ≥ |t|) = 2 * (1 - CDF(|t|))
        #   trong đó CDF = hàm phân phối tích lũy của Student-t
        # - p nhỏ (ví dụ < 0.05) -> β gần như chắc chắn khác 0 -> biến có ý nghĩa thống kê
        # p < 0.05 (thường dùng, chuẩn mực chung) -> bác bỏ H0 -> β gần như chắc chắn khác 0 -> hệ số β có ý nghĩa thống kê
        degrees_of_freedom = n - n_features
        self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), degrees_of_freedom))

        return self

    def predict(self, X):
        """Make predictions on new data"""
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return np.dot(X_with_intercept, self.coefficients_)

    def statistical_summary(self):
        """Print comprehensive statistical summary"""
        print("=" * 60)
        print("LINEAR REGRESSION STATISTICAL SUMMARY")
        print("=" * 60)

        print(f"\nModel Performance:")
        print(f"R-squared: {self.r_squared_:.4f}")
        print(f"Adjusted R-squared: {self.adjusted_r_squared_:.4f}")
        print(f"RMSE: {self.rmse_:.4f}")
        print(f"MSE: {self.mse_:.4f}")

        print(f"\nCoefficients Analysis:")
        feature_names = ['Intercept'] + [f'X{i + 1}' for i in range(len(self.weights_))]

        print(
            f"{'Feature':<12} {'Coefficient':<12} {'Std Error':<12} {'t-stat':<10} {'p-value':<10} {'Significance':<12}")
        print("-" * 80)

        for i, (name, coef, se, t_stat, p_val) in enumerate(zip(
                feature_names, self.coefficients_, self.std_errors_, self.t_stats_, self.p_values_
        )):
            significance = ""
            if p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
            elif p_val < 0.1:
                significance = "."

            print(f"{name:<12} {coef:<12.4f} {se:<12.4f} {t_stat:<10.4f} {p_val:<10.4f} {significance:<12}")

        print("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    def check_assumptions(self, X, y, feature_names=None):
        """Check linear regression assumptions"""
        if feature_names is None:
            feature_names = [f'X{i + 1}' for i in range(X.shape[1])]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Linear Regression Assumption Checks', fontsize=16)

        # 1. Residuals vs Fitted (Linearity & Homoscedasticity)
        axes[0, 0].scatter(self.fitted_values_, self.residuals_, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted\n(Check Linearity & Homoscedasticity)')

        # 2. Q-Q Plot (Normality of residuals)
        stats.probplot(self.residuals_, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot\n(Check Normality of Residuals)')

        # 3. Scale-Location Plot (Homoscedasticity)
        standardized_residuals = self.residuals_ / np.std(self.residuals_)
        sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))
        axes[1, 0].scatter(self.fitted_values_, sqrt_abs_residuals, alpha=0.7)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location Plot\n(Check Homoscedasticity)')

        # 4. Residuals vs Leverage (Outliers & Influential points)
        # Calculate leverage (hat values)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
        leverage = np.diag(H)

        axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Standardized Residuals')
        axes[1, 1].set_title('Residuals vs Leverage\n(Check Influential Points)')

        plt.tight_layout()
        plt.show()

        # Statistical tests for assumptions
        print("\n" + "=" * 50)
        print("ASSUMPTION TESTING")
        print("=" * 50)

        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = stats.shapiro(self.residuals_)
        print(f"\n1. Normality of Residuals:")
        print(f"   Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        print(
            f"   {'✓ Residuals are normally distributed' if shapiro_p > 0.05 else '✗ Residuals are NOT normally distributed'}")

        # Breusch-Pagan test for heteroscedasticity (simplified)
        # Regress squared residuals on fitted values
        squared_residuals = self.residuals_ ** 2
        bp_slope = np.corrcoef(self.fitted_values_, squared_residuals)[0, 1]
        print(f"\n2. Homoscedasticity:")
        print(f"   Correlation(fitted_values, residuals²): {bp_slope:.4f}")
        print(
            f"   {'✓ Homoscedasticity assumption satisfied' if abs(bp_slope) < 0.1 else '✗ Heteroscedasticity detected'}")

        # Durbin-Watson test for autocorrelation (if data is time series)
        dw_stat = np.sum(np.diff(self.residuals_) ** 2) / np.sum(self.residuals_ ** 2)
        print(f"\n3. Independence (Durbin-Watson test):")
        print(f"   Durbin-Watson statistic: {dw_stat:.4f}")
        print(f"   {'✓ No autocorrelation' if 1.5 < dw_stat < 2.5 else '⚠ Possible autocorrelation'}")

    def detect_multicollinearity(self, X, feature_names=None):
        """Calculate VIF (Variance Inflation Factor) for multicollinearity detection"""
        if feature_names is None:
            feature_names = [f'X{i + 1}' for i in range(X.shape[1])]

        print("\n" + "=" * 40)
        print("MULTICOLLINEARITY ANALYSIS")
        print("=" * 40)

        vif_scores = []

        for i in range(X.shape[1]):
            # For each feature, regress it on all other features
            X_others = np.delete(X, i, axis=1)
            y_target = X[:, i]

            # Fit regression
            temp_model = LinearRegressionStats()
            temp_model.fit(X_others, y_target)

            # Calculate VIF = 1 / (1 - R²)
            vif = 1 / (1 - temp_model.r_squared_) if temp_model.r_squared_ < 0.999 else np.inf
            vif_scores.append(vif)

        print(f"{'Feature':<12} {'VIF':<10} {'Multicollinearity':<20}")
        print("-" * 45)

        for name, vif in zip(feature_names, vif_scores):
            if vif < 5:
                status = "Low"
            elif vif < 10:
                status = "Moderate"
            else:
                status = "High"

            print(f"{name:<12} {vif:<10.2f} {status:<20}")

        print("\nVIF Interpretation:")
        print("< 5: Low multicollinearity")
        print("5-10: Moderate multicollinearity")
        print("> 10: High multicollinearity")

        return vif_scores


def generate_sample_data(n_samples=100, n_features=3, noise_std=0.1, seed=42):
    """Generate sample dataset for testing"""
    np.random.seed(seed)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # True coefficients
    true_weights = np.array([2.5, -1.3, 0.8])
    true_intercept = 1.2

    # Generate target with some noise
    y = true_intercept + X @ true_weights + np.random.normal(0, noise_std, n_samples)

    # Add some correlation between features to test multicollinearity
    X[:, 2] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)

    return X, y, true_weights, true_intercept


def main():
    """Main function to demonstrate Linear Regression implementation"""
    print("Day 1: Linear Regression & Statistical Foundations")
    print("=" * 60)

    # Generate sample data
    X, y, true_weights, true_intercept = generate_sample_data(n_samples=200, noise_std=0.5)

    print(f"\nDataset Info:")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"True coefficients: {true_weights}")
    print(f"True intercept: {true_intercept}")

    # Fit the model
    model = LinearRegressionStats()
    model.fit(X, y)

    # Print statistical summary
    model.statistical_summary()

    # Check assumptions
    feature_names = ['Height', 'Weight', 'Age']  # Example feature names
    model.check_assumptions(X, y, feature_names)

    # Check multicollinearity
    model.detect_multicollinearity(X, feature_names)

    # Make predictions on new data
    X_new = np.array([[0.5, -0.3, 0.2], [1.1, 0.8, -0.5]])
    predictions = model.predict(X_new)
    print(f"\nPredictions on new data:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i + 1}: {pred:.4f}")


if __name__ == "__main__":
    main()