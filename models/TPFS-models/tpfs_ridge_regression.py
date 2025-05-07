import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns

# Load and filter data
df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS.csv")
df = df[df["exp_type"].isin(["Capital", "Non-Capital"])]
df = df.dropna(subset=["chained_value", "gov_level", "mode", "exp_type"])

# Drop rare categories (< 5 entries in combination)
group_counts = df.groupby(["gov_level", "mode"]).size().reset_index(name="count")
valid_combos = group_counts[group_counts["count"] >= 5][["gov_level", "mode"]]
df = df.merge(valid_combos, on=["gov_level", "mode"])

coef_dict = {}

for spending_type in ["Capital", "Non-Capital"]:
    print(f"\nRidge Regression with Log + Interaction for: {spending_type}")

    sub_df = df[df["exp_type"] == spending_type].copy()
    sub_df["log_spending"] = np.log1p(sub_df["chained_value"])  # log(1 + x) to handle zeros

    # Create interaction terms with patsy
    formula = "log_spending ~ C(gov_level) * C(mode)"
    y, X = dmatrices(formula, data=sub_df, return_type="dataframe")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

    model = make_pipeline(
        StandardScaler(with_mean=False),
        PolynomialFeatures(degree=1, include_bias=False),
        Ridge(alpha=1.0)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2:.3f}")

    # Extract Ridge step coefficients (after polynomial expansion)
    feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out(X.columns)
    ridge_coefs = model.named_steps["ridge"].coef_

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": ridge_coefs
    })

    # Save full coef set and filter by |coef| >= 0.05
    coef_dict[spending_type] = coef_df.copy()
    filtered = coef_df[coef_df["Coefficient"].abs() >= 0.05].sort_values(by="Coefficient", ascending=False)
    print(filtered.to_string(index=False))

    # Actual vs Predicted scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual log Spending")
    plt.ylabel("Predicted log Spending")
    plt.title(f"Actual vs Predicted - {spending_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Bar plot of top coefficients (limit to top 10 by abs value)
    top_10 = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index).head(10)
    top_10.plot(kind="barh", x="Feature", y="Coefficient", legend=False)
    plt.title(f"Top 10 Ridge Coefficients - {spending_type}")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()
    plt.close()

# -------- Extra Visualizations --------

# Heatmap of average log spending by gov_level x mode
df['log_spending'] = np.log1p(df['chained_value'])
pivot = df.pivot_table(index='gov_level', columns='mode', values='log_spending', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Average Log Spending by Government Level and Mode")
plt.tight_layout()
plt.show()
plt.close()

# Boxplot of log spending by government level
# plt.figure(figsize=(8, 5))
# sns.boxplot(data=df, x="gov_level", y="log_spending")
# plt.title("Distribution of Log Spending by Government Level")
# plt.tight_layout()
# plt.show()
# plt.close()

from sklearn.metrics import mean_squared_error

residuals = y_test - model.predict(X_test)
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Actual Log Spending')
plt.ylabel('Residuals')
plt.show()

# sns.violinplot(data=df, x='gov_level', y='log_spending')
# plt.title('Log Spending Density by Government Level')
# plt.xticks(rotation=45)
# plt.show()

sns.barplot(data=df, x="mode", y="log_spending", hue="exp_type", estimator=np.mean)
plt.title("Avg Log Spending by Mode and Type (Capital vs Non-Capital)")
plt.ylabel("Log of Chained Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_interactions = coef_dict["Capital"].query("Coefficient.abs() > 0.5")  # filter key interactions
sns.barplot(data=top_interactions, x="Coefficient", y="Feature", palette="coolwarm")
plt.title("High-Impact Interaction Coefficients (Capital)")
plt.tight_layout()
plt.show()
top_interactions.to_csv("TPFS_ridge_top_coefs.csv", index=False)

g = sns.catplot(data=df, x="mode", y="log_spending", hue="gov_level", kind="box", height=6, aspect=2)
g.fig.suptitle("Log Spending Distribution by Mode and Government Level", y=1.03)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
