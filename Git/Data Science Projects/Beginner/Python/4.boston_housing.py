import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import os

# Step 1: Load the Boston Housing dataset
file_path = "Boston_Housing.csv"  # Adjust path if necessary
data = pd.read_csv(file_path)
output_path = os.path.join("app", "static")
os.makedirs(output_path, exist_ok=True)

# Step 2: Check for empty columns (like the last one) and remove them
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Remove the unnamed last column

# Step 3: Explore the data
print("Data Head:\n", data.head())
print("\nData Info:\n", data.info())
print("\nData Description:\n", data.describe())


data.hist(bins=15, figsize=(10, 8))
plt.suptitle("Histograms for Numerical Columns")
# plt.show()
plt.savefig(os.path.join(output_path, "boston_histograms.png"))

# Step 4: Handle missing values (if any)
print("\nNumber of missing values before filling:")
print(data.isnull().sum())

# Handle missing values by filling with the mean for numerical columns
mean_values = data.mean(numeric_only=True)
data.fillna(mean_values, inplace=True)

# Check again for missing values after filling
print("\nNumber of missing values after filling:")
print(data.isnull().sum())

# Step 5: Preprocess the data (Ensure there are no NaNs in features and target)
X = data.drop(columns=['CRIM'])  # Features
y = data['CRIM']  # Target variable

# Ensure there are no NaNs in features (X) and target (y)
if X.isnull().sum().any():
    print("Warning: There are missing values in the feature columns!")
else:
    print("No missing values in feature columns.")

if y.isnull().sum() > 0:
    print("Warning: There are missing values in the target column!")
else:
    print("No missing values in the target column.")

# Step 6: Handle outliers using IQR (Interquartile Range)
outlier_columns = []
outliers_dict = {}

# Identifying outliers for each numeric column
for feature in data.select_dtypes(include=[np.number]).columns:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile Range (difference between Q3 and Q1)
    outlier_mask = (data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))

    if outlier_mask.any():  # If there is at least one outlier
        outliers_dict[feature] = data[outlier_mask]  # Store outliers in the dictionary
        outlier_columns.append(feature)  # Add the feature name to the outlier_columns list

# Generate scatter plots only for the columns that contain outliers
for feature in outlier_columns:
    plt.figure(figsize=(8, 6))
    non_outliers = data[~outlier_mask]  # Data without outliers
    plt.scatter(non_outliers.index, non_outliers[feature], color='skyblue', label="Data", alpha=0.7)
    outliers = outliers_dict[feature]
    plt.scatter(outliers.index, outliers[feature], color='red', label="Outliers", s=100, marker='X')
    plt.title(f"Scatter Plot for {feature} with Outliers in Red")
    plt.xlabel('Index')
    plt.ylabel(feature)
    plt.legend()
    # plt.show()
    plot_filename = f"boston_scatter_{feature}.png"  # Use feature name in the file
    plt.savefig(os.path.join(output_path, plot_filename))
    
    
    print(f"\nOutliers Detected for {feature}:")
    print(outliers)


# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='CRIM', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with CRIM Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "boston_outliers_crim.png"))

# Remove the outliers from the dataset

cleaned_data = data.copy()

# Loop through outlier columns to remove outliers
for feature in outlier_columns:
    outliers = outliers_dict[feature]  # Get outliers for the current feature
    cleaned_data = cleaned_data[~cleaned_data.index.isin(outliers.index)]  # Remove rows with outliers

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='CRIM', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with CRIM Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "boston_removed_outliers_crim.png"))



# Step 7: Preprocess the data (encoding and scaling)
label_encoder = LabelEncoder()
data['CRIM'] = label_encoder.fit_transform(data['CRIM'])

for column in data.select_dtypes(include=['object']).columns:
    if column != 'CRIM':
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])

X = data.drop(columns=['CRIM'])  # Features
y = data['CRIM']  # Target variable

# Step 8: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Model selection and training
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Step 11: Model evaluation
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model
}

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
conf_matrices = {}
class_reports = {}

for model_name, model in models.items():
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="weighted")
    recall = recall_score(y_test, pred, average="weighted")
    f1 = f1_score(y_test, pred, average="weighted")
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    conf_matrices[model_name] = confusion_matrix(y_test, pred)
    class_reports[model_name] = classification_report(y_test, pred)

metrics_df = pd.DataFrame({
    "Model": list(models.keys()),
    "Accuracy": accuracy_scores,
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1 Score": f1_scores
})

metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 8))
sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Model Metrics Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Metric")
# plt.show()
plt.savefig(os.path.join(output_path, "boston_metrics.png"))

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (model_name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=axes[i], cbar=False)
    axes[i].set_title(f"Confusion Matrix: {model_name}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Ground Truth")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(output_path, "boston_metrix.png"))

# Display classification reports
for model_name, report in class_reports.items():
    print(f"\nClassification Report for {model_name}:\n{report}")
