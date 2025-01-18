# Import necessary libraries
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
import os
warnings.filterwarnings("ignore")

# Step 1: Load the Iris dataset

file_path = "bigmart_sales.csv"
data = pd.read_csv(file_path)
output_path = os.path.join("app", "static")
os.makedirs(output_path, exist_ok=True)

# Step 2: Explore the data

print("Data Head:\n", data.head())
print("\nData Info:\n", data.info())
print("\nData Description:\n", data.describe())

# Step 3: Column details

print("\nColumn Details:")
for col in data.columns:
    print(f"Column: {col}, Type: {data[col].dtype}, Unique values: {data[col].nunique()}")

# Step 4: Visualize histograms for each numerical column

data.hist(bins=15, figsize=(10, 8))
plt.suptitle("Histograms for Numerical Columns")
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_histograms.png"))

# Step 5: Handle missing values

# Display the number of missing values per column
print("Number of missing values in each column before filling:")
print(data.isnull().sum())

# Save the mean values for numerical columns with missing values
mean_values_with_na = data.mean(numeric_only=True)

# Save the mode values for categorical columns with missing values
mode_values_with_na = data.mode().iloc[0]

# Fill missing values with the mean of numerical columns and mode of categorical columns
for column in data.columns:
    if data[column].dtype == 'object':  # Categorical columns
        if data[column].isnull().any():
            data[column].fillna(mode_values_with_na[column], inplace=True)
    else:  # Numerical columns
        if data[column].isnull().any():
            data[column].fillna(mean_values_with_na[column], inplace=True)

# Display the mean/mode values used to fill each column with missing values
print("\nValues used to fill columns with missing values:")
for column in data.columns:
    if data[column].dtype == 'object':  # Categorical columns
        if column in mode_values_with_na:
            print(f"{column}: {mode_values_with_na[column]}")
    else:  # Numerical columns
        if column in mean_values_with_na:
            print(f"{column}: {mean_values_with_na[column]}")


# Step 6: Identify and remove outliers using IQR (Interquartile Range)

# Identifying outliers for each numeric column
outlier_columns = []
outliers_dict = {}

# Iterate over each numeric column in the dataset
for feature in data.select_dtypes(include=[np.number]).columns:
    # Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR (Interquartile Range)
    Q1 = data[feature].quantile(0.25)  # 25th percentile
    Q3 = data[feature].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile Range (difference between Q3 and Q1)

    # Identify outliers using the IQR rule: values lower than Q1 - 1.5 * IQR or higher than Q3 + 1.5 * IQR
    outlier_mask = (data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))

    # If there are any outliers for this feature, store them in the outliers_dict
    if outlier_mask.any():  # If there is at least one outlier
        outliers_dict[feature] = data[outlier_mask]  # Store outliers in the dictionary
        outlier_columns.append(feature)  # Add the feature name to the outlier_columns list

# Generate scatter plots only for the columns that contain outliers
for feature in outlier_columns:
    plt.figure(figsize=(8, 6))  # Create a figure for the scatter plot

    # Plot all data points (excluding outliers) for the feature
    non_outliers = data[~outlier_mask]  # Data without outliers
    plt.scatter(non_outliers.index, non_outliers[feature], color='skyblue', label="Data", alpha=0.7)

    # Plot the outliers in red
    outliers = outliers_dict[feature]  # Get the outliers for the current feature
    plt.scatter(outliers.index, outliers[feature], color='red', label="Outliers", s=100, marker='X')

    # Set title and labels for the plot
    plt.title(f"Scatter Plot for {feature} with Outliers in Red")
    plt.xlabel('Index')
    plt.ylabel(feature)
    plt.legend()
    # plt.show()
    plot_filename = f"bigmart_scatter_{feature}.png"  # Use feature name in the file
    plt.savefig(os.path.join(output_path, plot_filename))

    # Print the outliers detected for the current feature
    print(f"\nOutliers Detected for {feature}:")
    print(outliers)

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Item_Fat_Content', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Fat_Content Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_fat.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Item_Visibility', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Visibility Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_visibility.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Item_Type', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Type Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_type.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Item_MRP', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_MRP Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_mrp.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Outlet_Identifier', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Identifier Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_identif.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Outlet_Establishment_Year', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Establishment_Year Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_establishment.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Outlet_Size', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Size Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_size.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Outlet_Location_Type', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Location_Type Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_location.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Outlet_Type', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Type Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_typeoutlet.png"))

# Visualize the original data with a pairplot (including outliers)
sns.pairplot(data, hue='Item_Outlet_Sales', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Outlet_Sales Class Labels (Including Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_outliers_sales.png"))

# Remove the outliers from the dataset

cleaned_data = data.copy()

# Loop through outlier columns to remove outliers
for feature in outlier_columns:
    outliers = outliers_dict[feature]  # Get outliers for the current feature
    cleaned_data = cleaned_data[~cleaned_data.index.isin(outliers.index)]  # Remove rows with outliers

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Item_Fat_Content', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Fat_Content Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_fat.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Item_Visibility', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Visibility Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_visibility.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Item_Type', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Type Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_type.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Item_MRP', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_MRP Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_mrp.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Outlet_Identifier', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Identifier Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_identifier.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Outlet_Establishment_Year', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Establishment_Year Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_establishment.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Outlet_Size', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Size Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_size.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Outlet_Location_Type', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Location_Type Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_location.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Outlet_Type', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Outlet_Type Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_typeoutlet.png"))

# Visualize the cleaned data with a pairplot (after removing outliers)
sns.pairplot(cleaned_data, hue='Item_Outlet_Sales', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Features with Item_Outlet_Sales Class Labels (After Removing Outliers)", y=1.02)
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_removed_outliers_sales.png"))

# Step 8: Preprocess the data

# Encode the target variable
label_encoder = LabelEncoder()
data['Item_Weight'] = label_encoder.fit_transform(data['Item_Weight'])

# Step 9: Handle non-numeric columns in X

# Convert categorical features to numeric using one-hot encoding or label encoding
# Here we use LabelEncoder for each categorical column (excluding the target 'Loan_Status')
for column in data.select_dtypes(include=['object']).columns:
    if column != 'Item_Weight':  # Avoid encoding the target variable
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])

# Step 10: Define features (X) and target (y)

X = data.drop(columns=['Item_Weight'])  # Features
y = data['Item_Weight']  # Target variable

# Step 11: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 12: Feature scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 12: Model selection and training

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Step 13: Model evaluation

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model
}


# Step 14: Display metrics

# Initialize dictionaries for storing metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
conf_matrices = {}
class_reports = {}

# Calculate metrics for each model
for model_name, model in models.items():
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="weighted", zero_division=1)  # Added zero_division=1
    recall = recall_score(y_test, pred, average="weighted", zero_division=1)  # Added zero_division=1
    f1 = f1_score(y_test, pred, average="weighted")

    # Append metric scores
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Store confusion matrices and classification reports if needed
    conf_matrices[model_name] = confusion_matrix(y_test, pred)
    class_reports[model_name] = classification_report(y_test, pred)

# Creating the DataFrame with the metrics
metrics_df = pd.DataFrame({
    "Model": list(models.keys()),
    "Accuracy": accuracy_scores,
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1 Score": f1_scores
})

# Melt the DataFrame for easier plotting with Seaborn
metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Plotting the metrics
plt.figure(figsize=(12, 8))
sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Model Metrics Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Metric")
# plt.show()
plt.savefig(os.path.join(output_path, "bigmart_metrics.png"))

print(conf_matrices)


try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (model_name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=axes[i], cbar=False)
        axes[i].set_title(f"Confusion Matrix: {model_name}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Ground Truth")

    plt.tight_layout()
    # plt.show()
except Exception as e:
    print(f"Error while plotting: {e}")


# Display classification reports
for model_name, report in class_reports.items():
    print(f"\nClassification Report for {model_name}:\n{report}")