# -------------------------------
# Import libraries
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("C:/Users/dgregson/Downloads/water_potability (1).csv")

# -------------------------------
# 2. Handle missing values
# -------------------------------
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# -------------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------------

# a) Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()

# b) Boxplots for each feature
for feature in df.columns[:-1]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[feature], color='lightgreen')
    plt.title(f'Boxplot of {feature}')
    plt.show()

# c) Histograms by Potability (grid layout)
features = df.columns[:-1]  # exclude target
num_features = len(features)
cols = 3  # number of columns in grid
rows = (num_features + cols - 1) // cols  # calculate rows needed

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(data=df, x=feature, hue="Potability", bins=20, kde=True, palette="Set1", alpha=0.6)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# -------------------------------
# 4. Prepare features and target
# -------------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. Train Random Forest model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 6. Save the trained model
# -------------------------------
joblib.dump(model, "water_quality_model.pkl")
print("\nModel saved as 'water_quality_model.pkl'")


# -------------------------------
# 7. Interactive single-feature menu for potability
# -------------------------------
def check_single_feature_menu(model, df):
    """
    Predict water potability by letting the user pick one feature from a menu.
    All other features are filled with mean values from the dataset.
    """
    features = df.columns[:-1]  # all features except target
    default_values = df[features].mean().to_dict()  # default = mean

    # Show numbered menu
    print("\nChoose one feature to provide:")
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")

    # Get user's choice
    while True:
        try:
            choice = int(input("Enter the number of the feature you want to provide: "))
            if 1 <= choice <= len(features):
                chosen_feature = features[choice - 1]
                break
            else:
                print("Invalid number, choose a number from the menu.")
        except ValueError:
            print("Please enter a valid number.")

    # Ask for the value of that feature
    while True:
        try:
            value = float(input(f"Enter value for {chosen_feature} (default={default_values[chosen_feature]:.2f}): "))
            break
        except ValueError:
            print("Please enter a valid number.")

    # Prepare input with default values for other features
    sample_dict = default_values.copy()
    sample_dict[chosen_feature] = value

    sample_df = pd.DataFrame([sample_dict])
    prediction = int(model.predict(sample_df)[0])
    prob = model.predict_proba(sample_df)[0][prediction]

    if prediction == 1:
        print(f"\nThe water is POTABLE (safe to drink) with confidence {prob:.2f}.")
    else:
        print(f"\nThe water is NOT POTABLE (unsafe to drink) with confidence {prob:.2f}.")


# Run the single-feature menu console
check_single_feature_menu(model, df)
