import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate biased data with 15,000 rows
n_samples = 15000

# Sensitive attributes
gender = np.random.choice(['Male', 'Female', 'Non-binary'], n_samples, p=[0.5, 0.35, 0.15])
race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n_samples, p=[0.4, 0.25, 0.2, 0.1, 0.05])
age = np.random.randint(18, 80, n_samples)

# Other features
experience = np.random.randint(0, 40, n_samples)
education_years = np.random.randint(12, 22, n_samples)
gpa = np.random.uniform(2.0, 4.0, n_samples)
test_score = np.random.randint(400, 1600, n_samples)

# Create BIASED target variable
# Introduce bias: favoring certain groups
target = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    # Base probability
    base_prob = 0.3 + (experience[i] * 0.02) + (gpa[i] * 0.1)
    
    # Gender bias: Males have higher probability
    if gender[i] == 'Male':
        base_prob += 0.15
    elif gender[i] == 'Non-binary':
        base_prob -= 0.10
    
    # Race bias: White and Asian candidates favored
    if race[i] in ['White', 'Asian']:
        base_prob += 0.12
    elif race[i] == 'Black':
        base_prob -= 0.15
    
    # Age bias: Younger candidates preferred
    if age[i] < 30:
        base_prob += 0.10
    elif age[i] > 55:
        base_prob -= 0.12
    
    # Generate target based on biased probability
    target[i] = 1 if np.random.random() < min(max(base_prob, 0), 1) else 0

# Create DataFrame
data = pd.DataFrame({
    'gender': gender,
    'race': race,
    'age': age,
    'experience': experience,
    'education_years': education_years,
    'gpa': gpa,
    'test_score': test_score,
    'hired': target  # Target variable (1=hired, 0=not hired)
})

print("Dataset shape:", data.shape)
print("\nDataset preview:")
print(data.head(10))
print("\nSensitive attributes distribution:")
print(f"Gender: \n{data['gender'].value_counts()}")
print(f"\nRace: \n{data['race'].value_counts()}")
print(f"\nTarget variable distribution: \n{data['hired'].value_counts()}")
print(f"\nHire rate by gender:\n{data.groupby('gender')['hired'].mean()}")
print(f"\nHire rate by race:\n{data.groupby('race')['hired'].mean()}")

# Split into train (70%) and test (30%) sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

print(f"\nTrain set size: {train_data.shape[0]}")
print(f"Test set size: {test_data.shape[0]}")

# Save train and test CSV files
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
print("\n✓ train.csv and test.csv created successfully")

# Prepare data for model training
X_train = train_data.drop('hired', axis=1)
y_train = train_data['hired']

# Encode categorical variables for model training
le_gender = LabelEncoder()
le_race = LabelEncoder()

X_train_encoded = X_train.copy()
X_train_encoded['gender'] = le_gender.fit_transform(X_train['gender'])
X_train_encoded['race'] = le_race.fit_transform(X_train['race'])

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_encoded, y_train)

print(f"\n✓ Model trained with accuracy on train set: {model.score(X_train_encoded, y_train):.4f}")

# Save model as pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ model.pkl created successfully")

# Verify files were created
print("\nFiles created:")
for file in ['train.csv', 'test.csv', 'model.pkl']:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"  ✓ {file} ({size_mb:.2f} MB)")