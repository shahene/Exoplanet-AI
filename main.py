import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Replace 'path_to_koi.csv' with the actual path where you saved the KOI table CSV
# For example, if the file is in your Downloads folder: 'Downloads/koi_cumulative.csv'
file_path = './kepler_cumulative.csv'

# Load the CSV file into a pandas DataFrame
df_koi = pd.read_csv(file_path)

# Display the first few rows of the data to get a sense of the structure
numeric_columns = df_koi.select_dtypes(include=['float', 'int']).columns
df_koi[numeric_columns] = df_koi[numeric_columns].fillna(df_koi[numeric_columns].median())

print(df_koi.isnull().sum())

# # Get a summary of the dataset
# print(df_koi.info())
# df_koi_cleaned = df_koi.fillna(df_koi.median())

# # Count the number of confirmed exoplanets
# confirmed_planets = df_koi[df_koi['koi_disposition'] == 'CONFIRMED']
# print(f"Number of confirmed exoplanets: {len(confirmed_planets)}")

# # Check unique dispositions in the dataset
# print(df_koi['koi_disposition'].value_counts())


# plt.hist(df_koi['koi_period'], bins=50)
# plt.title('Distribution of Orbital Periods')
# plt.xlabel('Orbital Period (Days)')
# plt.ylabel('Count')
# plt.show()

# plt.scatter(df_koi['koi_prad'], df_koi['koi_period'])
# plt.title('Planet Radius vs Orbital Period')
# plt.xlabel('Planet Radius (Earth Radii)')
# plt.ylabel('Orbital Period (Days)')
# plt.show()


# features = ['koi_period', 'koi_prad', 'koi_depth', 'koi_impact']
# X = df_koi_cleaned[features]

# # Normalize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
