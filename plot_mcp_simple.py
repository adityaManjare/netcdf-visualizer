import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load your CSV ---
csv_file = "your_file.csv"  # Replace with your CSV path
df = pd.read_csv(csv_file)

# --- Example: Plot Pressure (PRES) vs Level ---
sns.set(style="whitegrid")

plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='level', y='PRES', hue='profile', marker='o')
plt.title("Pressure (PRES) vs Level")
plt.xlabel("Level")
plt.ylabel("Pressure (PRES)")
plt.legend(title="Profile")
plt.tight_layout()
plt.show()

# --- Example: Plot Temperature (TEMP) vs Level ---
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='level', y='TEMP', hue='profile', marker='o')
plt.title("Temperature (TEMP) vs Level")
plt.xlabel("Level")
plt.ylabel("Temperature (°C)")
plt.legend(title="Profile")
plt.tight_layout()
plt.show()

# --- Example: Plot Salinity (PSAL) vs Level ---
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='level', y='PSAL', hue='profile', marker='o')
plt.title("Salinity (PSAL) vs Level")
plt.xlabel("Level")
plt.ylabel("Salinity")
plt.legend(title="Profile")
plt.tight_layout()
plt.show()
