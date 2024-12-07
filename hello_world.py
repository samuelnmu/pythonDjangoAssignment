import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('iris.csv')
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
print(df.info())



print(df.describe().to_markdown(numalign="left", stralign="left"))
print(df.groupby('Species')['Petal.Length'].mean().to_markdown(numalign="left", stralign="left"))


plt.figure(figsize=(10, 6))
df.groupby('Species')['Petal.Length'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['Petal.Length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['Petal.Length'], df['Petal.Width'], c=df['Species'].map({
    'Iris-setosa': 'red',
    'Iris-versicolor': 'green',
    'Iris-virginica': 'blue'
}))
plt.title('Petal Length vs. Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.grid(True)
plt.show()