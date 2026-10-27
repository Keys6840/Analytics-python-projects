# -------------------------------------------
#IMPORT LIBRARIES
# -------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
df_netflix= pd.read_csv("D:/FIIB/Term 2/DAP/Group/netflix_titles.csv") 

df_netflix.head()

#--------------------------------------------
# Handle missing values in important columns
#--------------------------------------------
df_netflix['country'] = df_netflix['country'].fillna('Unknown')
df_netflix['rating'] = df_netflix['rating'].fillna('Unknown')
df_netflix['duration'] = df_netflix['duration'].fillna('Unknown')

# -------------------------------------------
# Separate Movies and TV Shows (for later analysis)
# -------------------------------------------
movies = df_netflix[df_netflix['type'] == 'Movie'].copy()
shows  = df_netflix[df_netflix['type'] == 'TV Show'].copy()

print("Total titles:", len(df_netflix))
print("Movies:", len(movies))
print("TV Shows:", len(shows))

movies.head()

# -------------------------------------------
#First we will do What analysis using three graphs
#First one is
#-------------------------------------------
counts = [len(movies), len(shows)]
labels = ['Movies', 'TV Shows']
colors = ['orange', 'green']

plt.figure(figsize=(6,6))
plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
plt.title("Distribution of Movies vs TV Shows on Netflix")
plt.show()

# -------------------------------------------
#Second What Analysis Graph 
#-------------------------------------------
all_genres = []

for x in df_netflix['listed_in']:
    for g in x.split(', '):
        all_genres.append(g)

top_genres = pd.Series(all_genres).value_counts().head(10)
print(top_genres)

plt.figure()
plt.bar(top_genres.index, top_genres.values, color='orange')
bars = plt.bar(top_genres.index, top_genres.values, color='orange')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             int(bar.get_height()), ha='center', va='bottom')
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Genre")
plt.ylabel("Content Volume")
plt.xticks(rotation=45)

plt.show()

# -------------------------------------------
#Third What Analysis Graph
#-------------------------------------------
rating_counts = df_netflix['rating'].value_counts().head(10)

plt.figure()
plt.bar(rating_counts.index, rating_counts.values , color='green')
bars = plt.bar(rating_counts.index, rating_counts.values, color='green')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             int(bar.get_height()), ha='center', va='bottom')
plt.title("Top 10 Content Ratings on Netflix")
plt.xlabel("Rating")
plt.ylabel("Content Volume")
plt.xticks(rotation=45)

plt.show()

# -------------------------------------------
#Now we will do the Where Analysis
#-------------------------------------------
countries_list = []

for x in df_netflix['country']:
    if pd.isna(x):
        countries_list.append("Unknown")
    else:
        countries_list.append(x.split(',')[0].strip())

countries_series = pd.Series(countries_list)
top_countries = countries_series.value_counts().head(10)

print(top_countries)

plt.figure()
plt.bar(top_countries.index, top_countries.values,color ='green') 
bars = plt.bar(top_countries.index, top_countries.values, color='green')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             int(bar.get_height()), ha='center', va='bottom')
plt.title("Top 10 Countries by Content Volume on Netflix")
plt.xlabel("Country")
plt.ylabel("Content Volume")  
plt.xticks(rotation=45)
plt.ioff()
plt.show()

# -------------------------------------------
#Now we Will do the WHEN Analysis
#-------------------------------------------
year_list = []

for x in df_netflix['date_added']:
    if pd.isna(x):
        continue
    else:
        year_list.append(pd.to_datetime(x).year)

year_series = pd.Series(year_list)
titles_by_year = year_series.value_counts().sort_index()

print(titles_by_year)
plt.figure()
plt.plot(titles_by_year.index, titles_by_year.values, marker='o',color='orange')
bars = plt.plot(titles_by_year.index, titles_by_year.values, marker='o', color='orange')
for i in range(len(titles_by_year.index)):
    plt.text(titles_by_year.index[i], titles_by_year.values[i], 
             int(titles_by_year.values[i]), ha='center', va='bottom')
plt.title("Content Volume Added to Netflix by Year")
plt.xlabel("Date Added")
plt.ylabel("Content Volume")
plt.grid(True)

plt.show()

# -------------------------------------------
# Now Why analysis
# -------------------------------------------
print(df_netflix['type'].value_counts())

#-------------------------------------------
#Descriptive Statistics
#-------------------------------------------
print("\n==================== DESCRIPTIVE STATISTICS: Movie Duration ====================")
print("Most Common Type:", df_netflix['type'].mode()[0])
print("Most Common Rating:", df_netflix['rating'].mode()[0])
print("Most Common Country:", df_netflix['country'].mode()[0])

#Count of Movies vs TV Shows, Ratings, Countries
print("\nMovies vs TV Shows:")
print(df_netflix['type'].value_counts())

print("\nTop 5 Ratings:")
print(df_netflix['rating'].value_counts().head(5))

print("\nTop 5 Countries:")
print(df_netflix['country'].value_counts().head(5))

# -------------------------------------------
#Correlation Analysis
# -------------------------------------------
print("\n==================== CORRELATION ANALYSIS ====================")
# Count titles per release year
release_counts = df_netflix['release_year'].value_counts().sort_index()

# Convert into DataFrame for correlation
corr_df = pd.DataFrame({
    'release_year': release_counts.index,
    'title_count': release_counts.values
})
corr_matrix = corr_df.corr()
print(corr_matrix)
plt.figure()
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap: Release Year vs Content Volume")
plt.show()

# -------------------------------------------
#regression Year vs No of Titles
# -------------------------------------------
print("\n==================== REGRESSION: Year Added vs Content Volume ====================")

release_counts = df_netflix['release_year'].value_counts().sort_index()
print(release_counts.head())
X = release_counts.index.values.reshape(-1, 1)   # Release Year (Independent)
y = release_counts.values                        # Number of Titles (Dependent)

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
slope     = model.coef_[0]
r2        = model.score(X, y)

print("Intercept (β0):", intercept)
print("Slope (β1):", slope)
print("R-squared:", r2)

y_pred = model.predict(X)

plt.figure(figsize=(10,5))
plt.scatter(X.flatten(), y,color ='green', label='Actual Data')
plt.plot(X.flatten(), y_pred,color ='red', label='Regression Line')
plt.legend()
plt.xlabel("Release Year")
plt.ylabel("Content Volume")
plt.title("Regression: Release Year vs Content Volume")
plt.ioff()
plt.show()