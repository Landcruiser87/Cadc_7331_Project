---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python3 - python
    language: python
    name: ipython_python
---

# The Wonderful World of Coffee
### Business Understanding
This notebook explores a coffee dataset that has been scraped by a reddit user from the Coffee Quality
Institute's website and uploaded onto GitHub.  Our idea behind researching this dataset is to identify where the top
coffee brands come from, and what attributes go into the production of that coffee that makes it so desirable?
Can a model be built targeting those ranges of successful coffee producers in order to predict ratings for 
their future brands?  These are some of the questions we will investigate in our first project.

The data source for our dataset:

https://github.com/jldbc/coffee-quality-database

### Data Description (Meaning/Type/Quality)
Lets import our libraries and data. 


```python
#Add library references
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

```

```python
#Upload Data
df_ar = pd.read_csv('https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv',
                    sep=',', header=0) # read in the arabicaica data
df_rob = pd.read_csv('https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/robusta_data_cleaned.csv',
                     sep=',', header=0) # read in the Robusta data
#Column rename to match for merging
df_ar.rename(columns={'Unnamed: 0':'Id'}, inplace=True)
df_rob.rename(columns={'Unnamed: 0':'Id',
                       'Bitter...Sweet':'Sweetness',
                       'Uniform.Cup':'Uniformity',
                       'Salt...Acid':'Acidity',
                       'Fragrance...Aroma':'Aroma'}, inplace=True)

```

## Data meaning
Below is a list of continuous and categorical measures:
* Category - Description - Range
#### Continuous (Quality) Measures
* Aroma - Smell of the coffee - 1:10
* Flavor - Taste of the coffee - 1:10
* Aftertaste - Residual flavor - 1:10
* Acidity - Acidity of the coffee - 1:10
* Body - How does the coffee feel? - 1:10
* Balance - Flavor/Aroma Balance - 1:10
* Uniformity - Cup to Cup Differences - 1:10
* Cup Cleanliness - How clean is the flavor? 1:10
* Sweetness - Level of sweetness - 1:10
* Moisture - How dry is the flavor? - 1:10
* Defects - Count of any defects - 0:63
* Cupper Points - Overall Rating 1:10
* Total Cup Points - Total Cup Rating - 1:100

#### Categorical (Bean) Measures
* Processing Method - How was the bean processed?
* Color - What is its color?
* Species (arabica / robusta)

#### Categorical (Farm) Measures
* Owner
* Country of Origin
* Farm Name
* Lot Number
* Mill
* Company
* Altitude
* Region

Since the data came to us in two CSV's of arabica and robusta, lets combine the two datasets to begin our analysis.  First we'll need to remove 
a few columns and merge the two dataframes.

```python

#dropping columns we won't use
df_rob = df_rob.drop(['Lot.Number', 'altitude_low_meters', 'altitude_high_meters', 'Certification.Contact',
                      'Certification.Contact', 'Expiration', 'Certification.Body', 'ICO.Number',
                      'Certification.Address','Mouthfeel', 'Id'], axis=1)
df_ar = df_ar.drop(['Lot.Number', 'altitude_low_meters', 'altitude_high_meters', 'Certification.Contact',
                      'Certification.Contact', 'Expiration', 'Certification.Body', 'ICO.Number',
                      'Certification.Address', 'Id'], axis=1)

df_comb = df_ar.append(df_rob)

```

### Data Quality & Simple Statistics
Now that our dataframes are combined, lets analyze counts of missing values and simple statistics.
#### Missing Values

```python
print("Structure of data:\n",df_comb.shape,"\n")
print("Count of missing values:\n",df_comb.isnull().sum().sort_values(ascending=False),"\n")
```

The majority of missing values center around farm name, mill, producer, altitude, company.  At this stage, we need
to decide what categorical values we can keep for our analysis.  Country/region might be one
of the better attributes to start an analysis due to its low missing value count.   Categorical classification might 
not be as successful with farm name, color and processing method as they are missing quite a few values.  We won't 
target those for analysis. The continuous data has very low NA counts which means any regression, will likely 
fair well.  One area we think influences coffee production is altitude_mean_meters.  To clean that column, we'll be be 
replacing blank and altitude means of 1 because we believe them to be a data entry error.  We also removed a row that 
didn't have a country of origin. 


```python
#Changing datatypes
conv_dict = {'Species': str,
                'Owner': str,
                'Mill': str,
                'Company': str,
                'Region': str,
                'Producer': str,
                'Variety': str
                }
df_comb = df_comb.astype(conv_dict)

#Outlier Removal altitude
df_comb = df_comb.replace({'altitude_mean_meters': {1: df_comb['altitude_mean_meters'].mean()}})
df_comb.loc[[896,1040,1144,543],'altitude_mean_meters'] = np.nan
df_comb['altitude_mean_meters'].fillna((df_comb['altitude_mean_meters'].mean()),inplace=True)

#nan removal from country
df_comb = df_comb.drop(df_comb.index[1197])
#second check of missing values to ensure integrity. 
print("Count of missing values:\n",df_comb.isnull().sum().sort_values(ascending=False),"\n")


```

### Simple Statistics
As any good Data Scientist, we first must check our data ranges, means, max's, mins, and quartiles, to see where
the data sits.  This is an important step because it allows another view of possible faulty or outlier data within our
dataset.  Most of our continuous variables range from 1:10 for the coffee property ratings and have no missing values.  


```python
#Simple Stats
# print(df_comb.head().append(df_comb.tail()), "\n")
print("Summary Statistic's:\n",round(df_comb.describe(),2),"\n")

```

### Visualize Attributes
Now that we've got our data organized and a little cleaner, lets begin visualizing our data. We'll start with a 
bar chart of the Number of coffee samples by Country.


```python
counts = df_comb['Country.of.Origin'].value_counts().to_dict()
min_count = min(counts.values())
max_count = max(counts.values())

#Bar graph of number of coffee samples per country, top 50
counts_top50 = dict(list(counts.items())[int(len(counts)/2):])
plt.figure(figsize=(20,10))
plt.bar(range(len(counts)), list(counts.values()), align='center')
plt.xticks(range(len(counts)), list(counts.keys()), rotation=90)
plt.title("Number of Coffee Samples by Country", fontsize='18')
```

With Mexico coming in first, we're not surprised by those that follow it.  Columbia, Guatamala, Brazil, Taiwan are
all excellent climates for growing coffee beans.  This matches with common knowledge of larger coffee producing
countries.  Another interesting side-note is that Hawaii produces 3x as many coffee varieties as the mainland US.  


Now that we've got an idea behind who are the biggest producers, lets see who produces some of the better brands 
via the reviewers "Total Cup Score."  We'll group by country and analyze their average score.

```python
#Dataframe of countries by avg Cup rating

country_lists=list(df_comb['Country.of.Origin'].unique())
country_total_cup_ratio=[]
for each in country_lists:
    country=df_comb[df_comb['Country.of.Origin']==each]
    country_total_cup_avg=round(sum(country['Total.Cup.Points'])/len(country),2)
    country_total_cup_ratio.append(country_total_cup_avg)

    
data=pd.DataFrame({'Country of Origin':country_lists,'Total Cup Avg':country_total_cup_ratio})
new_index=(data['Total Cup Avg'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
sorted_data.head()
```

Here we see a few differing results as alot of the central american countries are lower than expected in 
terms of overall coffee rating.  The best coffee it would seem comes from Papua New Guinea, Ethiopia, and Japan, 
and the United States.  

Now that we've got a few countries of interest, lets delve into the continuous variables themselves to get an 
idea behind their distributions.  So first a frequency plot of the continuous variables we initially predict to 
be relevant.  We'll also do a pairplot to view some scatterplots of the same variables.


```python
#some initial plots 
col_names = ['Aroma','Aftertaste', 'Aroma','Balance', 
             'Flavor', 'Acidity','Moisture', 'Cupper.Points', 'Total.Cup.Points']

fig, ax = plt.subplots(len(col_names), figsize=(16,12))

for i, col_val in enumerate(col_names):

    sns.distplot(df_comb[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq dist '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    ax[i].set_ylabel('Count', fontsize=8)

plt.show()

# #Huge pairplot matrix.  Probably need to whittle down the attributes a bit first.Example drops below
sns.pairplot(df_comb, vars = ['Aroma','Aftertaste', 'Aroma','Balance', 
             'Flavor', 'Acidity','Moisture', 'Cupper.Points', 'Total.Cup.Points'],
             hue = 'Species');
```

Now that we've got a better higher level view of the data, we begin to see clustering in the continuous
variables around the 6 to 9 rating mark.  Add to that our range for the rating scale is 1 to 10 and we begin to see
some problems with our dataset.  Mainly since our variables have a very similar distribution, we could end up with
a highly correlated dataset.  So our next step is to look at some individual histograms and
a correlation heat map to confirm our suspicions.

```python
df_num = df_comb.select_dtypes(include=['float64'])
df_num.hist(figsize =(14,12))
#Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
f, ax = plt.subplots(figsize=(10, 10))
corr = df_comb.corr()
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".1f",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
```

There is some correlation to be concerned with, but for now lets just keep it in mind.  

```python
country_lists=list(df_comb['Country.of.Origin'].unique())
country_total_cup_ratio=[]
for each in country_lists:
    country=df_comb[df_comb['Country.of.Origin']==each]
    country_total_cup_avg=round(sum(country['Total.Cup.Points'])/len(country),2)
    country_total_cup_ratio.append(country_total_cup_avg)

    
data=pd.DataFrame({'Country of Origin':country_lists,'Total Cup Avg':country_total_cup_ratio})
new_index=(data['Total Cup Avg'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
sorted_data.head()
```

```python
# Now lets look at comparisons of attributes. Lets look at quality (total cup) vs country
# to do
import matplotlib.colors
plt.figure(figsize = (20,10))
cmap = plt.cm.coolwarm_r
norm = matplotlib.colors.Normalize(vmin = 1, vmax = 30)
plt.bar(height=sorted_data['Total Cup Avg']-75,
        x = sorted_data['Country of Origin'],
        align = 'center',bottom = 75,
       color = cmap(norm(sorted_data['Total Cup Avg'].values-75)))
plt.xticks(rotation=90)
plt.title("Total coffee score by country", fontsize='18')
```

```python
def countrysorter(frame = df_comb,col='Total.Cup.Points'):
    contlist = list(frame['Country.of.Origin'].unique())
    contval = []
    for each in contlist:
        cont = frame[frame['Country.of.Origin']==each]
        contavg =  round(sum(cont[col])/len(cont),2)
        contval.append(contavg)


    dat = pd.DataFrame({'Country.of.Origin':contlist,'avgval':contval})
    nindex = (dat['avgval'].sort_values(ascending = False)).index.values
    sordat = dat.reindex(nindex)
    return(sordat)
```

```python



def countryplotter(dat, sub = 0,
                   nmin = 1, nmax = 30,tit = 'TODO'):
    plt.figure(figsize = (20,10))
    cmap = plt.cm.coolwarm_r
    norm = matplotlib.colors.Normalize(vmin = nmin, vmax = nmax)
    plt.bar(height = dat['avgval']-sub,
            x = dat['Country.of.Origin'],
            align = 'center',
            bottom = sub,
            color = cmap(norm(dat['avgval'].values-sub)))
    plt.xticks(rotation = 90)
    plt.title(tit,fontsize = 18)


```

```python
countryplotter(dat = countrysorter(col = 'Acidity'),sub = 7,nmin = 2, nmax = 30, tit = 'Acidity by Country')
```

```python
#shit
fig, ax = plt.subplots()
for c, df in df_comb.groupby('Country.of.Origin'):
    ax.scatter(df['Clean.Cup'], df['Total.Cup.Points'], label = c)
ax.legend()
```

```python
(sns 
 .FacetGrid(df_comb, hue='Country.of.Origin',height = 10)
 .map(plt.scatter, 'Clean.Cup', 'Total.Cup.Points')
 .add_legend()
 .set(
    title='Clean cup vs total score grouped by country',
    xlabel='cup cleanliness',
    ylabel='score'
))

```

```python
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax  = plt.subplots(figsize = (30,30))
pnts  =  ax.scatter(x=np.log(df_comb['Acidity']), y =  np.log(df_comb['Aroma']), c = df_comb['Total.Cup.Points'], s  = (df_comb['Total.Cup.Points']-60)*4, cmap =cmap)

f.colorbar(pnts)
```

```python
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax  = plt.subplots(figsize = (30,30))
pnts  =  ax.scatter((df_comb['altitude_mean_meters']), y =  (df_comb['Aroma']), c = df_comb['Total.Cup.Points'], s  = (df_comb['Total.Cup.Points']-60)*8, cmap =cmap)

f.colorbar(pnts)
```

```python
df_comb[df_comb.Quakers==0]['Total.Cup.Points'].hist(density = True)
df_comb[df_comb.Quakers==1]['Total.Cup.Points'].hist(density = True)
```

```python
(sns
    .FacetGrid(df_comb,
        hue = 'Quakers',
        height = 10)
    .map(sns.kdeplot,'Total.Cup.Points',shade = True)
            .add_lengend()
)
```

```python
fig, ax  = plt.subplots()

for quaker in df_comb['Quakers'].unique():
    s = df_comb[df_comb['Quakers'] == quaker]['Total.Cup.Points']
    s.plot.kde(ax=ax, label = quaker)
ax.legend()
```

```python
fig, ax  = plt.subplots()
fig.set_size_inches(14,14)

ax = (sns
    .violinplot(x = "Quakers",
               y = "Total.Cup.Points",
               data = df_comb,
               )
)
```

```python
def violinplotter(cat,cont="Total.Cup.Points",dat=df_comb):
    fig, ax  = plt.subplots()
    fig.set_size_inches(14,14)

    ax = (sns
        .violinplot(x = cat,
                   y = cont,
                   data = dat,
                   )
    )
```

Data meaning

Below is a list of continuous and categorical measures:
Continuous (Quality) Measures

    Aroma
    Flavor
    Aftertaste
    Acidity
    Body
    Balance
    Uniformity
    Cup Cleanliness
    Sweetness
    Moisture
    Defects
    Cupper Points
    Total Cup Points

Categorical (Bean) Measures

    Processing Method
    Color
    Species (arabica / robusta)

Categorical (Farm) Measures

    Owner
    Country of Origin
    Farm Name
    Lot Number
    Mill
    Company
    Altitude
    Region


```python
violinplotter(cat = "Processing.Method")
violinplotter(cat = "Color")
violinplotter("Species")
violinplotter("Owner")

violinplotter("Processing.Method","Clean.Cup")
```

```python
#Outlier Removal altitude
df_comb = df_comb.replace({'altitude_mean_meters': {1: df_comb['altitude_mean_meters'].mean()}})
df_comb.loc[[896,1040,1144,543],'altitude_mean_meters'] = np.nan
df_comb['altitude_mean_meters'].fillna((df_comb['altitude_mean_meters'].mean()),inplace=True)
plt.style.available
```

```python
#colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
colors = ["windows blue", "amber", "dusty purple"]

cmap = matplotlib.colors.ListedColormap(sns.xkcd_palette(colors).as_hex())
#cmap = matplotlib.colors.ListedColormap(sns.color_palette("RdBu_r",100).as_hex())
with plt.style.context(('fast')):
    f, ax  = plt.subplots(figsize = (30,30))
    pnts  =  ax.scatter((df_comb['Flavor']), y =  np.log(df_comb['Total.Cup.Points']), c = (df_comb['altitude_mean_meters']),
                        s = 100, cmap =cmap)

    f.colorbar(pnts)
```
