# Databricks notebook source
# basic operations
import numpy as np
import pandas as pd 
import re # regular expressions
# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns # seaborn for better graphics

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures



# COMMAND ----------

data = pd.read_csv("/dbfs/FileStore/tables/data.csv", header='infer',dtype='unicode')
print(data.shape)

# COMMAND ----------

data.head()

# COMMAND ----------

#Data Cleaning & Preparation 
#dropping off the unnecessary columns
dfa = data.drop(['Unnamed: 0'],axis=1)

#checking for nulls
dfa.isnull().sum()


# COMMAND ----------

dfa['Overall']=pd.to_numeric(dfa['Overall'])
dfa['Potential']=pd.to_numeric(dfa['Potential'])

dfa['Crossing']=pd.to_numeric(dfa['Crossing'])
dfa['Finishing']=pd.to_numeric(dfa['Finishing'])
dfa['HeadingAccuracy']=pd.to_numeric(dfa['HeadingAccuracy'])
dfa['ShortPassing']=pd.to_numeric(dfa['ShortPassing'])
dfa['Volleys']=pd.to_numeric(dfa['Volleys'])
dfa['Dribbling']=pd.to_numeric(dfa['Dribbling'])
dfa['Curve']=pd.to_numeric(dfa['Curve'])
dfa['FKAccuracy']=pd.to_numeric(dfa['FKAccuracy'])
dfa['LongPassing']=pd.to_numeric(dfa['LongPassing'])
dfa['BallControl']=pd.to_numeric(dfa['BallControl'])
dfa['Acceleration']=pd.to_numeric(dfa['Acceleration'])
dfa['SprintSpeed']=pd.to_numeric(dfa['SprintSpeed'])
dfa['Agility']=pd.to_numeric(dfa['Agility'])
dfa['Reactions']=pd.to_numeric(dfa['Reactions'])
dfa['Balance']=pd.to_numeric(dfa['Balance'])
dfa['ShotPower']=pd.to_numeric(dfa['ShotPower'])
dfa['Jumping']=pd.to_numeric(dfa['Jumping'])
dfa['Stamina']=pd.to_numeric(dfa['Stamina'])
dfa['Strength']=pd.to_numeric(dfa['Strength'])
dfa['LongShots']=pd.to_numeric(dfa['LongShots'])
dfa['Aggression']=pd.to_numeric(dfa['Aggression'])
dfa['Interceptions']=pd.to_numeric(dfa['Interceptions'])
dfa['Positioning']=pd.to_numeric(dfa['Positioning'])
dfa['Vision']=pd.to_numeric(dfa['Vision'])
dfa['Penalties']=pd.to_numeric(dfa['Penalties'])
dfa['Composure']=pd.to_numeric(dfa['Composure'])
dfa['Marking']=pd.to_numeric(dfa['Marking'])
dfa['StandingTackle']=pd.to_numeric(dfa['StandingTackle'])
dfa['SlidingTackle']=pd.to_numeric(dfa['SlidingTackle'])
dfa['GKHandling']=pd.to_numeric(dfa['GKHandling'])
dfa['GKKicking']=pd.to_numeric(dfa['GKKicking'])
dfa['GKPositioning']=pd.to_numeric(dfa['GKPositioning'])
dfa['GKReflexes']=pd.to_numeric(dfa['GKReflexes'])
dfa['GKDiving']=pd.to_numeric(dfa['GKDiving'])



# COMMAND ----------

#filling the missing value for the continous variables for proper data visualization

dfa['ShortPassing'].fillna(dfa['ShortPassing'].mean(), inplace = True)
dfa['Volleys'].fillna(dfa['Volleys'].mean(), inplace = True)
dfa['Dribbling'].fillna(dfa['Dribbling'].mean(), inplace = True)
dfa['Curve'].fillna(dfa['Curve'].mean(), inplace = True)
dfa['FKAccuracy'].fillna(dfa['FKAccuracy'], inplace = True)
dfa['LongPassing'].fillna(dfa['LongPassing'].mean(), inplace = True)
dfa['BallControl'].fillna(dfa['BallControl'].mean(), inplace = True)
dfa['HeadingAccuracy'].fillna(dfa['HeadingAccuracy'].mean(), inplace = True)
dfa['Finishing'].fillna(data['Finishing'].median(), inplace = True)
dfa['Crossing'].fillna(data['Crossing'].median(), inplace = True)
dfa['Contract Valid Until'].fillna(2019, inplace = True)
dfa['Height'].fillna("5'11", inplace = True)
dfa['Loaned From'].fillna('None', inplace = True)
dfa['Position'].fillna('ST', inplace = True)
dfa['Club'].fillna('No Club', inplace = True)
dfa['Work Rate'].fillna('Medium/ Medium', inplace = True)
dfa['Skill Moves'].fillna(dfa['Skill Moves'].median(), inplace = True)
dfa['Preferred Foot'].fillna('Right', inplace = True)
dfa['Wage'].fillna('€200K', inplace = True)

dfa['Marking'].fillna(dfa['Marking'].mean(), inplace = True)
dfa['StandingTackle'].fillna(dfa['StandingTackle'].mean(), inplace = True)
dfa['SlidingTackle'].fillna(dfa['SlidingTackle'].mean(), inplace = True)

dfa['HeadingAccuracy'].fillna(dfa['HeadingAccuracy'].mean(), inplace = True)
dfa['Dribbling'].fillna(dfa['Dribbling'].mean(), inplace = True)
dfa['Curve'].fillna(dfa['Curve'].mean(), inplace = True)
dfa['BallControl'].fillna(dfa['BallControl'].mean(), inplace = True)

dfa['Aggression'].fillna(dfa['Aggression'].mean(), inplace = True)
dfa['Interceptions'].fillna(dfa['Interceptions'].mean(), inplace = True)
dfa['Positioning'].fillna(dfa['Positioning'].mean(), inplace = True)
dfa['Vision'].fillna(dfa['Vision'].mean(), inplace = True)
dfa['Composure'].fillna(dfa['Composure'].mean(), inplace = True)

dfa['Crossing'].fillna(dfa['Crossing'].mean(), inplace = True)
dfa['ShortPassing'].fillna(dfa['ShortPassing'].mean(), inplace = True)
dfa['LongPassing'].fillna(dfa['LongPassing'].mean(), inplace = True)

dfa['Acceleration'].fillna(dfa['Acceleration'].mean(), inplace = True)
dfa['SprintSpeed'].fillna(dfa['SprintSpeed'].mean(), inplace = True)
dfa['Agility'].fillna(dfa['Agility'].mean(), inplace = True)
dfa['Reactions'].fillna(dfa['Reactions'].mean(), inplace = True)

dfa['Balance'].fillna(dfa['Balance'].mean(), inplace = True)
dfa['Jumping'].fillna(dfa['Jumping'].mean(), inplace = True)
dfa['Stamina'].fillna(dfa['Stamina'].mean(), inplace = True)
dfa['Strength'].fillna(dfa['Strength'].mean(), inplace = True)

dfa['Potential'].fillna(dfa['Potential'].mean(), inplace = True)
dfa['Overall'].fillna(dfa['Overall'].mean(), inplace = True)

dfa['Finishing'].fillna(dfa['Finishing'].mean(), inplace = True)
dfa['Volleys'].fillna(dfa['Volleys'].mean(), inplace = True)
dfa['FKAccuracy'].fillna(dfa['FKAccuracy'].mean(), inplace = True)
dfa['ShotPower'].fillna(dfa['ShotPower'].mean(), inplace = True)
dfa['LongShots'].fillna(dfa['LongShots'].mean(), inplace = True)
dfa['Penalties'].fillna(dfa['Penalties'].mean(), inplace = True)

dfa['GKHandling'].fillna(dfa['GKHandling'].mean(), inplace = True)
dfa['GKKicking'].fillna(dfa['GKKicking'].mean(), inplace = True)
dfa['GKPositioning'].fillna(dfa['GKPositioning'].mean(), inplace = True)
dfa['GKReflexes'].fillna(dfa['GKReflexes'].mean(), inplace = True)
dfa['GKDiving'].fillna(dfa['GKDiving'].mean(), inplace = True)

# COMMAND ----------

#formatting the fields
dfa['Value'] = dfa['Value'].str.replace('€','').str.replace('M','000').str.replace('K','')
dfa['Wage'] = dfa['Wage'].str.replace('€','').str.replace('K','')
dfa['Release Clause'] = dfa['Release Clause'].str.replace('€','').str.replace('M','000').str.replace('K','')
#changing datatypes
dfa['Value'] = pd.to_numeric(dfa['Value'])
dfa['Wage'] = pd.to_numeric(dfa['Wage'])
dfa['Release Clause'] = pd.to_numeric(dfa['Release Clause'])

dfa['Value'].fillna(dfa['Value'].mean(), inplace = True)
dfa['Wage'].fillna(dfa['Wage'].mean(), inplace = True)
dfa['Release Clause'].fillna(dfa['Release Clause'].mean(), inplace = True)


# COMMAND ----------




#Drop Columns
#dropping off the unnecessary columns

dfa.drop(['ID','Weak Foot','Photo', 'Flag','Weight','Club Logo', 'International Reputation', 'Body Type', 'Real Face','Jersey Number', 'Joined','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF',  'RW','LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM','CDM', 'RDM', 'RWB', 'LB', 'LCB','CB', 'RCB', 'RB'],
          axis=1,inplace=True)

dfa = dfa.dropna(axis = 0, how = 'any')
dfa.info()

# COMMAND ----------

y= dfa['Value'].values
x = dfa['Wage'].values

fig = plt.figure(figsize=(12,8))
plt.title("How does Market Value relate to Wages (in thousand €)?")
plt.scatter(x,y)
plt.xlabel('Wages')
plt.ylabel('Market Value')
display(plt.show())


# COMMAND ----------

# getting top ten most popular countries
top_ten_countries = dfa['Nationality'].value_counts().head(10).index.values
top_ten_countries_data = dfa.loc[dfa['Nationality'].isin(top_ten_countries), :]

# COMMAND ----------

# How does the distribution of players overall score differ from country to country?
sns.set(style="white")
plt.figure(figsize=(11, 8))
display(sns.boxplot(x = 'Nationality', y = 'Overall', data = top_ten_countries_data))

# COMMAND ----------

def club(x):
    return dfa[dfa['Club'] == x][['Name','Position','Overall','Nationality','Age','Wage',
                                    'Value']]

club('Arsenal')

# COMMAND ----------

# different positions acquired by the players 
plt.figure(figsize = (18, 8))
sns.set(style="ticks")
plt.style.use('fivethirtyeight')
ax = sns.countplot('Position', data = dfa, palette = 'bone')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
display(plt.show())


# COMMAND ----------

# To show Different nations participating in the FIFA 2019

#plt.style.use('dark_background')
dfa['Nationality'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))
plt.title('Different Nations Participating in FIFA 2019', fontsize = 30, fontweight = 20)
plt.xlabel('Name of The Country')
plt.ylabel('count')
display(plt.show())

# COMMAND ----------

#Finding CLubs having highest Rating
cutoff = 85
players = dfa[dfa['Overall']>cutoff]
grouped_players = dfa[dfa['Overall']>cutoff].groupby('Club')
number_of_players = grouped_players.count()['Name'].sort_values(ascending = False)

ax = sns.countplot(x = 'Club', data = players, order = number_of_players.index)

ax.set_xticklabels(labels = number_of_players.index, rotation='vertical')
ax.set_ylabel('Number of players (Over 90)')
ax.set_xlabel('Club')
ax.set_title('Top players (Overall > %.i)' %cutoff)

display(ax)

# COMMAND ----------

# Find Best Squad (Dream Team) 
def get_best_squad(position):
    df_copy = dfa.copy()
    store = []
    for i in position:
        store.append([i,df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False),df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Nationality'].to_string(index = False), df_copy[df_copy['Position'] == i]['Overall'].max()])
        df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace = True)
    #return store
    return pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', 'Nationality','Overall']).to_string(index = False)

# COMMAND ----------

# Best Squad based on 4-3-3 formation
squad_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']
print ('4-3-3')
print (get_best_squad(squad_433))

# COMMAND ----------

# Best Squad based on 3-5-2 formation
squad_352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']
print ('3-5-2')
print (get_best_squad(squad_352))

# COMMAND ----------

# Top five teams with the best players
dfa.groupby(['Club'])['Overall'].max().sort_values(ascending = False).head()

# COMMAND ----------

#Top 10 Players with the best growth potential


# New Column
dfa['Growth'] = dfa['Potential'] - dfa['Overall']


from matplotlib.pyplot import figure
potential_df = dfa.filter(['Name','Age','Overall','Potential'])
potential_df['Growth'] = potential_df['Potential'] - potential_df['Overall']
potential_df = potential_df.sort_values(by=['Growth'],ascending=False)
potential_df = potential_df.iloc[:10]
players_list = list(potential_df.iloc[:, 0])
overall_list = list(potential_df.iloc[:,2])
growth_list = list(potential_df.iloc[:,4])
with plt.style.context('ggplot'):
    figure(num=None, figsize=(15, 8), dpi=80, edgecolor='k')
    plt.barh(players_list, overall_list, color='green', label='Overall')
    plt.barh(players_list, growth_list, left=overall_list, color='green', label='Growth', alpha=0.3)
    plt.legend()
    plt.title('Top 10 Players with the best growth potential')
    plt.xlabel('Overall -------------->')
    display(plt.show())
    
#We notice that A.Dabo has the maximum potential. Dabo currently has an OVR of just over 60 and has the potential to reach around 92-93. He is closely followed by G.Azzinnari and J.von Moos. Managers will surely love to keep an eye on these players.

# COMMAND ----------

#Finding Best Player based on skills
pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
i=0
while i < len(pr_cols):
    print('Best {0} : {1}'.format(pr_cols[i],dfa.loc[dfa[pr_cols[i]].idxmax()][0]))
    i += 1

# COMMAND ----------

# Recommending Similar Player for the Team (Negotiation)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# COMMAND ----------

attributes = dfa.iloc[:, 16:]
attributes['Skill Moves'] = dfa['Skill Moves']
attributes['Age'] = dfa['Age']
workrate = dfa['Work Rate'].str.get_dummies(sep='/ ')
attributes = pd.concat([attributes, workrate], axis=1)
df = attributes
attributes = attributes.dropna()
df['Name'] = dfa['Name']
df['Position'] = dfa['Position']
df = df.dropna()


# COMMAND ----------

#From the correlation chart below, we can see a lot of Goalkeepers attributes have a negative correlation with the attributes possessed by a Forward, Midfielder and Defender

plt.figure(figsize=(9,9))

# Compute the correlation matrix
corr = attributes.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
display(sns.heatmap(corr, mask=mask, cmap="RdBu", vmax=.3, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": .7}))

# COMMAND ----------

scaled = StandardScaler()
X = scaled.fit_transform(attributes)

recommendations = NearestNeighbors(n_neighbors=5,algorithm='kd_tree')
recommendations.fit(X)
player_index = recommendations.kneighbors(X)[1]

# COMMAND ----------

def get_index(x):
    return df[df['Name']==x].index.tolist()[0]

def recommend_similar(player):
    print("These are 4 players similar to {} : ".format(player))
    index=  get_index(player)
    for i in player_index[index][1:]:
        print("Name: {0}\nPosition: {1}\n".format(df.iloc[i]['Name'],df.iloc[i]['Position']))

# COMMAND ----------

#Test 1
recommend_similar('E. Hazard')

# COMMAND ----------

#Test 2
recommend_similar('M. Neuer')

# COMMAND ----------

#Player Valuation with respect to Potential
pot = list(dfa.iloc[:,4])
val = list(dfa['Value'])
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
with plt.style.context('ggplot'):
    plt.scatter(pot,val,s=15,alpha=0.8,c='black')
    plt.xlabel('Potential -------->')
    plt.ylabel('Valuation in millions -------->')
    plt.title('Variation of Valuation with Potential')
    display(plt.show())
    

# COMMAND ----------

# "DREAM TEAM According to status" 
# Best players per each position with their age, club, and nationality based on their overall scores
dream_team=dfa.iloc[dfa.groupby(dfa['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]

# COMMAND ----------
# "DREAM TEAM According to Postion" 
CAM = dream_team.loc[dream_team['Position']=='CAM'].values[0][1]
CB = dream_team.loc[dream_team['Position']=='CB'].values[0][1]
CDM = dream_team.loc[dream_team['Position']=='CDM'].values[0][1]
CF = dream_team.loc[dream_team['Position']=='CF'].values[0][1]
CM = dream_team.loc[dream_team['Position']=='CM'].values[0][1]
GK = dream_team.loc[dream_team['Position']=='GK'].values[0][1]
LAM = dream_team.loc[dream_team['Position']=='LAM'].values[0][1]
LB = dream_team.loc[dream_team['Position']=='LB'].values[0][1]
LCB = dream_team.loc[dream_team['Position']=='LCB'].values[0][1]
LCM = dream_team.loc[dream_team['Position']=='LCM'].values[0][1]
LDM = dream_team.loc[dream_team['Position']=='LDM'].values[0][1]
LF = dream_team.loc[dream_team['Position']=='LF'].values[0][1]
LM = dream_team.loc[dream_team['Position']=='LM'].values[0][1]
LS = dream_team.loc[dream_team['Position']=='LS'].values[0][1]
LW = dream_team.loc[dream_team['Position']=='LW'].values[0][1]
LWB = dream_team.loc[dream_team['Position']=='LWB'].values[0][1] 
RAM = dream_team.loc[dream_team['Position']=='RAM'].values[0][1]
RB = dream_team.loc[dream_team['Position']=='RB'].values[0][1]
RCB = dream_team.loc[dream_team['Position']=='RCB'].values[0][1]
RCM = dream_team.loc[dream_team['Position']=='RCM'].values[0][1] 
RDM = dream_team.loc[dream_team['Position']=='RDM'].values[0][1] 
RF = dream_team.loc[dream_team['Position']=='RF'].values[0][1] 
RM = dream_team.loc[dream_team['Position']=='RM'].values[0][1] 
RS = dream_team.loc[dream_team['Position']=='RS'].values[0][1] 
RW = dream_team.loc[dream_team['Position']=='RW'].values[0][1]
RWB = dream_team.loc[dream_team['Position']=='RWB'].values[0][1]
ST = dream_team.loc[dream_team['Position']=='ST'].values[0][1]

# COMMAND ----------

def create_football_formation(formation = [] , label_1 = None ,
                              label_2 = None , label_3 = None ,
                              label_4 = None,label_4W = None ,
                              label_5 = None , label_3W = None):
    
    plt.scatter(x = [1] , y = [6] , s = 300 , color = 'blue')
    plt.annotate('De Gea \n(Manchester United)' , (1 - 0.5 , 6 + 0.5))
    plt.plot(np.ones((11 , ))*1.5 , np.arange(1 , 12) , 'w-')
    plt.plot(np.ones((5 , ))*0.5 , np.arange(4 , 9) , 'w-')
    
    n = 0
    for posi in formation:
        if posi ==  1:
            n += 3
            dot = plt.scatter(x = [n]  , y = [6] , s = 400 , color = 'white')
            plt.scatter(x = [n]  , y = [6] , s = 300 , color = 'red')
            for i, txt in enumerate(label_1):
                txt = str(txt+'\n('+dfa['Club'][dfa['Name'] == txt].values[0]+')')
                plt.annotate(txt, ( n-0.5 , 6+0.5))
            
        elif posi == 2:
            n += 3
            y = [5 , 7.5]
            x = [ n , n ]
            plt.scatter(x  , y , s = 400 , color = 'white')
            plt.scatter(x  , y , s = 300 , color = 'red')
            for i, txt in enumerate(label_2):
                txt = str(txt+'\n('+dfa['Club'][dfa['Name'] == txt].values[0]+')') 
                plt.annotate(txt, (x[i] - 0.5, y[i]+0.5))
        elif posi == 3:
            n+=3
            y = [3.333 , 6.666 , 9.999]
            x = [n , n  , n ]
            plt.scatter(x  , y , s = 400 , color = 'white')
            plt.scatter(x  , y , s = 300 , color = 'red')
            for i, txt in enumerate(label_3):
                txt = str(txt+'\n('+dfa['Club'][dfa['Name'] == txt].values[0]+')')
                plt.annotate(txt, (x[i] - 0.5, y[i]+0.5))
            
            if not label_3W == None:
                n+=3
                y = [3.333 , 6.666 , 9.999]
                x = [n , n  , n ]
                plt.scatter(x  , y , s = 400 , color = 'white')
                plt.scatter(x  , y , s = 300 , color = 'red')
                for i, txt in enumerate(label_3W):
                    txt = str(txt+'\n('+dfa['Club'][dfa['Name'] == txt].values[0]+')')
                    plt.annotate(txt, (x[i] - 0.5, y[i]+0.5))
            
        elif posi == 4 and not label_4 == None:
            n+=3
            y = [2.5 , 5 , 7.5 , 10]
            x = [n , n  , n , n ]
            plt.scatter(x  , y , s = 400 , color = 'white')
            plt.scatter(x  , y , s = 300 , color = 'red')
            for i, txt in enumerate(label_4):
                txt = str(txt+'\n('+dfa['Club'][dfa['Name'] == txt].values[0]+')')
                plt.annotate(txt, (x[i] - 0.5, y[i]+0.5))
                
            if not label_4W == None:
                n+=3
                y = [2.5 , 5 , 7.5 , 10]
                x = [n , n  , n , n ]
                plt.scatter(x  , y , s = 400 , color = 'white')
                plt.scatter(x  , y , s = 300 , color = 'red')
                for i, txt in enumerate(label_4W):
                    txt = str(txt+'\n('+dfa['Club'][dfa['Name'] == txt].values[0]+')')
                    plt.annotate(txt, (x[i] - 0.5, y[i]+0.5))
                
                
        elif posi == 5:
            n+=3
            y = [2 , 4 , 6 , 8 , 10]
            x = [n , n , n  , n  , n]
            plt.scatter(x  , y , s = 400 , color = 'white')
            plt.scatter(x  , y , s = 300 , color = 'red')
            for i, txt in enumerate(label_5):
                txt = str(txt+'\n('+dfa['Club'][dfa['Name'] == txt].values[0]+')')
                plt.annotate(txt, (x[i] - 0.5, y[i]+0.5))
            
    plt.plot(np.ones((5 , ))*(n+0.5) , np.arange(4 , 9) , 'w-')
    plt.plot(np.ones((11 , ))*(n/2) , np.arange(1 , 12) , 'w-')
    plt.yticks([])
    plt.xticks([])
    ax = plt.gca()
    ax.set_facecolor('tab:green')            

# COMMAND ----------

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 ] , 
                         label_4 = [LWB , LCB , RCB , RWB],
                         label_4W = [LW , LCM , CM , RW],
                         label_2 = [LF , RF],
                         )
plt.title('Best Fit for formation 4-4-2')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 ] , 
                         label_4 = [LB , CB , RCB , RB],
                         label_4W = [LAM , LDM , RDM , RAM],
                         label_2 = [LS , RS],
                         )
plt.title('OR\nBest Fit for formation 4-4-2')
display(plt.show())


plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 ] , 
                         label_4 = [LB , CB , RCB , RB],
                         label_4W = [LW , LDM , RDM , RW],
                         label_2 = [CF , ST],
                         )
plt.title('OR\nBest Fit for formation 4-4-2')
display(plt.show())


plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 ] , 
                         label_4 = [LB , CB , RCB , RB],
                         label_4W = [LW , LCM , RCM , RW],
                         label_2 = [CF , ST],
                         )
plt.title('OR\nBest Fit for formation 4-4-2')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 ] , 
                         label_4 = [LWB , LCB , RCB , RWB],
                         label_4W = [LW , LCM , CM , RW],
                         label_2 = [LF , RF],
                         )
plt.title('OR\nBest Fit for formation 4-4-2')
display(plt.show())


plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 , 3 , 1] , 
                         label_4 = [LWB , LCB , RCB , RWB],
                         label_2 = [LCM , RCM],
                         label_3 = [LF , CAM , RF],
                         label_1 = [ST])
plt.title('Best Fit for formation 4-2-3-1')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 , 3 , 1] , 
                         label_4 = [LWB , LB , RB , RWB],
                         label_2 = [LAM , RAM],
                         label_3 = [LW , CF , RW],
                         label_1 = [ST])
plt.title('OR\nBest Fit for formation 4-2-3-1')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))
create_football_formation(formation = [ 4 , 2 , 3 , 1] , 
                         label_4 = [LWB , CB , RCB , RWB],
                         label_2 = [CM , CAM],
                         label_3 = [LF , CM , RF],
                         label_1 = [ST])
plt.title('OR\nBest Fit for formation 4-2-3-1')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 2 , 3 , 1] , 
                         label_4 = [LWB , LCB , RCB , RWB],
                         label_2 = [LCM , RCM],
                         label_3 = [LDM , CAM , RDM],
                         label_1 = [ST])
plt.title('OR\nBest Fit for formation 4-2-3-1')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 5, 4 , 1 ] , 
                         label_5 = [LWB , LCB , CB , RCB , RWB],
                         label_4 = [LW, LDM , RDM , RW],
                         label_1 = [ST])
plt.title('Best Fit for formation 5-4-1')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 3 ] , 
                         label_4 = [LWB , LCB , RCB , RWB],
                         label_3 = [LW, CAM , RW],
                         label_3W = [LF , ST , RF])
plt.title('Best Fit for formation 4-3-3')
display(plt.show())


plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 3 ] , 
                         label_4 = [LWB , CB , RB , RWB],
                         label_3 = [LAM, CM , RAM],
                         label_3W = [LS , CF , RS])
plt.title('OR\nBest Fit for formation 4-3-3')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 3 ] , 
                         label_4 = [LB , LCB , RCB , RB],
                         label_3 = [LDM, CDM , RDM],
                         label_3W = [LF , CF , RF])
plt.title('OR\nBest Fit for formation 4-3-3')
display(plt.show())

plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 3] , 
                         label_4 = [LWB , CB , RB , RWB],
                         label_3 = [LAM, CAM , RAM],
                         label_3W = [LS , ST , RS])
plt.title('OR\nBest Fit for formation 4-3-3')
display(plt.show())


plt.figure(1 , figsize = (15 , 7))           
create_football_formation(formation = [ 4 , 3] , 
                         label_4 = [LWB , CB , RB , RWB],
                         label_3 = [LCM, CAM , RCM],
                         label_3W = [LF , ST , RF])
plt.title('OR\nBest Fit for formation 4-3-3')
display(plt.show())


# COMMAND ----------

#PREDICTING PLAYER VALUATION USING RANDOM FOREST REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


X = dfa.iloc[:,3:5]
y = dfa.iloc[:,7]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.20, random_state =42)
print('Training matrix of features shape: ', train_X.shape)
print('Training dependent variable shape: ', train_y.shape)
print('Test matrix of features shape: ', test_X.shape)
print('Test dependent variable shape: ', test_y.shape)


# COMMAND ----------

regressor = RandomForestRegressor(n_estimators=10, random_state=42)
regressor.fit(train_X, train_y)
predictions = regressor.predict(test_X)
errors = abs(predictions - test_y)

print('Mean absolute error: ',round(np.mean(errors),2) )
#Mean absolute Percentage error
mape = 100 * (errors/test_y)

#Calculating accuracy
acc = 100 - np.mean(mape)

print('Accuracy: ', round(acc,2), ' %')
