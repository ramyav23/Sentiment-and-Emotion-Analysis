#!/usr/bin/env python
# coding: utf-8

# In[1]:


consumer_key='qRTjyYnY0O0zLT9IhJQ4nAL71'
consumer_secret='IViedldBDkljlsNJ0N9MGajLwhe4gf4p7WAJepT5RUgOX9Ued7'
access_token='1432949494660964354-Q2Fe1TRYJoULq9wmtyuFaGkBMmB59l'
access_token_secret='NxonmM1W23TmdMz5DFkLEuMTjl6IoPVxsM1iAf6VKUSVF'


# In[2]:


import tweepy


# In[3]:



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


# In[4]:


import pandas as pd


# In[5]:


coordinates = '19.402833,-99.141051,50mi'
language = 'en'
result_type = 'recent'
until_date = '2021-12-7'
max_tweets = 150
 

tweets = tweepy.Cursor(api.search,screen_name ='@Vitality_UK', geocode=coordinates, lang=language, result_type = result_type, until = until_date, count = 100).items(max_tweets)
tweets


lst = [[tweet.text ,tweet.created_at, tweet.id_str,tweet.coordinates,  tweet.user.screen_name, tweet.user.id_str, tweet.user.location,tweet.retweet_count] for tweet in tweets]
 
tweets_df = pd.DataFrame(lst, columns=['tweet','tweet_dt','tweetid', 'cordinates','username', 'userid', 'geo','retweet_count'])
tweets_df = pd.DataFrame(lst)


# In[6]:


tweets_df


# In[7]:


import datetime
today = datetime.date.today()
yesterday= today - datetime.timedelta(days=1)


# In[8]:


tweets_df
   
    
    
    


# In[9]:


tweets_df.columns=['tweet','tweet_dt','tweetid', 'cordinates','username', 'userid', 'geo','retweet_count']


# In[10]:


max(tweets_df['retweet_count'])


# In[11]:


for i,j in zip(tweets_df['tweet'],tweets_df['retweet_count']):
    if j==max(tweets_df['retweet_count']):
        print("Tweet :",i)
        print("count: ",j)


# In[12]:


from geopy.geocoders import Nominatim




# In[13]:


l=[]


# In[14]:



    
    
    for i in tweets_df['geo']:
        geolocator = Nominatim(user_agent = "geoapiExercises")
        location = geolocator.geocode(i)

        if location is None:
            l.append("None")
        else:

            Country=(list(location))


            c=Country[0].split(',')
            l.append(c[-1])
   


# In[15]:


len(l)


# In[16]:


tweets_df['Country']=l


# In[17]:


tweets_df


# In[18]:


from textblob import TextBlob


# In[19]:


def detect_polarity(text):
    return TextBlob(text).sentiment.polarity
tweets_df['polarity'] = tweets_df.tweet.apply(detect_polarity)


# In[20]:


tweets_df


# In[21]:


m=[]
for i in tweets_df['Country']:
    import unidecode


    m.append(unidecode.unidecode(i))



    


# In[22]:


tweets_df['Country_name']=m
tweets_df['Country_name'] = tweets_df['Country_name'].str.strip()
m=tweets_df['Country_name']


# In[23]:


m


# In[24]:


import pycountry


input_countries = m

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

tweets_df['codes'] = [countries.get(country, 'Unknown code') for country in input_countries]

tweets_df['codes'].str.lower()


# In[25]:


tweets_df


# In[26]:


tweets_df.to_csv('data_new.csv')


# # TILL ABOVE FOR SAVING DATA

# In[27]:



import numpy as np
import random
tweets_df['Sentimenet']=""
Sentimenet = tweets_df.Sentimenet.apply(lambda x: random.choice(['Positive','Negative','Neutral']) )
tweets_df['Sentimenet'] = Sentimenet


# In[28]:


f={}
for i in tweets_df['codes'].unique():
    l=[]
    for j,k in zip(tweets_df['codes'],tweets_df['polarity']):
        if i==j:
            l.append(k)
    f[i]=sum(l)
    l=[]
    


# In[29]:


f


# In[30]:


# del f["Unknown code"]
# print(f)


# In[31]:


Sentimenet


# In[32]:


tweets_df


# In[33]:


C=tweets_df['codes'].unique()
d={}
for i in C:
    l=[]
    for j,k in zip(tweets_df['codes'],tweets_df['Sentimenet']):
        if i==j:
            l.append(k)
    d[i]=l
    l=[]


# In[34]:


d


# In[35]:


s=[]
for i,j in zip(d.keys(),d.values()):
    
    pos=j.count("Positive")
    neg=j.count("Negative")
    neu=j.count("Neutral")
    string={"Positive":pos,"Negative":neg,"Neutral":neu}
    s.append(string)
    string=""


# In[36]:


s


# In[ ]:





# In[37]:


data = {'Sentiment':s,
        'code':list(d.keys()),
                    'polarity':list(f.values())}


# In[38]:


data


# In[39]:


maps= pd.DataFrame(data)


# In[40]:



maps


# In[ ]:





# In[41]:


maps


# In[42]:


import pandas as pd

df2 = pd.json_normalize(maps['Sentiment'])


# In[43]:


df2


# In[44]:


import numpy as np
import matplotlib.pyplot as plt


barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))


Pos = df2['Positive']
Neg = df2['Negative']
Neu = df2['Neutral']


br1 = np.arange(len(Pos))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


plt.bar(br1, Neg, color ='r', width = barWidth,edgecolor ='grey', label ='Negative')
plt.bar(br2, Pos, color ='g', width = barWidth,edgecolor ='grey', label ='Positive')
plt.bar(br3, Neu, color ='b', width = barWidth,edgecolor ='grey', label ='Neutral')


plt.xlabel('Country', fontweight ='bold', fontsize = 15)
plt.ylabel('Polarity Count', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(Pos))],maps.code)

plt.legend()
plt.show()


# In[45]:


str="S"
str.lower()


# In[46]:


js = maps.to_dict(orient = "records")


# In[47]:


js


# In[48]:


# animals = ['dog','cat','fish']
# >>> print(str(animals))
# ['dog', 'cat', 'fish']
import json
print(json.dumps(js))


# In[49]:


tweets_df


# In[ ]:





# In[50]:


maps


# In[51]:


df2


# In[52]:


pd.concat([maps, df2], axis=1, sort=False)


# In[53]:


result = pd.concat([maps, df2], axis=1)


# In[54]:


result


# In[55]:


result["totalcount"]=result['Positive']+result['Negative']+result['Neutral']


# In[56]:


result


# In[57]:


result.nlargest(3,['totalcount'])


# In[58]:


result=result[result.code !='Unknown code']


# In[59]:


result


# In[60]:


Label=[]
for i,j,k in zip(result['Positive'],result['Negative'],result['Neutral']):
    if i>j and i>k:
        Label.append("Positive Tweets")
    elif j>i and j>k:
        Label.append("Negative Tweets")
    elif  k>i and k>j:
        Label.append("Neutral Tweets")
    else:
        Label.append("Neutral Tweets")
        
result["Label"]=Label


# In[61]:


result


# In[62]:


Country=[]
for i in tweets_df["codes"]:
    print(i)


# In[63]:


Country={}
for i,j in zip(tweets_df["Country"].str.strip(),tweets_df["codes"]):
    Country[i]=j
    
   


# In[64]:


Country


# In[65]:


country=[]
for i in result['code']:
    for j,k in zip(Country.values(),Country.keys()):
        if i==j:
            country.append(k)


# In[66]:


country


# In[67]:


result['Country']=country


# In[68]:


result['Country'].str.strip()


# In[69]:


result


# In[70]:


# result.drop(['polarity', 'Positive', 'Negative','Neutral'], axis=1,inplace=True)


# In[ ]:





# In[71]:


result


# In[72]:



Highpos=max(result['Positive'])
Highneg=max(result['Negative'])
Highneu=max(result['Neutral'])
HighTotal=max(result['totalcount'])
for i,j,k,l,m,n,o in zip(result['code'],result['Positive'],result['Negative'],result['Neutral'],result['totalcount'],result['Sentiment'],result['Country']):
    if j==Highpos:
        Codepos=i
        countrypos=o
        sentimentpos=n    
        
        
    if k==Highneg:
        Codeneg=i
        countryneg=o
        sentimentneg= n 
        
    if l==Highneu:
        Codeneu=i
        countryneu=o
        sentimentneu= n 
        
    if m==HighTotal:
        Codetotal=i
        countrytotal=o
        sentimenttotal= n 
        


# In[73]:


new={'Label':['Positive Tweets','Negative Tweets','Neutral Tweets','Total tweets'],'value':[Codepos,Codeneg,Codeneu,Codetotal],'totalCount':[Highpos,Highneg,Highneu,HighTotal],'sentiment':[sentimentpos,sentimentneg,sentimentneu,sentimenttotal],'Country':[countrypos,countryneg,countryneu,countrytotal]}


# In[74]:


table= pd.DataFrame(new)
table


# In[75]:


import plotly.express as px


# In[76]:


tweets_df


# In[77]:


map_fig= px.scatter_geo(tweets_df,
                       locations='codes',
                       projection='orthographic',
                       color='Country_name',
#                        opacity=0.8,
                       hover_name='username',
                       hover_data=['tweet'])


# In[78]:


map_fig.show()


# In[79]:


import plotly.graph_objects as go
import pandas as pd
fig = go.Figure(data=go.Choropleth(
    locations = maps['code'],
    z = maps['polarity'],
    text =maps['Sentiment'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,

    colorbar_title = 'Polarity',
))

fig.update_layout(
    title_text='Global Polarity',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),

)

fig.show()


# In[80]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install plotly')
get_ipython().system('pip install dash')
get_ipython().system('pip install dash_bootstrap_components')


# In[81]:


import pandas as pd
pd.set_option('max_rows',20)
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


# In[82]:


import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


# In[83]:


def globe():
    import plotly.graph_objects as go
    import pandas as pd
    fig = go.Figure(data=go.Choropleth(
        locations = maps['code'],
        z = maps['polarity'],
        text =maps['Sentiment'],
        colorscale = 'Blues',
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.5,

        colorbar_title = 'Polarity',
    ))

    fig.update_layout(
        title_text='Global Polarity',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),

    )

    return fig


# In[84]:


external_stylesheets = [dbc.themes.BOOTSTRAP]


# In[85]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP, 'style.css']
import numpy as np
import pandas as pd 

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server 

# df = pd.read_csv('restaurants_zomato.csv',encoding="ISO-8859-1")

navbar = dbc.Nav()


# In[86]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd


# In[87]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# In[88]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# In[89]:


result


# In[90]:


# country iso with counts
col_label = 'code'
col_values = 'polarity'

v = result[col_label].value_counts()
new = pd.DataFrame({
    col_label: v.index,
    col_values: v.values
})


# In[91]:


hexcode = 0

borders = [hexcode for x in range(len(new))],
map = dcc.Graph(

            id='8',
            figure = {
            'data': [{
            'locations':result['code'],
            'z':result['polarity'],
            'colorscale': 'Earth',
            'reversescale':True,
            'hover-name':result['Country'],
            'type': 'choropleth'
            
            }],
            
            'layout':{'title':dict(
            
                text = 'Polarity Of Each Country',
                font = dict(size=20,
                color = 'white')),
                "paper_bgcolor":"#111111",
                "plot_bgcolor":"#111111",
                "height": 800,
                "geo":dict(bgcolor= 'rgba(0,0,0,0)') } 
                
                })


# In[92]:


# df2 = pd.DataFrame(df.groupby(by='Restaurant Name')['Votes'].mean())
# df2 = df2.reset_index()
df2 = result.sort_values(['Positive'],ascending=False)
df3 = df2.head(3)

bar1 =  dcc.Graph(id='bar1',
              figure={
        'data': [go.Bar(x=df3['Country'],
                        y=df3['Positive'])],
        'layout': {'title':dict(
            text = 'Top Countires by Positive Tweets',
            font = dict(size=20,
            color = 'white')),
        "paper_bgcolor":"#111111",
        "plot_bgcolor":"#111111",
        'height':600,
        "line":dict(
                color="white",
                width=4,
                dash="dash",
            ),
        'xaxis' : dict(tickfont=dict(
            color='white'),showgrid=False,title='Country',color='white'),
        'yaxis' : dict(tickfont=dict(
            color='white'),showgrid=False,title='Number of Positive Tweets',color='white')
    }})


# In[93]:


tweets_df


# In[94]:


pos=tweets_df['Sentimenet'].value_counts().Positive
neu=tweets_df['Sentimenet'].value_counts().Neutral
neg=tweets_df['Sentimenet'].value_counts().Negative


# In[95]:


# Import pandas library
import pandas as pd

# initialize list of lists
data = [['Positive', pos], ['Neutral', neu], ['Negative', neg]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns = ['sentiment', 'count'])

# print dataframe.
df


# In[96]:


# col_label = "Rating text"
# col_values = "Count"

# v = df[col_label].value_counts()
# new2 = pd.DataFrame({
#     col_label: v.index,
#     col_values: v.values
# })

pie3 = dcc.Graph(
        id = "pie3",
        figure = {
          "data": [
            {
            "labels":df['sentiment'],
            "values":df['count'],
              "hoverinfo":"label+percent",
              "hole": .7,
              "type": "pie",
                 'marker': {'colors': [
                                                   '#0052cc',  
                                                   '#3385ff',
                                                   '#99c2ff'
                                                  ]
                                       },
             "showlegend": True
}],
          "layout": {
                "title" : dict(text ="Sentiment Distribution",
                               font =dict(
                               size=20,
                               color = 'white')),
                "paper_bgcolor":"#111111",
                "showlegend":True,
                'height':600,
                'marker': {'colors': [
                                                 '#0052cc',  
                                                 '#3385ff',
                                                 '#99c2ff'
                                                ]
                                     },
                "annotations": [
                    {
                        "font": {
                            "size": 20
                        },
                        "showarrow": False,
                        "text": "",
                        "x": 0.2,
                        "y": 0.2
                    }
                ],
                "showlegend": True,
                "legend":dict(fontColor="white",tickfont={'color':'white' }),
                "legenditem": {
    "textfont": {
       'color':'white'
     }
              }
        } }
)


# In[98]:


import dash
import dash_html_components as html
import base64

image_filename = 'C:/Users/Ramya Venkatesh/OneDrive/Desktop/asset/twitter.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

image_filename2 = 'C:/Users/Ramya Venkatesh/OneDrive/Desktop/asset/twitter1.png' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())


# In[99]:


graphRow1 = dbc.Row([dbc.Col(map,md=12)])
graphRow2 = dbc.Row([dbc.Col(bar1, md=6), dbc.Col(pie3, md=6)])


# In[100]:


app.layout = html.Div([navbar,html.Br(),graphRow1,html.Br(),graphRow2,html.Br(),html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()))], style={'backgroundColor':'black'})


# if __name__ == '__main__':
# #     app.run_server(debug=True,port=8056)
#     app.run_server(host= '0.0.0.0',debug=False)


# In[ ]:



import dash_auth


auth = dash_auth.BasicAuth(
    app,
    {'Waqas' : 'top',
     'ramya': 'secret'}
)


if __name__ == '__main__':
#     app.run_server(debug=True,port=8056)
    app.run_server(host= '0.0.0.0',debug=False)


# In[ ]:


get_ipython().system('pip install dash_auth')


# In[ ]:


http://localhost:8050/


# In[ ]:


get_ipython().system('pip install chart_studio')

