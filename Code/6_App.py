#!/usr/bin/env python
# coding: utf-8

# ### Building the web application

# In[1]:


#import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Import dataset
diamonds = pd.read_csv('../Data/diamonds.csv')


# In[3]:


#create the app instance
app = dash.Dash(__name__)


# In[4]:


#import an external CSS file : for our app to look a bit nicer
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


# ### 4. Load the trained objects

# In[5]:


model = joblib.load('diamond-prices-model.joblib')
pca = joblib.load('pca.joblib')
scaler = joblib.load('scaler.joblib')


# ### 5. Build the input components and their respective divs
#  We will use input boxes to input the values of numerical features and drop-down menus for the categorical ones. We will place each of the nine inputs (6 numerical & 3 categorical) in its own four-column div.

# In[6]:


numerical_features = ['price','carat','depth','table','x','y','z']
options_dropdown = [{'label':x.upper(), 'value':x} for x in numerical_features]

dd_x_var = dcc.Dropdown(
        id='x-var',
        options = options_dropdown,
        value = 'carat'
        )

div_x_var = html.Div(
        children=[html.H4('Variable for x axis: '), dd_x_var],
        className="six columns"
        )

dd_y_var = dcc.Dropdown(
        id='y-var',
        options = options_dropdown,
        value = 'price'
        ) 

div_y_var = html.Div(
        children=[html.H4('Variable for y axis: '), dd_y_var],
        className="six columns"
        )


# In[7]:


### 1. For numerical features

## Div for carat
input_carat = dcc.Input(
    id='carat',
    type='numeric',
    value=0.7)

div_carat = html.Div(
        children=[html.H3('Carat:'), input_carat],
        className="four columns"
        )

## Div for depth
input_depth = dcc.Input(
    id='depth',
    placeholder='',
    type='numeric',
    value=60)

div_depth = html.Div(
        children=[html.H3('Depth:'), input_depth],
        className="four columns"
        )

## Div for table
input_table = dcc.Input(
    id='table', 
    placeholder='',
    type='numeric',
    value=60)

div_table = html.Div(
        children=[html.H3('Table:'), input_table],
        className="four columns"
        )

## Div for x
input_x = dcc.Input(
    id='x', 
    placeholder='',
    type='numeric',
    value=5)

div_x = html.Div(
        children=[html.H3('x value:'), input_x],
        className="four columns"
        )

## Div for y
input_y = dcc.Input(
    id='y',
    placeholder='',
    type='numeric',
    value=5)

div_y = html.Div(
        children=[html.H3('y value:'), input_y],
        className="four columns"
        )

## Div for z
input_z = dcc.Input(
    id='z', 
    placeholder='',
    type='numeric',
    value=3)

div_z = html.Div(
        children=[html.H3('z value: '), input_z],
        className="four columns"
        )

### 2. For categorical features

## Div for cut
cut_values = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
cut_options = [{'label': x, 'value': x} for x in cut_values]
input_cut = dcc.Dropdown(
    id='cut',
    options = cut_options,
    value = 'Ideal'
    )

div_cut = html.Div(
        children=[html.H3('Cut:'), input_cut],
        className="four columns"
        )

## Div for color
color_values = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
color_options = [{'label': x, 'value': x} for x in color_values]
input_color = dcc.Dropdown(
    id='color', 
    options = color_options,
    value = 'G'
    )

div_color = html.Div(
        children=[html.H3('Color:'), input_color],
        className="four columns"
        )

## Div for clarity
clarity_values = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
clarity_options = [{'label': x, 'value': x} for x in clarity_values]
input_clarity = dcc.Dropdown(
    id='clarity', 
    options = clarity_options,
    value = 'SI1'
    )

div_clarity = html.Div(
        children=[html.H3('Clarity:'), input_clarity],
        className="four columns"
        )


# In[8]:


#### Further group the nine inputs into three sub-groups & use a div for each group:
## Div for numerical characteristics
div_numerical = html.Div(
        children = [div_carat, div_depth, div_table],
       className="row"
        )

## Div for dimensions
div_dimensions = html.Div(
        children = [div_x, div_y, div_z],
        className="row"
        )

## Div for categorical features
div_categorical = html.Div(
        children = [div_cut, div_color, div_clarity],
        className="row"
        )


# ### 6. Build the prediction function

# In[9]:


def get_prediction(carat, depth, table, x, y, z, cut, color, clarity):
    '''takes the inputs from the user and produces the price prediction'''
    
    cols = ['carat', 'depth', 'table',
            'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good',
            'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
            'clarity_IF','clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2','clarity_VVS1', 'clarity_VVS2',
            'dim_index']

    cut_dict = {x: 'cut_' + x for x in cut_values[1:]}
    color_dict = {x: 'color_' + x for x in color_values[1:]}
    clarity_dict = {x: 'clarity_' + x for x in clarity_values[1:]}
    
    ## produce a dataframe with a single row of zeros
    df = pd.DataFrame(data = np.zeros((1,len(cols))), columns = cols)
    
    ## get the numeric characteristics
    df.loc[0,'carat'] = carat
    df.loc[0,'depth'] = depth
    df.loc[0,'table'] = table
    
    ## transform dimensions into a single dim_index using PCA
    dims_df = pd.DataFrame(data=[[x, y, z]], columns=['x','y','z'])
    df.loc[0,'dim_index'] = pca.transform(dims_df).flatten()[0]
    
    ## Use the one-hot encoding for the categorical features
    if cut!='Fair':
        df.loc[0, cut_dict[cut]] = 1
    
    if color!='D':
        df.loc[0, color_dict[color]] = 1
    
    if clarity != 'I1':
        df.loc[0, clarity_dict[clarity]] = 1
    
    ## Scale the numerical features using the trained scaler
    numerical_features = ['carat', 'depth', 'table', 'dim_index']
    df.loc[:,numerical_features] = scaler.transform(df.loc[:,numerical_features])
    
    ## Get the predictions using our trained neural network
    prediction = model.predict(df.values).flatten()[0]
    
    ## Transform the log-prices to prices
    prediction = np.exp(prediction)
   
    return int(prediction)
    


# ### 7. Create the layout of the application

# In[10]:


# for histogram
trace = go.Histogram(
    x = diamonds['price']
    )

layout = go.Layout(
    title = 'Diamond Prices',
    xaxis = dict(title='Price'),
    yaxis = dict(title='Count')
    )

figure = go.Figure(
    data = [trace],
    layout = layout
    )


# In[11]:


## App layout
app.layout = html.Div([
        html.H1('IDR Predict diamond prices'),
        
        html.H2('Enter the diamond characteristics to get the predicted price'),
        
        html.Div(
                children=[div_numerical, div_dimensions, div_categorical]
                ),
        html.H1(id='output',
                style={'margin-top': '50px', 'text-align': 'center'}),
        html.H3('Interactive scatter plot of the numerical features'),
        html.P('Select your x and y features to view plot'),
        html.Div(
                children=[div_x_var, div_y_var],
                className="row"
                ), 
        dcc.Graph(id='scatter'),
        html.H3('Distribution of diamond prices'),
        html.P('This is the original distribution of the diamond prices.'), 
        dcc.Graph(id='histogram', figure=figure)
        ])


# ### 8.  Build the decorator (callback)

# In[12]:


#for predictor
predictors = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
@app.callback(
        Output('output', 'children'),
        [Input(x, 'value') for x in predictors])

def show_prediction(carat, depth, table, x, y, z, cut, color, clarity): 
    pred = get_prediction(carat, depth, table, x, y, z, cut, color, clarity)
    return str("Predicted Price: {:,}".format(pred))


# In[13]:


# for scatter plot
@app.callback(
        Output(component_id='scatter', component_property='figure'),
        [Input(component_id='x-var', component_property='value'), Input(component_id='y-var', component_property='value')])


def scatter_plot(x_col, y_col):
    trace = go.Scatter(
            x = diamonds[x_col],
            y = diamonds[y_col],
            mode = 'markers'
            )
    
    layout = go.Layout(
            title = 'Scatter plot',
            xaxis = dict(title = x_col.upper()),
            yaxis = dict(title = y_col.upper())
            )
    
    output_plot = go.Figure(
            data = [trace],
            layout = layout
            )
    
    return output_plot


# ### 9. Code to run the server

# In[14]:


if __name__ == '__main__':
    app.run_server(debug=False) #setting debug as false instead of true works


# In[ ]:





# In[ ]:





# In[ ]:




