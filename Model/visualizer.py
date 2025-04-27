import json
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback_context, ctx
import urllib

# Load the real estate portfolio data
with open(r'Model\Pre Trained Model\Results\2000000 Initial Capital\100 Properties\real_estate_investment_results.json', 'r') as f:
    data = json.load(f)
properties = pd.DataFrame(data['portfolio'])

# Load geographic data from geopackage files
states_gdf = gpd.read_file('state_level.gpkg')
counties_gdf = gpd.read_file('county_level.gpkg')
cities_gdf = gpd.read_file('city_level.gpkg')

# Create city to county mapping using the city_level geopackage
cities_df = cities_gdf[['NAME', 'COUNTY', 'STATE']].copy()
cities_df['name_lower'] = cities_df['NAME'].str.lower()

# Create mapping dictionary with both city name and state
city_state_to_county = {}
for _, row in cities_df.iterrows():
    key = (row['name_lower'], row['STATE'])
    city_state_to_county[key] = row['COUNTY']

# Apply mapping to properties dataframe
properties['city_lower'] = properties['city'].str.lower()
properties['county'] = properties.apply(
    lambda row: city_state_to_county.get((row['city_lower'], row['state']), None), 
    axis=1
)
properties.drop('city_lower', axis=1, inplace=True)

# Calculate profits by state
state_profits = properties.groupby('state')['monthly_cash_flow'].sum().reset_index()

# Calculate profits by county (for properties with county data)
county_properties = properties.dropna(subset=['county'])
county_profits = county_properties.groupby(['state', 'county'])['monthly_cash_flow'].sum().reset_index()

# Create FIPS code for counties (StateCode + CountyCode)
# We'll need to join with the counties_gdf to get these codes
# First, prepare a mapping of county names to FIPS codes
county_fips_map = {}
for _, row in counties_gdf.iterrows():
    # Assuming counties_gdf has 'name', 'state_name', and 'countyfp' columns
    state_abbr = row.get('stusps')
    county_name = row.get('name')
    county_fips = row.get('geoid')  # This should be the FIPS code
    
    if state_abbr and county_name and county_fips:
        key = (county_name, state_abbr)
        county_fips_map[key] = county_fips

# Add FIPS codes to county_profits
county_profits['fips'] = county_profits.apply(
    lambda row: county_fips_map.get((row['county'], row['state']), None),
    axis=1
)

# Create hover data for states
state_data = state_profits.copy()
state_data['prop_count'] = state_data['state'].apply(
    lambda x: len(properties[properties['state'] == x])
)
state_data['text'] = state_data.apply(
    lambda row: f"Monthly Cash Flow: ${row['monthly_cash_flow']:,.2f}<br>Properties: {row['prop_count']}",
    axis=1
)

# Create hover data for counties
county_data = county_profits.copy()
county_data['prop_count'] = county_data.apply(
    lambda row: len(properties[(properties['state'] == row['state']) & 
                             (properties['county'] == row['county'])]),
    axis=1
)
county_data['text'] = county_data.apply(
    lambda row: f"Monthly Cash Flow: ${row['monthly_cash_flow']:,.2f}<br>Properties: {row['prop_count']}",
    axis=1
)

# Create a figure for the US states map
fig = px.choropleth(
    state_data,
    locations='state',
    locationmode='USA-states',
    color='monthly_cash_flow',
    color_continuous_scale=[
        [0, 'red'],
        [0.49, 'red'],
        [0.5, 'white'],
        [0.51, 'green'],
        [1, 'green']
    ],
    range_color=[-50000, 50000],  # Adjust this range based on your data
    scope="usa",
    labels={'monthly_cash_flow': 'Monthly Cash Flow'},
    custom_data=['text', 'state'],
    template='plotly_white'
)
fig.update_layout(coloraxis_showscale=False)

fig.update_layout(
    showlegend=False,
    coloraxis_showscale=False,
    height=800,
    width=1200,
    margin={"r":0,"t":0,"l":0,"b":0},  # Remove all margins
    autosize=True,
    geo=dict(
        scope="usa",
        projection=dict(type="albers usa"),
        showlakes=True,
        lakecolor="rgb(255, 255, 255)",
        showframe=False  # Removes the frame around the map
    )
)


# Update hover template
fig.update_traces(
    hovertemplate="<b>%{location}</b><br>%{customdata[0]}<extra></extra>"
)

# Create county-level figures - one for each state
state_county_figs = {}
for state in state_data['state'].unique():
    # Filter counties for this state with properties
    state_counties = county_data[county_data['state'] == state].copy()
    
    if len(state_counties) == 0:
        continue
    
    # Get GeoJSON for all counties in this state
    with urllib.request.urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties_geojson = json.load(response)
    
    # Get state FIPS prefix 
    state_fips_prefix = states_gdf[states_gdf['stusps'] == state]['geoid'].iloc[0][:2]
    
    # Filter GeoJSON to include only counties from the selected state
    state_counties_features = [feature for feature in counties_geojson['features'] 
                              if feature['properties']['STATE'] == state_fips_prefix]
    
    state_counties_geojson = {
        'type': 'FeatureCollection',
        'features': state_counties_features
    }
    
    # Create a DataFrame with ALL counties in the state
    all_county_fips = [feature['id'] for feature in state_counties_features]
    all_county_names = [feature['properties']['NAME'] for feature in state_counties_features]
    
    # Create complete DataFrame with all counties
    all_counties_df = pd.DataFrame({
        'fips': all_county_fips,
        'county': all_county_names,
        'state': state,
        'monthly_cash_flow': [0] * len(all_county_fips),  # Default value
        'text': ['No properties'] * len(all_county_fips)  # Default text
    })
    
    # Update with actual data for counties with properties
    for idx, row in state_counties.iterrows():
        match_idx = all_counties_df[all_counties_df['fips'] == row['fips']].index
        if len(match_idx) > 0:
            all_counties_df.loc[match_idx, 'monthly_cash_flow'] = row['monthly_cash_flow']
            all_counties_df.loc[match_idx, 'text'] = row['text']
    
    # Create county figure with ALL counties
    county_fig = px.choropleth(
        all_counties_df,  # Use the complete DataFrame with ALL counties
        geojson=state_counties_geojson,
        locations='fips',
        color='monthly_cash_flow',
        color_continuous_scale=[
            [0, 'red'],
            [0.49, 'red'],
            [0.5, 'white'],
            [0.51, 'green'],
            [1, 'green']
        ],
        range_color=[-50000, 50000],
        scope="usa",
        labels={'monthly_cash_flow': 'Monthly Cash Flow'},
        custom_data=['county', 'text'],
        template='plotly_white'
    )
    
    # Update layout
    county_fig.update_layout(
        coloraxis_showscale=False,
        geo=dict(
            visible=True,
            projection=dict(type="albers usa"),
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
            landcolor='white',       # Base color for counties
            showland=True,           # Show all land areas
            showcountries=False,     
            showsubunits=True,       # Show county borders
            subunitcolor='lightgray', # County border color
            countrywidth=0,
            subunitwidth=0.5,        # Width of county borders
            fitbounds="geojson"      # Fit to all counties in the state
        )
    )
    
    # Store the figure
    state_county_figs[state] = county_fig



# Create a function to generate the appropriate figure based on dropdown selection
def create_figure(selected_state=None):
    if selected_state is None or selected_state == "All States":
        return fig
    else:
        return state_county_figs.get(selected_state, fig)

# Create the initial figure
initial_fig = create_figure()

# Create dropdown options
dropdown_options = [
    {'label': 'All States', 'value': 'All States'}
]
for state in sorted(state_data['state'].unique()):
    if state in state_county_figs:
        dropdown_options.append({'label': state, 'value': state})

app = Dash(__name__)
app.scripts.config.serve_locally = True

app.layout = html.Div([
    html.H1("Real Estate Investment Portfolio - Monthly Cash Flow", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Add back button
    html.Button("← Back to States View", id="back-button", 
                style={'marginBottom': 20, 'display': 'none'}),

    dcc.Graph(
        id='choropleth-map', 
        figure=initial_fig,
        style={'marginLeft': 'auto', 'marginRight': 'auto', 'width': '90%'}
    ),

])

# Add this callback to your app
def display_county_map(clickData):
    if clickData is None:
        return initial_fig
    
    # Extract the state code from clickData
    state_code = clickData['points'][0]['location']
    
    # Check if we have county data for this state
    if state_code in state_county_figs:
        return state_county_figs[state_code]
    else:
        return initial_fig

# Add callback to handle the back button
@app.callback(
    Output('choropleth-map', 'figure'),
    Output('back-button', 'style'),
    Input('back-button', 'n_clicks'),
    Input('choropleth-map', 'clickData'),
    prevent_initial_call=True
)

def update_display(n_clicks, clickData):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'back-button':
        # Return to the state view
        return initial_fig, {'display': 'none'}
    
    elif trigger_id == 'choropleth-map':
        if clickData is None:
            return initial_fig, {'display': 'none'}
        
        # Extract the state code
        state_code = clickData['points'][0]['location']
        
        # Check if we have county data for this state
        if state_code in state_county_figs:
            # Show the back button when displaying county map
            return state_county_figs[state_code], {'display': 'block'}
        else:
            return initial_fig, {'display': 'none'}
    
    # Default case
    return initial_fig, {'display': 'none'}


def update_map(selected_state):
    return create_figure(selected_state)

# If you don't want to use Dash, you can create an HTML file with the dropdown
# using Plotly's updatemenus feature
if __name__ != '__main__':  # If not running the app directly
    # Create a combined figure with updatemenus
    combined_fig = go.Figure(fig)
    
    # Add legend manually
    combined_fig.add_annotation(
        x=0.9, y=0.1, xref="paper", yref="paper",
        text="<b>Legend:</b>", showarrow=False, font=dict(size=12)
    )
    combined_fig.add_annotation(
        x=0.9, y=0.07, xref="paper", yref="paper",
        text="<span style='color:green'>■</span> Positive Cash Flow", showarrow=False, font=dict(size=10)
    )
    combined_fig.add_annotation(
        x=0.9, y=0.04, xref="paper", yref="paper",
        text="<span style='color:red'>■</span> Negative Cash Flow", showarrow=False, font=dict(size=10)
    )
    combined_fig.add_annotation(
        x=0.9, y=0.01, xref="paper", yref="paper",
        text="<span style='color:grey'>■</span> No Properties", showarrow=False, font=dict(size=10)
    )
    
    # Save the HTML
    combined_fig.write_html("real_estate_investment_map_plotly_combined.html")

if __name__ == '__main__':
    app.run(debug=True)