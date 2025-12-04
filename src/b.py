import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re 
import sys 
from urllib.parse import unquote 

# --- Machine Learning Imports (from your notebook) ---
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

# =============================================================================
# 1. ADVANCED DATA PREPROCESSING ENGINE
# =============================================================================

def dms_to_decimal(dms_str):
    """
    Parses coordinate strings like "48°12′30″N" to decimal degrees.
    Handles various prime/double-prime characters found in your dataset.
    """
    if pd.isna(dms_str): return None
    try:
        # Standardize symbols
        dms_str = str(dms_str).strip().replace("''", '"')
        parts = re.split(r'[°′″"\'’]', dms_str)
        
        degrees = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 and parts[1] else 0
        seconds = float(parts[2]) if len(parts) > 2 and parts[2] else 0
        
        direction = 'N' # Default
        if 'S' in dms_str: direction = 'S'
        elif 'E' in dms_str: direction = 'E'
        elif 'W' in dms_str: direction = 'W'
        
        decimal_val = degrees + (minutes / 60) + (seconds / 3600)
        
        if direction in ['S', 'W']:
            decimal_val *= -1
            
        return decimal_val
    except Exception:
        return None

def load_and_process_data():
    print("--- Starting Data Pipeline ---")
    
    # 1. Load Raw Data
    # Note: city_data.csv uses '|' separator and has headers on row 1
    try:
        df_city = pd.read_csv('city_data.csv', sep='|', header=1)
        df_coords = pd.read_csv('coordinates2.csv') # Standard CSV
        print("Files loaded successfully.")
    except FileNotFoundError:
        print("Error: Files not found. Ensure 'city_data.csv' and 'coordinates2.csv' are present.")
        return pd.DataFrame()

    # 2. Clean Coordinate Data
    # Rename columns to match standard keys if necessary
    if 'Latitude' in df_coords.columns:
        df_coords['lat_decimal'] = df_coords['Latitude'].apply(dms_to_decimal)
        df_coords['lon_decimal'] = df_coords['Longitude'].apply(dms_to_decimal)
    
    # 3. Clean Main City Data
    # Remove extra whitespace from column names
    df_city.columns = df_city.columns.str.strip()
    
    # Split "City, Country" into separate columns
    # We use regex to split on the last comma to handle "Washington, D.C., USA" cases correctly
    split_data = df_city['City'].str.rsplit(', ', n=1, expand=True)
    df_city['City_Name'] = split_data[0]
    df_city['Country_Name'] = split_data[1]

    # 4. Merge Datasets
    # Merging on City Name. 
    df_main = pd.merge(
        df_city,
        df_coords[['City', 'lat_decimal', 'lon_decimal']],
        left_on='City_Name',
        right_on='City',
        how='left'
    )

    # 5. Numeric Conversion & Cleaning
    numeric_cols = [
        'Population', 'Population Density', 'Average Monthly Salary', 
        'Avgerage Rent Price', 'Average Cost of Living', 
        'Average Price Groceries', 'Unemployment Rate', 
        'Days of very strong heat stress'
    ]
    
    for col in numeric_cols:
        if col in df_main.columns:
            # Remove symbols ($, ,, %, etc)
            df_main[col] = df_main[col].astype(str).str.replace(r'[$,%]', '', regex=True)
            df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

    # Drop rows that don't have coordinates (we can't map them)
    df_main.dropna(subset=['lat_decimal', 'lon_decimal'], inplace=True)

    # 6. Machine Learning Imputation (KNN)
    # Using the logic from your notebook to fill missing values
    print("Running KNN Imputation...")
    
    imputer_cols = numeric_cols  # We impute all key numeric columns
    
    # Scale data first (KNN is distance-based, so scale matters)
    scaler = RobustScaler()
    df_scaled = df_main[imputer_cols].copy()
    df_scaled_vals = scaler.fit_transform(df_scaled)
    
    # Impute
    imputer = KNNImputer(n_neighbors=5)
    df_imputed_vals = imputer.fit_transform(df_scaled_vals)
    
    # Inverse Scale
    df_final_vals = scaler.inverse_transform(df_imputed_vals)
    df_main[imputer_cols] = df_final_vals

    # 7. Feature Engineering
    # Metric from your notebook: Disposable Income
    df_main['Disposable Income'] = df_main['Average Monthly Salary'] - df_main['Average Cost of Living']
    
    # Spoken Languages Analysis
    df_main['English Spoken'] = df_main['Main Spoken Languages'].astype(str).apply(
        lambda x: 'English' in x
    )

    print("--- Data Pipeline Complete ---")
    return df_main

# --- Load Data Once on Startup ---
df_main = load_and_process_data()

# =============================================================================
# 2. DASHBOARD APP INITIALIZATION
# =============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# =============================================================================
# 3. LAYOUT DEFINITIONS
# =============================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # -- Header --
    html.Div([
        html.Div([
            html.H1("Global City Analytics", style={'margin': '0', 'color': '#2C3E50', 'fontSize': '28px'}),
            html.P("Cost of Living & Quality of Life Explorer", style={'margin': '5px 0 0 0', 'color': '#7F8C8D'})
        ]),
        html.Img(src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/SARS-CoV-2_without_background.png/220px-SARS-CoV-2_without_background.png", style={'height': '0px', 'display': 'none'}) # Placeholder for logo if needed
    ], style={
        'padding': '20px 40px', 
        'backgroundColor': 'white', 
        'borderBottom': '1px solid #ddd',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
    }),
    
    # -- Main Content --
    html.Div(id='page-content', style={'padding': '30px', 'backgroundColor': '#F4F7F6', 'minHeight': '100vh'})
])


# --- Main Map View Layout ---
layout_map_view = html.Div([
    
    # Top Row: KPI Cards
    html.Div([
        html.Div([
            html.H4("Total Cities", style={'margin': '0', 'color': '#95A5A6', 'fontSize': '14px'}),
            html.H2(f"{len(df_main)}", style={'margin': '5px 0', 'color': '#2C3E50'})
        ], className='kpi-card'),
        
        html.Div([
            html.H4("Avg. Global Salary", style={'margin': '0', 'color': '#95A5A6', 'fontSize': '14px'}),
            html.H2(f"${df_main['Average Monthly Salary'].mean():,.0f}", style={'margin': '5px 0', 'color': '#27AE60'})
        ], className='kpi-card'),
        
        html.Div([
            html.H4("English Speaking %", style={'margin': '0', 'color': '#95A5A6', 'fontSize': '14px'}),
            html.H2(f"{(df_main['English Spoken'].sum() / len(df_main) * 100):.1f}%", style={'margin': '5px 0', 'color': '#2980B9'})
        ], className='kpi-card'),
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px', 'marginBottom': '20px'}),

    # Main Grid: Filters | Map | List
    html.Div([
        
        # 1. Filters Panel
        html.Div([
            html.H3("Filters", style={'color': '#34495E', 'marginBottom': '20px', 'borderBottom': '2px solid #3498DB', 'paddingBottom': '10px'}),
            
            html.Label("Map Color Metric", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='color-metric-dropdown',
                options=[
                    {'label': 'Disposable Income (Salary - Cost)', 'value': 'Disposable Income'},
                    {'label': 'Average Monthly Salary', 'value': 'Average Monthly Salary'},
                    {'label': 'Cost of Living', 'value': 'Average Cost of Living'},
                    {'label': 'Rent Price', 'value': 'Avgerage Rent Price'},
                    {'label': 'Heat Stress Days', 'value': 'Days of very strong heat stress'}
                ],
                value='Disposable Income',
                clearable=False,
                style={'marginBottom': '20px'}
            ),

            html.Label("Population Range", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px'}),
            dcc.RangeSlider(
                id='pop-slider',
                min=df_main['Population'].min(),
                max=df_main['Population'].max(),
                value=[df_main['Population'].min(), df_main['Population'].max()],
                marks={int(x): {'label': f'{int(x/1000)}k', 'style': {'display': 'none'}} for x in np.linspace(df_main['Population'].min(), df_main['Population'].max(), 5)},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
            
            html.Label("Country", style={'fontWeight': 'bold', 'display': 'block', 'marginTop': '30px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': c, 'value': c} for c in sorted(df_main['Country_Name'].unique())],
                multi=True,
                placeholder="All Countries"
            )
            
        ], style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}),
        
        # 2. Map Panel
        html.Div([
            dcc.Graph(id='main-map', style={'height': '600px', 'borderRadius': '12px'}, config={'displayModeBar': False})
        ], style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}),
        
        # 3. City List Panel
        html.Div([
            html.H3("Top Cities", style={'color': '#34495E', 'marginBottom': '15px', 'borderBottom': '2px solid #27AE60', 'paddingBottom': '10px'}),
            html.Div(id='city-list-container', style={'height': '540px', 'overflowY': 'auto'})
        ], style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'})
        
    ], style={'display': 'grid', 'gridTemplateColumns': '1fr 3fr 1fr', 'gap': '25px'})
])


# --- City Detail View Layout ---
def create_city_layout(city_name):
    city_data = df_main[df_main['City_Name'] == city_name].iloc[0]
    
    # Calculate Rank (e.g., Top 10% in salary)
    salary_rank = (df_main['Average Monthly Salary'] > city_data['Average Monthly Salary']).sum() + 1
    total_cities = len(df_main)
    
    return html.Div([
        # Back Button & Title
        html.Div([
            dcc.Link("← Back to Map", href="/", style={'textDecoration': 'none', 'color': '#7F8C8D', 'fontWeight': 'bold', 'fontSize': '16px'}),
            html.Div([
                html.H1(f"{city_name}", style={'margin': '10px 0 5px 0', 'color': '#2C3E50'}),
                html.H4(f"{city_data['Country_Name']}", style={'margin': '0', 'color': '#95A5A6', 'fontWeight': 'normal'})
            ])
        ], style={'marginBottom': '30px'}),
        
        # Comparison Control
        html.Div([
            html.Label("Compare with:", style={'marginRight': '15px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='compare-city-dropdown',
                options=[{'label': c, 'value': c} for c in sorted(df_main['City_Name'].unique()) if c != city_name],
                placeholder="Select a city to compare stats...",
                style={'width': '300px'}
            )
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px', 'display': 'flex', 'alignItems': 'center'}),
        
        # Charts Grid
        html.Div([
            # Left: Gauge
            html.Div([
                dcc.Graph(id='detail-gauge', style={'height': '350px'})
            ], className='chart-card'),
            
            # Right: Bar Chart
            html.Div([
                dcc.Graph(id='detail-bar', style={'height': '350px'})
            ], className='chart-card')
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1.5fr', 'gap': '30px', 'marginBottom': '30px'}),
        
        # Bottom: Economic Stats
        html.Div([
            html.H3("Financial Breakdown", style={'marginBottom': '20px', 'color': '#34495E'}),
            html.Div([
                html.Div([
                    html.P("Monthly Salary", className='stat-label'),
                    html.H3(f"${city_data['Average Monthly Salary']:,.0f}", style={'color': '#27AE60'})
                ], className='stat-box'),
                html.Div([
                    html.P("Rent (Avg)", className='stat-label'),
                    html.H3(f"${city_data['Avgerage Rent Price']:,.0f}", style={'color': '#E74C3C'})
                ], className='stat-box'),
                html.Div([
                    html.P("Groceries", className='stat-label'),
                    html.H3(f"${city_data['Average Price Groceries']:,.0f}", style={'color': '#E67E22'})
                ], className='stat-box'),
                html.Div([
                    html.P("Disposable Income", className='stat-label'),
                    html.H3(f"${city_data['Disposable Income']:,.0f}", style={'color': '#2980B9'})
                ], className='stat-box'),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px'})
        ], style={'backgroundColor': 'white', 'padding': '30px', 'borderRadius': '12px'})
        
    ])

# =============================================================================
# 4. CALLBACKS
# =============================================================================

# -- Router --
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname and pathname.startswith('/city/'):
        city_name = unquote(pathname.split('/')[-1])
        if city_name in df_main['City_Name'].values:
            return create_city_layout(city_name)
        else:
            return html.Div([html.H2("City Not Found"), dcc.Link("Go Back", href="/")])
    return layout_map_view

# -- Map & List Updater --
@app.callback(
    [Output('main-map', 'figure'), Output('city-list-container', 'children')],
    [Input('color-metric-dropdown', 'value'),
     Input('pop-slider', 'value'),
     Input('country-filter', 'value')]
)
def update_map(color_metric, pop_range, selected_countries):
    # Filter
    dff = df_main[
        (df_main['Population'] >= pop_range[0]) & 
        (df_main['Population'] <= pop_range[1])
    ]
    if selected_countries:
        dff = dff[dff['Country_Name'].isin(selected_countries)]
    
    # Logic for colors
    if color_metric == 'Disposable Income':
        color_scale = 'Viridis' # Green/Yellow = Good
        rev = False
    elif 'Cost' in color_metric or 'Rent' in color_metric:
        color_scale = 'Reds' # Red = Expensive
        rev = False
    elif 'Heat' in color_metric:
        color_scale = 'Magma'
        rev = False
    else:
        color_scale = 'Blues'
        rev = False

    # Map Figure
    fig = px.scatter_mapbox(
        dff,
        lat='lat_decimal', lon='lon_decimal',
        color=color_metric,
        size='Population',
        hover_name='City_Name',
        hover_data={
            'Country_Name': True,
            'Average Monthly Salary': ':$,.0f',
            'Average Cost of Living': ':$,.0f',
            'Disposable Income': ':$,.0f',
            'lat_decimal': False, 'lon_decimal': False,
            'Population': False, color_metric: False
        },
        color_continuous_scale=color_scale,
        size_max=25,
        zoom=2,
        mapbox_style='carto-positron'
    )
    fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
    
    # List Items
    list_items = []
    # Sort by the selected metric descending
    dff_sorted = dff.sort_values(by=color_metric, ascending=False).head(50)
    
    for _, row in dff_sorted.iterrows():
        item = html.Div([
            dcc.Link(html.H4(row['City_Name'], style={'margin': '0', 'color': '#2980B9'}), href=f"/city/{row['City_Name']}", style={'textDecoration': 'none'}),
            html.Div([
                html.Span(f"{row['Country_Name']}", style={'fontSize': '12px', 'color': '#95A5A6'}),
                html.Span(f"${row[color_metric]:,.0f}", style={'float': 'right', 'fontWeight': 'bold', 'color': '#2C3E50'})
            ])
        ], style={'padding': '15px 0', 'borderBottom': '1px solid #eee'})
        list_items.append(item)
        
    return fig, list_items

# -- Detail View Updater --
@app.callback(
    [Output('detail-gauge', 'figure'), Output('detail-bar', 'figure')],
    [Input('url', 'pathname'), Input('compare-city-dropdown', 'value')]
)
def update_details(pathname, compare_city):
    if not pathname or not pathname.startswith('/city/'):
        return go.Figure(), go.Figure()
    
    base_city = unquote(pathname.split('/')[-1])
    base_data = df_main[df_main['City_Name'] == base_city].iloc[0]
    
    # 1. Gauge Chart (Cost of Living)
    # -------------------------------
    current_val = base_data['Average Cost of Living']
    delta = None
    gauge_title = f"Cost of Living: {base_city}"
    
    if compare_city:
        comp_data = df_main[df_main['City_Name'] == compare_city].iloc[0]
        ref_val = comp_data['Average Cost of Living']
        delta = {'reference': ref_val, 'relative': False, 'valueformat': '$,.0f'}
        gauge_title = f"Cost: {base_city} vs {compare_city}"

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta" if delta else "gauge+number",
        value = current_val,
        delta = delta,
        title = {'text': gauge_title},
        number = {'prefix': "$"},
        gauge = {
            'axis': {'range': [df_main['Average Cost of Living'].min(), df_main['Average Cost of Living'].max()]},
            'bar': {'color': "#E74C3C"},
            'steps': [
                {'range': [0, 1000], 'color': "#D5F5E3"},
                {'range': [1000, 2000], 'color': "#FCF3CF"}
            ]
        }
    ))
    
    # 2. Bar Chart (Macro Comparison)
    # -------------------------------
    metrics = ['Average Monthly Salary', 'Average Cost of Living', 'Avgerage Rent Price', 'Average Price Groceries']
    pretty_metrics = ['Salary', 'Cost of Living', 'Rent', 'Groceries']
    
    base_vals = [base_data[m] for m in metrics]
    
    if compare_city:
        comp_data = df_main[df_main['City_Name'] == compare_city].iloc[0]
        comp_vals = [comp_data[m] for m in metrics]
        
        fig_bar = go.Figure(data=[
            go.Bar(name=base_city, x=pretty_metrics, y=base_vals, marker_color='#2980B9'),
            go.Bar(name=compare_city, x=pretty_metrics, y=comp_vals, marker_color='#95A5A6')
        ])
        fig_bar.update_layout(title=f"Economic Comparison: {base_city} vs {compare_city}", barmode='group')
    else:
        # Compare vs Global Average
        global_vals = [df_main[m].mean() for m in metrics]
        fig_bar = go.Figure(data=[
            go.Bar(name=base_city, x=pretty_metrics, y=base_vals, marker_color='#2980B9'),
            go.Bar(name="Global Avg", x=pretty_metrics, y=global_vals, marker_color='#BDC3C7')
        ])
        fig_bar.update_layout(title=f"Economic Overview: {base_city} vs Global Avg", barmode='group')

    return fig_gauge, fig_bar

# --- CSS Injection for "Card" Styling ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 0; background-color: #F4F7F6; }
            .kpi-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; }
            .chart-card { background-color: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            .stat-box { text-align: center; padding: 10px; background-color: #F8F9F9; border-radius: 8px; }
            .stat-label { font-size: 12px; color: #7F8C8D; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
