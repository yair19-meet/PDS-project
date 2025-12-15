import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from urllib.parse import unquote 

# =============================================================================
# 1. OPTIMIZED DATA LOADING
# =============================================================================

def load_and_process_data():
    print("--- Loading Cleaned Data ---")
    
    try:
        # Load the final dataset
        df_main = pd.read_csv('merged_data.csv')
        print("merged_data.csv loaded successfully.")
    except FileNotFoundError:
        print("Error: 'merged_data.csv' not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()

    # 1. Rename columns to match the dashboard's internal names
    df_main.rename(columns={
        'City': 'City_Name',
        'Country': 'Country_Name',
        'Latitude': 'lat_decimal',
        'Longitude': 'lon_decimal'
    }, inplace=True)

    # 2. Feature Engineering
    if 'Average Monthly Salary' in df_main.columns and 'Average Cost of Living' in df_main.columns:
        df_main['Disposable Income'] = df_main['Average Monthly Salary'] - df_main['Average Cost of Living']
    
    if 'Main Spoken Languages' in df_main.columns:
        df_main['English Spoken'] = df_main['Main Spoken Languages'].astype(str).apply(
            lambda x: 'English' in x
        )
    else:
        df_main['English Spoken'] = False

    print("--- Data Ready ---")
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

# -- Main App Layout (Flexbox Container) --
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # -- Header (Fixed Height) --
    html.Div([
        html.Div([
            html.H1("Global City Analytics", style={'margin': '0', 'color': '#2C3E50', 'fontSize': '28px'}),
            html.P("Cost of Living & Quality of Life Explorer", style={'margin': '5px 0 0 0', 'color': '#7F8C8D'})
        ]),
    ], style={
        'padding': '20px 40px', 
        'backgroundColor': 'white', 
        'borderBottom': '1px solid #ddd',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.05)',
        'flex': '0 0 auto', # Prevent header from shrinking
        'height': '60px'    # Fixed visual height
    }),
    
    # -- Main Content (Fills remaining space) --
    html.Div(id='page-content', style={
        'flex': '1', 
        'overflow': 'hidden', # Prevent double scrollbars; individual pages handle their own scrolling
        'backgroundColor': '#F4F7F6', 
        'position': 'relative',
        'display': 'flex',
        'flexDirection': 'column'
    })
], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'overflow': 'hidden'})


# --- Main Map View Layout ---
layout_map_view = html.Div([
    
    # Top Row: KPI Cards
    html.Div([
        html.Div([
            html.H4("Total Cities", style={'margin': '0', 'color': '#95A5A6', 'fontSize': '14px'}),
            html.H2(id='kpi-total-cities', children=f"{len(df_main)}", style={'margin': '5px 0', 'color': '#2C3E50'})
        ], className='kpi-card'),
        
        html.Div([
            html.H4("Avg. Selected Salary", style={'margin': '0', 'color': '#95A5A6', 'fontSize': '14px'}),
            html.H2(id='kpi-avg-salary', children=f"${df_main['Average Monthly Salary'].mean():,.0f}" if not df_main.empty else "$0", style={'margin': '5px 0', 'color': '#27AE60'})
        ], className='kpi-card'),
        
        html.Div([
            html.H4("English Speaking %", style={'margin': '0', 'color': '#95A5A6', 'fontSize': '14px'}),
            html.H2(id='kpi-english-pct', children=f"{(df_main['English Spoken'].sum() / len(df_main) * 100):.1f}%" if not df_main.empty else "0%", style={'margin': '5px 0', 'color': '#2980B9'})
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
                min=df_main['Population'].min() if not df_main.empty else 0,
                max=df_main['Population'].max() if not df_main.empty else 1000000,
                value=[df_main['Population'].min() if not df_main.empty else 0, df_main['Population'].max() if not df_main.empty else 1000000],
                marks={int(x): {'label': f'{int(x/1000)}k', 'style': {'display': 'none'}} for x in np.linspace(df_main['Population'].min() if not df_main.empty else 0, df_main['Population'].max() if not df_main.empty else 1000000, 5)},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
            
            html.Label("Country", style={'fontWeight': 'bold', 'display': 'block', 'marginTop': '30px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': c, 'value': c} for c in sorted(df_main['Country_Name'].unique())] if not df_main.empty else [],
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
            html.H3("Cities", style={'color': '#34495E', 'marginBottom': '15px', 'borderBottom': '2px solid #27AE60', 'paddingBottom': '10px'}),
            html.Div(id='city-list-container', style={'height': '540px', 'overflowY': 'auto'})
        ], style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'})
        
    ], style={'display': 'grid', 'gridTemplateColumns': '1fr 3fr 1fr', 'gap': '25px'})

], style={'padding': '30px', 'overflowY': 'auto', 'height': '100%'}) # Map view scrolls internally


# --- City Detail View Layout ---
def create_city_layout(city_name):
    city_data = df_main[df_main['City_Name'] == city_name].iloc[0]
    
    return html.Div([
        # 1. Compact Header (Same as before)
        html.Div([
            html.Div([
                dcc.Link("← Back", href="/", style={'textDecoration': 'none', 'color': '#7F8C8D', 'fontWeight': 'bold', 'fontSize': '14px', 'marginRight': '15px'}),
                html.Span(f"{city_name}, {city_data['Country_Name']}", style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#2C3E50'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.Label("Compare:", style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '14px'}),
                dcc.Dropdown(
                    id='compare-city-dropdown',
                    options=[{'label': c, 'value': c} for c in sorted(df_main['City_Name'].unique()) if c != city_name],
                    placeholder="Select city...",
                    style={'width': '220px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'height': '50px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '10px', 'flexShrink': 0}),
        
        # 2. Main Dashboard Content
        html.Div([
            # -- Left Sidebar: Stats (Now Empty Container) --
            html.Div(id='sidebar-stats-container', style={'width': '250px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'marginRight': '15px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.05)', 'overflowY': 'auto'}),
            
            # -- Right Area: Charts --
            html.Div([
                html.Div([dcc.Graph(id='detail-gauge', style={'height': '100%'})], style={'flex': '4', 'backgroundColor': 'white', 'borderRadius': '10px', 'marginBottom': '10px', 'padding': '5px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.05)'}),
                html.Div([dcc.Graph(id='detail-bar', style={'height': '100%'})], style={'flex': '6', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '5px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.05)'})
            ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'})
            
        ], style={'flex': '1', 'display': 'flex', 'overflow': 'hidden', 'minHeight': '0'})

    ], style={'height': '100%', 'padding': '20px', 'boxSizing': 'border-box', 'display': 'flex', 'flexDirection': 'column'})

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
    [Output('main-map', 'figure'), 
     Output('city-list-container', 'children'),
     Output('kpi-total-cities', 'children'),
     Output('kpi-avg-salary', 'children'),
     Output('kpi-english-pct', 'children')],
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
    
    # Calculate Dynamic KPIs
    total_cities = len(dff)
    if total_cities > 0:
        avg_salary = dff['Average Monthly Salary'].mean()
        english_pct = (dff['English Spoken'].sum() / total_cities) * 100
        
        kpi_cities = f"{total_cities}"
        kpi_salary = f"${avg_salary:,.0f}"
        kpi_english = f"{english_pct:.1f}%"
    else:
        kpi_cities = "0"
        kpi_salary = "$0"
        kpi_english = "0%"

    # Logic for colors
    if color_metric == 'Disposable Income':
        color_scale = 'Viridis' 
    elif 'Cost' in color_metric or 'Rent' in color_metric:
        color_scale = 'Reds' 
    elif 'Heat' in color_metric:
        color_scale = 'Magma'
    else:
        color_scale = 'Blues'

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
        custom_data=['City_Name'], 
        color_continuous_scale=color_scale,
        size_max=25,
        zoom=2,
        mapbox_style='carto-positron'
    )
    fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
    
    # List Items
    list_items = []
    dff_sorted = dff.sort_values(by=color_metric, ascending=False)
    
    for _, row in dff_sorted.iterrows():
        item = html.Div([
            dcc.Link(html.H4(row['City_Name'], style={'margin': '0', 'color': '#2980B9'}), href=f"/city/{row['City_Name']}", style={'textDecoration': 'none'}),
            html.Div([
                html.Span(f"{row['Country_Name']}", style={'fontSize': '12px', 'color': '#95A5A6'}),
                html.Span(f"{row[color_metric]:,.0f}", style={'float': 'right', 'fontWeight': 'bold', 'color': '#2C3E50'})
            ])
        ], style={'padding': '15px 0', 'borderBottom': '1px solid #eee'})
        list_items.append(item)
        
    return fig, list_items, kpi_cities, kpi_salary, kpi_english

# -- Detail View Updater --
@app.callback(
    [Output('detail-gauge', 'figure'), 
     Output('detail-bar', 'figure'),
     Output('sidebar-stats-container', 'children')],
    [Input('url', 'pathname'), 
     Input('compare-city-dropdown', 'value')]
)
def update_details(pathname, compare_city):
    if not pathname or not pathname.startswith('/city/'):
        return go.Figure(), go.Figure(), html.Div()
    
    base_city = unquote(pathname.split('/')[-1])
    base_data = df_main[df_main['City_Name'] == base_city].iloc[0]
    
    # --- 1. Sidebar Logic (Kept as requested) ---
    comp_data = None
    if compare_city:
        comp_data = df_main[df_main['City_Name'] == compare_city].iloc[0]

    def create_stat_row(label, base_val, comp_val, base_color, fmt="${:,.0f}"):
        val_text = fmt.format(base_val) if isinstance(base_val, (int, float)) else str(base_val)
        
        content = [
             html.P(label, style={'fontSize': '10px', 'color': '#95A5A6', 'margin': '0', 'textTransform': 'uppercase', 'fontWeight': 'bold'}),
             html.Div(val_text, style={'color': base_color, 'fontSize': '16px', 'fontWeight': 'bold'})
        ]
        
        if comp_val is not None:
            comp_text = fmt.format(comp_val) if isinstance(comp_val, (int, float)) else str(comp_val)
            content.append(
                html.Div(f"vs {comp_text}", style={'color': '#BDC3C7', 'fontSize': '12px', 'marginTop': '2px'})
            )
        return html.Div(content, style={'backgroundColor': '#F8F9F9', 'borderRadius': '6px', 'padding': '8px', 'textAlign': 'center'})

    sidebar_html = html.Div([
        html.H5("Economics", style={'margin': '0 0 10px 0', 'color': '#34495E', 'borderBottom': '1px solid #ddd'}),
        html.Div([
            create_stat_row("Salary", base_data['Average Monthly Salary'], comp_data['Average Monthly Salary'] if comp_data is not None else None, '#27AE60'),
            create_stat_row("Costs", base_data['Average Cost of Living'], comp_data['Average Cost of Living'] if comp_data is not None else None, '#C0392B'),
            create_stat_row("Disposable", base_data['Disposable Income'], comp_data['Disposable Income'] if comp_data is not None else None, '#2980B9'),
            create_stat_row("Unemp.", base_data['Unemployment Rate'], comp_data['Unemployment Rate'] if comp_data is not None else None, '#E67E22', fmt="{:.1f}%"),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px', 'marginBottom': '15px'}),
        
        html.H5("Liveability", style={'margin': '0 0 10px 0', 'color': '#34495E', 'borderBottom': '1px solid #ddd'}),
        html.Div([
            create_stat_row("Rent", base_data['Avgerage Rent Price'], comp_data['Avgerage Rent Price'] if comp_data is not None else None, '#E74C3C'),
            create_stat_row("Heat Days", base_data['Days of very strong heat stress'], comp_data['Days of very strong heat stress'] if comp_data is not None else None, '#D35400', fmt="{:.0f}"),
            create_stat_row("Density", base_data['Population Density'], comp_data['Population Density'] if comp_data is not None else None, '#7F8C8D'),
            create_stat_row("Youth Dep.", base_data['Youth Dependency Ratio'], comp_data['Youth Dependency Ratio'] if comp_data is not None else None, '#2C3E50', fmt="{:.1f}%"),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px'}),
    ])


    # --- 2. Chart Logic (Restored Original Title Text) ---
    current_val = base_data['Average Cost of Living']
    delta = None
    gauge_title = f"Cost of Living: {base_city}"
    
    if comp_data is not None:
        ref_val = comp_data['Average Cost of Living']
        diff = current_val - ref_val
        
        # --- RESTORED LOGIC HERE ---
        if diff > 0:
            gauge_title = f"{base_city} is ${diff:,.0f} More Expensive than {compare_city}"
        else:
            gauge_title = f"{base_city} is ${abs(diff):,.0f} Cheaper than {compare_city}"
        # ---------------------------

        delta = {'reference': ref_val, 'relative': False, 'valueformat': '$,.0f'}

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta" if delta else "gauge+number",
        value = current_val,
        delta = delta,
        title = {'text': gauge_title, 'font': {'size': 14}}, # Size 14 ensures long text fits
        number = {'prefix': "$", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [df_main['Average Cost of Living'].min(), df_main['Average Cost of Living'].max()]},
            'bar': {'color': "#2C3E50"}, 
            'steps': [
                {'range': [0, 1000], 'color': "#D5F5E3"},
                {'range': [1000, 2000], 'color': "#FCF3CF"}
            ]
        }
    ))
    fig_gauge.update_layout(margin={'l': 30, 'r': 30, 't': 50, 'b': 10})
    
    fig_bar = make_subplots(rows=1, cols=4, column_widths=[0.35, 0.2, 0.2, 0.25], subplot_titles=("Financials", "Social", "Density", "Heat"))

    def add_city_traces(city_name, city_data, color, show_legend):
        fig_bar.add_trace(go.Bar(name=city_name, x=['Salary', 'Cost', 'Rent'], y=[city_data['Average Monthly Salary'], city_data['Average Cost of Living'], city_data['Avgerage Rent Price']], marker_color=color, showlegend=show_legend), row=1, col=1)
        fig_bar.add_trace(go.Bar(name=city_name, x=['Unemp.', 'Youth'], y=[city_data['Unemployment Rate'], city_data['Youth Dependency Ratio']], marker_color=color, showlegend=False), row=1, col=2)
        fig_bar.add_trace(go.Bar(name=city_name, x=['Density'], y=[city_data['Population Density']], marker_color=color, showlegend=False), row=1, col=3)
        fig_bar.add_trace(go.Bar(name=city_name, x=['Heat Days'], y=[city_data['Days of very strong heat stress']], marker_color=color, showlegend=False), row=1, col=4)

    add_city_traces(base_city, base_data, '#2980B9', True)

    if comp_data is not None:
        add_city_traces(compare_city, comp_data, '#95A5A6', True)
        title_text = f"{base_city} vs {compare_city}"
    else:
        global_avg = df_main.mean(numeric_only=True)
        add_city_traces("Global Avg", global_avg, '#BDC3C7', True)
        title_text = f"{base_city} vs Global Avg"

    fig_bar.update_layout(
        title={'text': title_text, 'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
        barmode='group',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
    )

    return fig_gauge, fig_bar, sidebar_html

# -- Handle Map Clicks to Navigate --
@app.callback(
    Output('url', 'pathname'),
    [Input('main-map', 'clickData')],
    [State('url', 'pathname')]
)
def go_to_city_view(clickData, current_path):
    if clickData is None:
        return dash.no_update
    
    try:
        # Custom Data set in px.scatter_mapbox
        city_name = clickData['points'][0]['customdata'][0]
        new_path = f"/city/{city_name}"
        if new_path != current_path:
            return new_path
    except Exception as e:
        print(f"Error handling map click: {e}")
        
    return dash.no_update

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