import dash
from dash import html, dcc, Input, Output, State, register_page, callback_context
import dash_bootstrap_components as dbc

# Register the page
register_page(__name__, path="/analytical", name="Analytical Methods")

# Load images function
def load_images(year_range):
    base_url = "https://storage.googleapis.com/databucket_seniorproj/Natural_Gas_Clustering/Natural_Gas_"
    images = []
    for year in range(year_range[0], year_range[1] + 1):
        image_url = f"{base_url}{year}.png"
        images.append(image_url)
    print("Loaded images:", images)
    return images

# Layout for the page
layout = html.Div([
    # Header
    html.Div([
        html.H1("Analytical Methods", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("The data-driven techniques behind our infrastructure insights.",
               style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
    ], style={
        'background': 'linear-gradient(to right, #0f172a, #1e293b)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    # Energy Infrastructure Section
    html.Div([
        html.H2("Energy Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center'}),
        dcc.RangeSlider(
            id='year-range-slider',
            min=2000,
            max=2021,
            step=1,
            marks={year: str(year) for year in range(2000, 2022, 2)},
            value=[2000, 2005],
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id='slideshow-container', style={'textAlign': 'center', 'margin': '30px 0'}),
        html.Div([
            dbc.Button("Previous", id="prev-button", color="primary", className="me-2", n_clicks=0),
            dbc.Button("Next", id="next-button", color="primary", n_clicks=0),
        ], style={'textAlign': 'center'}),
        dcc.Store(id='image-index', data=0),
        dcc.Store(id='image-list', data=load_images([2000, 2005]))

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),

    # Transportation Infrastructure Section (Placeholder)
    html.Div([
        html.H2("Transportation Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P("Coming soon...", style={'color': '#cbd5e0', 'textAlign': 'center'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])

# Combined callback for both year range and buttons
@dash.callback(
    Output('image-list', 'data'),
    Output('image-index', 'data'),
    Input('year-range-slider', 'value'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    State('image-index', 'data'),
    State('image-list', 'data')
)
def update_images(year_range, prev_clicks, next_clicks, current_index, image_list):
    ctx = callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Range slider changed: reload images
    if trigger_id == 'year-range-slider':
        new_images = load_images(year_range)
        return new_images, 0

    if not image_list:
        return dash.no_update, 0

    # Button navigation
    if trigger_id == 'next-button':
        current_index = (current_index + 1) % len(image_list)
    elif trigger_id == 'prev-button':
        current_index = (current_index - 1) % len(image_list)

    return dash.no_update, current_index

# Callback to display current image
@dash.callback(
    Output('slideshow-container', 'children'),
    Input('image-index', 'data'),
    State('image-list', 'data')
)
def display_image(index, image_list):
    if not image_list:
        return html.P("No images found for the selected range.", style={'color': 'red'})
    return html.Img(src=image_list[index], style={'width': '60%', 'height': 'auto'})
