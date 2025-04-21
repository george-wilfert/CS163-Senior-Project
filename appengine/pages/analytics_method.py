from dash import register_page, html

register_page(__name__, path="/analytical", name="Analytical Methods")

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

    # Content
    html.Div([
        html.H2("Core Approaches", style={'color': '#34d399', 'textAlign': 'center'}),
        html.Ul([
            html.Li("Geospatial Analysis using GIS data"),
            html.Li("Machine Learning models to predict usage trends"),
            html.Li("Clustering techniques for region segmentation"),
            html.Li("Time-series forecasting for demand planning"),
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '800px',
            'margin': '40px auto'
        })
    ], style={'padding': '40px 20px', 'backgroundColor': '#111827'}),

    html.Div([
        html.H2("Tools We Use", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P(
            "Our methods combine statistical rigor and modern computational tools such as Python, pandas, scikit-learn, TensorFlow, and PostgreSQL to power our dashboards and predictions.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])
