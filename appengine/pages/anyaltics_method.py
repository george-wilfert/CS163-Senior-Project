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
        html.Div([
        html.H2("Enegry Infastrucutre", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P(
            "",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
        
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    html.Div([
        html.H2("Transportation Infastrucuture", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P(
            "",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])
