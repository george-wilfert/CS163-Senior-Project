from dash import register_page, html

# Register page with path "/objective"
register_page(__name__, path="/objective", name="Projective Objective")

layout = html.Div([
    html.Div([
        html.H1("Projective Objective", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("Our long-term infrastructure vision and strategic direction.",
               style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
    ], style={
        'background': 'linear-gradient(to right, #0f172a, #1e293b)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H2("Strategic Goals", style={'color': '#34d399', 'textAlign': 'center'}),
        html.Ul([
            html.Li("Expand sustainable energy infrastructure by 2035"),
            html.Li("Modernize national transportation systems"),
            html.Li("Improve access and equity across regions"),
            html.Li("Leverage data to guide smart development"),
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '800px',
            'margin': '40px auto'
        })
    ], style={'padding': '40px 20px', 'backgroundColor': '#111827'}),

    html.Div([
        html.H2("How We Get There", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P(
            "We aim to collaborate with agencies, leverage real-time data analytics, and drive innovation in infrastructure planning and deployment. Our strategy is forward-thinking and inclusive, prioritizing environmental stewardship and technological resilience.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])
