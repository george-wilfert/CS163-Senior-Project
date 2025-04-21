from dash import register_page, html

register_page(__name__, path="/findings", name="Major Findings")

layout = html.Div([
    html.Div([
        html.H1("Major Findings", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("Key insights and takeaways from our infrastructure analysis.",
               style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
    ], style={
        'background': 'linear-gradient(to right, #1e3a8a, #0f172a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H2("Highlights", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.Ul([
            html.Li("Aging infrastructure in densely populated regions requires urgent upgrades."),
            html.Li("Energy grid capacity is nearing peak load limits in 30% of states."),
            html.Li("Urban transit systems show strong correlations with income mobility."),
            html.Li("Renewable energy sources are underutilized despite geographic potential."),
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '800px',
            'margin': '40px auto'
        }),
    ], style={'padding': '40px 20px', 'backgroundColor': '#111827'}),

    html.Div([
        html.H2("Our Hypthoesises Answered", style={'color': '#34d399', 'textAlign': 'center'}),
        html.P(
            "",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),

    html.Div([
        html.H2("How Our Project Can be Used", style={'color': '#34d399', 'textAlign': 'center'}),
        html.P(
            "",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])