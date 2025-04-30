from dash import register_page, html

# Register this page with its path as the home page (landing page)
register_page(__name__, path='/')

layout = html.Div([
    # Hero Banner Section
    html.Div([
        html.H1("Welcome to Our Infastructure Anyalsis", style={
            'fontSize': '4rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("Your one-stop hub for insights on energy and transportation infrastructure.",
               style={
                   'fontSize': '1.5rem',
                   'color': '#cbd5e0',
                   'marginBottom': '40px'
               }),
        
    ], style={
        'background': 'linear-gradient(135deg, #1e293b, #0f172a)',
        'padding': '150px 20px',
        'textAlign': 'center'
    }),

    # Features / Highlights Section
    html.Div([
        html.Div([
            html.H2("Energy Insights", style={'color': '#34d399'}),
            html.P("Detailed analytics and data visualizations on our national energy infrastructure.",
                   style={'color': '#cbd5e0'})
        ], style={
            'flex': '1',
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#1f2937',
            'borderRadius': '12px',
            'textAlign': 'center'
        }),
        html.Div([
            html.H2("Transport Networks", style={'color': '#2563eb'}),
            html.P("Insights into highways, railroads, and public transit systems across the country.",
                   style={'color': '#cbd5e0'})
        ], style={
            'flex': '1',
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#1f2937',
            'borderRadius': '12px',
            'textAlign': 'center'
        }),
        html.Div([
            html.H2("Expanded Effects of Infastructure", style={'color': '#fbbf24'}),
            html.P("Comprehensive reports and data analysis to guide policy and investment decisions.",
                   style={'color': '#cbd5e0'})
        ], style={
            'flex': '1',
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#1f2937',
            'borderRadius': '12px',
            'textAlign': 'center'
        })
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'flexWrap': 'wrap',
        'padding': '40px 20px',
        'backgroundColor': '#111827'
    }),
])