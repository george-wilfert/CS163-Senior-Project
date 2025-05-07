from dash import register_page, html

# Register this page with its path as the home page (landing page)
register_page(__name__, path='/')

layout = html.Div([
    # Hero Banner Section
    html.Div([
        html.H1("Welcome to Our Infrastructure Analysis!", style={
            'fontSize': '4rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("Your one-stop hub for analytical insights on energy, highway, and transportation infrastructure.",
               style={
                   'fontSize': '1.5rem',
                   'color': '#cbd5e0',
                   'marginBottom': '40px'
               }),
        html.P("Our project explores how government infrastructure spending across energy systems, highways, and broader transportation networks relates to economic and regional development. Using public datasets like TPFS and NHCCI and an API from the Energy Information Administration, we apply clustering, forecasting, and regression techniques to uncover patterns, assess cost drivers, and evaluate funding impacts.",
                style={
                    'fontSize': '1.2rem',
                    'color': '#e5e7eb',
                    'maxWidth': '900px',
                    'margin': '0 auto 40px'
                }),
        html.Img(
                src="https://storage.googleapis.com/databucket_seniorproj/EDA_Graphs/choropleth_map.png",
                style={
                    "width": "70%",          
                    "maxWidth": "600px",     
                    "display": "block",      
                    "margin": "auto",        
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
                }), 
        html.Img(
                src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_lassoCV_predicted_actual.png",
                style={
                    "width": "70%",          
                    "maxWidth": "600px",     
                    "display": "block",      
                    "margin": "auto",        
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
                }), 
    ], style={
        'background': 'linear-gradient(135deg, #1e293b, #0f172a)',
        'padding': '120px 20px',
        'textAlign': 'center'
    }),

    # Features / Highlights Section
    html.Div([
        html.Div([
            html.H2("Energy Infrastructure Insights", style={'color': '#34d399'}),
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
            html.H2("Transportation & Highway Infrastructure Insights", style={'color': '#2563eb'}),
            html.P("TPFS helps us understand how federal, state, and local governments fund and manage transportation systems across all modes.", style={'color': '#cbd5e0'}),
            html.P("NHCCI provides insight into the changing cost of highway construction by analyzing trends in materials, labor, and bidding data.", style={'color': '#cbd5e0'})

        ], style={
            'flex': '1',
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#1f2937',
            'borderRadius': '12px',
            'textAlign': 'center'
        }),
        html.Div([
            html.H2("Expanded Effects of Infrastructure", style={'color': '#fbbf24'}),
            html.P("Major Findings that can used to facilitate infrastructure strategy",
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