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
        html.P("A look into our domestic infastructure and its corresponding results",
               style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
    ], style={
        'background': 'linear-gradient(to right, #0f172a, #1e293b)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H2("About our Project", style={'color': '#34d399', 'textAlign': 'center'}),
        html.P(
            "Our goal is to develop a structured methodology to assess infrastructure impacts by analyzing the interconnected economic, environmental, social, and logistical factors that shape large-scale projects. Through data-driven analysis, we aim to provide insights into how infrastructure can be designed and implemented more effectively, optimizing for long-term sustainability, efficiency, and maximum public benefit.", 
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),
        html.Ul([
            html.Li("We split our project into two main parts one looking at the core of our energy infastructure and the other diving deeper into transportation infastruture"),
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '800px',
            'margin': '40px auto'
        })
    ], style={'padding': '40px 20px', 'backgroundColor': '#111827'}),
    html.Div([
        html.H2("Our Hypothesises", style={'color': '#34d399', 'textAlign': 'center'}),
        html.P(
              "Hypothesis 1: Higher infrastructure investment will correlate with increased economic growth, measured by several economic factors like GDP, employment rate, and productivity improvements.",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),
        html.P(),

        html.P(
              "Hypothesis 2:  Infrastructure costs will be more expensive in urban areas due to several factors, such as acquiring land permits, acquisition fees, and of course, labor fees. Rural areas will likely have a higher cost of transportation due to the distance traveled for material shipment.",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
        
    ], style={'padding': '40px 20px', 'backgroundColor': '#111827'}),
    

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