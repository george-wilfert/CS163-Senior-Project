from dash import html, dcc, page_container
import app  # Import the Dash app instance from app.py

# Define the layout of the app
app.app.layout = html.Div([
    # Navigation Bar
    html.Nav([
        html.Div("Infastructure Senior Project", style={'fontWeight': 'bold', 'fontSize': '1.5rem', 'color': '#38bdf8'}),
        html.Div([
            dcc.Link("Home", href="/", style={'marginRight': '20px', 'color': 'white', 'textDecoration': 'none'}),
            dcc.Link("Project Objective", href="/objective", style={'marginRight': '20px', 'color': 'white', 'textDecoration': 'none'}),
            dcc.Link("Analytical Methods", href="/analytical", style={'marginRight': '20px', 'color': 'white', 'textDecoration': 'none'}),
            dcc.Link("Major Findings", href="/findings", style={'color': 'white', 'textDecoration': 'none'}),
        ])
    ], style={
        'backgroundColor': '#1e293b',
        'padding': '20px 40px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'color': 'white',
        'position': 'sticky',
        'top': '0',
        'zIndex': '1000'
    }),

    # Page content
    html.Div(page_container, style={'padding': '40px', 'backgroundColor': '#111827'})
])

if __name__ == '__main__':
    app.app.run_server(debug=True)
