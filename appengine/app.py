from dash import Dash, html, dcc, page_container, Input, Output
import plotly.express as px

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

server = app.server

app.layout = html.Div([
    # Navigation Bar
    html.Nav([
        html.Div("Infrastructure Senior Project", style={
            'fontWeight': 'bold', 'fontSize': '1.5rem', 'color': '#38bdf8'
        }),
        html.Div([
            dcc.Link("Home", href="/", style={'marginRight': '20px', 'color': 'white', 'textDecoration': 'none'}),
            dcc.Link("Project Objective", href="/objective", style={'marginRight': '20px', 'color': 'white', 'textDecoration': 'none'}),
            dcc.Link("Analytical Methods", href="/analytical", style={'marginRight': '20px', 'color': 'white', 'textDecoration': 'none'}),
            dcc.Link("Major Findings", href="/findings", style={'marginRight': '20px', 'color': 'white', 'textDecoration': 'none'}),
            dcc.Link("Additional Findings", href="/additional-findings", style={'color': 'white', 'textDecoration': 'none'}),
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

    # Page Container
    html.Div(page_container, style={'padding': '40px', 'backgroundColor': '#111827'})
])

@app.callback(
    Output("anomaly-bar-plot", "figure"),
    Input("anomaly-mode-dropdown", "value")
)
def update_anomaly_barplot(selected_mode):
    from pages.more_findings import top_anomalies_cleaned
    df = top_anomalies_cleaned.copy()
    
    # Fix capitalization issues
    df["mode"] = df["mode"].str.strip().str.title()

    filtered_df = df[df["mode"] == selected_mode]

    # Optional: Add a fallback if no results
    if filtered_df.empty:
        return px.bar(title="No data found for selected mode.")

    # Sort by Isolation Score and take top 15
    filtered_df = filtered_df.sort_values("chained_value", ascending=False).head(10)

    # Use appropriate column names
    fig = px.bar(
        filtered_df,
        x="desccription",  # or "transit_type" if description not meaningful
        y="chained_value", 
        color="gov_level",
        title=f"Top 10 Anomalies - {selected_mode}",
        labels={"chained_value": "Spending ($, chained)", "desccription": "Description"}
    )

    fig.update_layout(xaxis_tickangle=45, title_x=0.5)
    return fig

@app.callback(
    Output("highway_econ_graph", "figure"),
    Input("indicator_dropdown", "value")
)
def update_highway_econ_graph(selected_indicator):
    from pages.major_findings import create_highway_econ_figure
    return create_highway_econ_figure(selected_indicator)

if __name__ == '__main__':
    app.run(debug=True)
