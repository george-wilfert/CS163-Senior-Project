from dash import register_page, html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

register_page(__name__, path="/additional-findings", name="Additional Findings")

ridge_cap_top_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_ridge_top_coefs.csv")
top_anomalies_cleaned = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_anomalies_clean.csv")
lasso_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_LASSO_plot_df.csv")

# Additional Finding #1, High-Impact Interaction Coeffecients for Captial Spending
# --------------------------------------------------------------------------- #
fig_ridge_capital_top = px.bar(
    ridge_cap_top_df.sort_values("Coefficient", ascending=False),
    x="Coefficient",
    y="Feature",
    orientation="h",
    title="High-Impact Interaction Coefficients (Capital Spending)"
)
fig_ridge_capital_top.update_layout(title_x=0.5)

# Additional Finding #3, Predicted vs Actual NHCCI by Lasso model
# ---------------------------------------------------------- #
fig_lasso = go.Figure()
fig_lasso.add_trace(go.Scatter(
    x=lasso_df["datetime"],
    y=lasso_df["NHCCI-Seasonally-Adjusted"],
    mode="lines",
    name="Actual",
    line=dict(color="#2563eb", width=2)
))

fig_lasso.add_trace(go.Scatter(
    x=lasso_df["datetime"],
    y=lasso_df["Predicted_NHCCI"],
    mode="lines",
    name="Predicted",
    line=dict(color="orange", width=2, dash="dash")
))

fig_lasso.update_layout(
    title="LassoCV Regression Model - Actual vs. Predicted NHCCI",
    title_x=0.5,
    xaxis_title="Date",
    yaxis_title="NHCCI (Seasonally Adjusted)",
    legend_title_text='',
    xaxis_tickangle=45
)

layout = html.Div([
    html.Div([
        html.H1("Additional Finding #1: Ridge Regression on TPFS Capital Spending", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("We applied Ridge regression with log-transformed TPFS spending data to model both capital and non-capital infrastructure expenses. Incorporating interaction terms between transportation modes and government levels allowed a richer understanding of funding dynamics. The model achieved strong explanatory performance, with R² scores of 0.80 (Capital) and 0.81 (Non-Capital).", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.P("Model Summary:", style={'fontSize': '1.2rem', 'color': '#fbbf24', 'marginTop': '20px'}),
        html.Ul([
            html.Li("Ridge Regression regularized model coefficients to reduce multicollinearity."),
            html.Li("Capital Spending Model R²: 0.802."),
            html.Li("Non-Capital Spending Model R²: 0.809."),
            html.Li("Top positive drivers for capital spending include federal water investment and highways."),
            html.Li("Residual plots confirmed reasonable model behavior with minor bias."),
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '900px',
            'margin': '20px auto'
        })
    ], style={
        'background': 'linear-gradient(to right, #1e3a8a, #0f172a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H3("Key Visualizations", style={'color': '#fbbf24', 'textAlign': 'center'}),

        html.P("Figure 1: High-Impact Capital Coefficients (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Interactive bar plot revealing the most influential mode-government interactions driving capital spending, such as federal investments in Water infrastructure and Highway projects.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        dcc.Graph(figure=fig_ridge_capital_top),

        html.P("Figure 2: Average Log Spending by Capital vs Non-Capital", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Comparative bar chart illustrating that capital projects account for the majority of infrastructure investments across key transportation modes.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_cap_vs_noncap_spending.png", style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 3: Actual vs Predicted Log Spending (Capital)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Line plot comparing predicted and actual capital expenditures. Close alignment along the 45° reference line validates model accuracy.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_ridge_capital_lineplot.png", style={'width': '85%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 4: Residual Plot for Capital Model", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Residuals scattered symmetrically around zero with little bias/heteroskedasticity, supporting strong generalization of our Ridge model.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_ridge_residual.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'})

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
        html.H1("Additional Finding #2: Isolation Forest Uncovers Spending Anomalies in TPFS Dataset", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("We applied the Isolation Forest algorithm to identify outliers in transportation infrastructure spending across federal, state/local, and Amtrak funding sources. The method effectively flagged individual spending records that deviated significantly from historical patterns, controlling for inflation using chained dollars.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),

        html.P("Key Anomaly Insights:", style={'fontSize': '1.2rem', 'color': '#fbbf24', 'marginTop': '20px'}),
        html.Ul([
            html.Li("Highways exhibit the most anomalies consistently over the years, particularly between 2010 and 2022."),
            html.Li("State and Local governments were responsible for the largest number of spending anomalies, especially in capital projects."),
            html.Li("Anomalies often correspond to unusually high capital outlays, as revealed by the top 15 largest outliers."),
            html.Li("Transit and Air modes show growing irregularities after 2015, possibly linked to policy changes or federal funding injections."),
            html.Li("Boxplot analysis confirmed that state/local funding levels have the widest range and largest outliers.")
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '900px',
            'margin': '20px auto'
        }),

        html.H3("Key Visualizations", style={'color': '#fbbf24', 'textAlign': 'center'}),

        html.P("Figure 1: Annual Anomaly Count by Mode of Transportation", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Highways and Transit show the greatest anomaly density, especially in 2020–2022, aligning with federal infrastructure boosts and stimulus grants.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_anomally_count_heatmap.png",
                style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),


        html.P("Figure 2: Anomaly Distribution by Transit Type and Government Level", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Most anomalies are concentrated in State/Local spending on Highways and Transit, followed by federal outliers in Water and Air transportation.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_heatmap_transittype_gov't_anomalies.png",
                style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),


        html.P("Figure 3: Boxplot of Chained Spending by Government Level", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("State and Local entities display a much wider range of spending behavior compared to Federal and Amtrak sources, suggesting greater variability and potential inefficiencies.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_isolation_boxplot.png",
                style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),

        # Interactive Dropdown + Graph
        html.Div([
            html.Label("Select Transit Mode:", style={'color': 'white', 'fontSize': '1.2rem', 'marginTop': '20px'}),
            dcc.Dropdown(
                id="anomaly-mode-dropdown",
                options=[{"label": mode, "value": mode} for mode in sorted(top_anomalies_cleaned["mode"].unique())],
                value="Highways",
                style={'width': '50%', 'margin': '20px auto'}
            ),
            dcc.Graph(id="anomaly-bar-plot"),
            html.P("Figure 4: Top 15 Most Isolated Transportation Spending Anomalies by Mode", style={'textAlign': 'center', 'color': '#cbd5e0'}),
            html.P("This interactive chart allows users to inspect major outliers in inflation-adjusted spending for a selected mode (e.g., Highways, Transit).", style={'textAlign': 'center', 'color': '#94a3b8'})
        ])
    ], style={
        'background': 'linear-gradient(to right, #0f172a, #1e3a8a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),
    
    html.Div([
        html.H1("Additional Finding #3: LassoCV Macro Regression on NHCCI", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("We applied a LassoCV model to determine which macroeconomic indicators most accurately predicted NHCCI trends. The model confirmed that a sparse subset of features, notably lagged PPI and personal consumption, capture key cost movement signals. The model achieved a high Test R² of 0.87.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.P("Model Summary:", style={'fontSize': '1.2rem', 'color': '#fbbf24', 'marginTop': '20px'}),
        html.Ul([
            html.Li("LassoCV performed variable selection among economic indicators."),
            html.Li("Optimal alpha: 0.00163"),
            html.Li("Selected Features: TTLCONS_lag1, PPIACO_lag1"),
            html.Li("Train R²: 0.94, Test R²: 0.87"),
            html.Li("Confirms hypothesis that lagged macro variables explain NHCCI trends.")
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '900px',
            'margin': '20px auto'
        })
    ], style={
        'background': 'linear-gradient(to right, #1e3a8a, #0f172a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H3("Key Visualization", style={'color': '#fbbf24', 'textAlign': 'center'}),
        
        html.P("Figure 1: NHCCI Distribution by Investment Cluster", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("The bar chart shows the coefficients selected by the LassoCV model, highlighting the most influential macroeconomic predictors of NHCCI. PPIACO (Producer Price Index) and TTLCONS (Total Construction Spending) emerged as the only retained features, indicating their strong predictive relationship with construction costs.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_lassocv_coefficients.png", style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 2: Actual vs Predicted NHCCI using LassoCV (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Clear alignment between predicted and actual NHCCI confirms model’s generalization and explanatory power.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        dcc.Graph(figure=fig_lasso),

        html.P("Figure 3: NHCCI Distribution by Investment Cluster", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("The heatmap displays the correlation between the selected predictors and NHCCI. Both PPIACO_lag1 and TTLCONS_lag1 exhibit high positive correlations (0.92 and 0.90 respectively) with NHCCI, supporting their inclusion in the model and reinforcing the strength of their linear association with highway construction costs.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_lassoCV_heatmap_corr.png", style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])