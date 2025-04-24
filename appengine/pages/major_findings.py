from dash import register_page, html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

register_page(__name__, path="/findings", name="Major Findings")

# Load SARIMA interactive data
plot_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_SARIMA_plot_df.csv")
residuals_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_SARIMA_residuals.csv")
train_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_SARIMA_train.csv")
test_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_SARIMA_test.csv")
forecast_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_SARIMA_forecast_plot.csv")
cluster_line_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_cluster_input.csv")
normalized_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_norm_cluster_centers.csv", index_col=0)
lasso_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_LASSO_plot_df.csv")

# Convert datetime fields
# plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])
# residuals_df["datetime"] = pd.to_datetime(residuals_df["datetime"])
# train_df["datetime"] = pd.to_datetime(train_df["datetime"])
# test_df["datetime"] = pd.to_datetime(test_df["datetime"])

# Create SARIMA forecast vs actual
forecast_upper = forecast_df["Forecast"] + 1.96 * residuals_df["residuals"].std()
forecast_lower = forecast_df["Forecast"] - 1.96 * residuals_df["residuals"].std()

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=train_df["datetime"],
    y=train_df["NHCCI-Seasonally-Adjusted"],
    mode="lines",
    name="Train",
    line=dict(color="black")
))

fig_forecast.add_trace(go.Scatter(
    x=test_df["datetime"],
    y=test_df["NHCCI-Seasonally-Adjusted"],
    mode="lines",
    name="Actual (Test)",
    line=dict(color="green")
))

fig_forecast.add_trace(go.Scatter(
    x=test_df["datetime"],
    y=forecast_df["Forecast"],
    mode="lines",
    name="Forecast",
    line=dict(color="orange")
))

# Add confidence interval (shaded area)
fig_forecast.add_trace(go.Scatter(
    x=test_df["datetime"],
    y=forecast_upper,
    line=dict(color='orange', width=0),
    showlegend=False,
    name='Upper Bound'
))
fig_forecast.add_trace(go.Scatter(
    x=test_df["datetime"],
    y=forecast_lower,
    fill='tonexty',
    fillcolor='rgba(255,165,0,0.3)',
    line=dict(color='orange', width=0),
    name='95% Confidence Interval'
))

fig_forecast.update_layout(
    title="SARIMA Forecast vs Actual NHCCI Values",
    title_x=0.5,
    xaxis_title="Date",
    yaxis_title="NHCCI Seasonally Adjusted Value",
    legend_title_text='',
    xaxis=dict(tickangle=45)
)

# Create SARIMA residuals plot
fig_residuals = px.line(
    residuals_df,
    x="datetime",
    y="residuals",
    title="SARIMA Forecast Residuals Over Time"
)
fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
fig_residuals.update_layout(title_x=0.5)

# Major Findings #2, Plot 3 - NHCCI over Time by Investment Cluster
fig_kmeans_line = px.line(
    cluster_line_df,
    x="datetime",
    y="NHCCI-Seasonally-Adjusted",
    color="Spending Cluster Label",
    title="NHCCI Over Time by Investment Cluster",
    labels={"datetime": "Year", "NHCCI-Seasonally-Adjusted": "NHCCI (Seasonally Adjusted)"},
)
fig_kmeans_line.update_layout(title_x=0.5, xaxis_tickangle=45)

# Major Findings #2, Plot 4 - Normalized Cluster Centers
norm_df = normalized_df.reset_index().rename(columns={"index": "Feature"})
norm_long = norm_df.melt(id_vars="Feature", var_name="Cluster", value_name="Value")

# Plot
fig_cluster_centers = px.line(
    norm_long,
    x="Feature",
    y="Value",
    color="Cluster",
    markers=True,
    title="Normalized Cluster Center Comparison Across Economic Indicators"
)
fig_cluster_centers.update_layout(title_x=0.5)

# Major Findings #3, Predicted vs Actual NHCCI by Lasso model
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
        html.H1("Major Finding #1: SARIMA Modeling on NHCCI", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("Using the SARIMA model on the NHCCI dataset, we observed a sharp increase in highway construction costs post-2020, indicating sustained demand and inflationary effects in infrastructure investment. The model confirmed strong seasonality, reliable parameter selection, and an overall good fit with low RMSE and unbiased residuals. These findings validate our hypothesis that temporal trends can accurately model highway cost behavior, providing meaningful guidance for future financial planning and policy decisions.",
               style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.P("Model Performance Highlights:", style={'fontSize': '1.2rem', 'color': '#fbbf24', 'marginTop': '20px'}),
        html.Ul([
            html.Li("The Augmented Dickey-Fuller test revealed the original series was non-stationary (p = 0.9571), but first-order differencing successfully stabilized it (p = 0.0002)."),
            html.Li("The model’s parameters were selected using ACF and PACF plots, which showed significant lag structure at 1 and 2."),
            html.Li("SARIMA forecasting showed good alignment with actual values, with RMSE = 0.2098, confirming a strong model fit."),
            html.Li("The forecast captures seasonal increases and was bounded within reasonable 95% confidence intervals."),
            html.Li("Residuals show low bias and relatively small variance, supporting that the model generalized well."),
            html.Li("Exogenous economic indicators such as GDP, unemployment rate (UNRATE), producer price index (PPIACO), and construction employment were integrated to enhance predictive accuracy by capturing broader macroeconomic influences.")
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '900px',
            'margin': '20px auto'
        }),
    ], style={
        'background': 'linear-gradient(to right, #1e3a8a, #0f172a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H3("Key Visualizations", style={'color': '#fbbf24', 'textAlign': 'center'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_simple_cost_over_time.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 1: NHCCI Seasonally Adjusted Value Over Time", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This plot reveals a noticeable acceleration in highway construction costs post-2020, suggesting inflationary pressures or increased infrastructure investments during the recovery phase after COVID-19.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_stationary_differenced.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 2: Differenced NHCCI Series (Stationarity Confirmed)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Differencing the series confirmed statistical stationarity, allowing reliable SARIMA modeling by neutralizing trend and variance drift.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_ACF_PACF.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 3: ACF and PACF Plots to Guide Parameter Selection", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("These plots were used to identify suitable SARIMA parameters, revealing a strong autocorrelation at lag 1 and partial autocorrelation at lag 2.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        dcc.Graph(figure=fig_forecast),
        html.P("Figure 4: Forecast vs Actual NHCCI with 95% Confidence Interval (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This visualization matches the original layout using train, test, and forecasted data, including shaded bounds for confidence.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        dcc.Graph(figure=fig_residuals),
        html.P("Figure 5: Forecast Residuals – Actual vs. Predicted NHCCI (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Residuals fluctuate around zero, indicating low bias. The pattern also lacks severe autocorrelation, confirming a good model fit.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_extended_forecast.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 6: Forecasted NHCCI Using Recent Trends", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Show a steady increse from 2024-2027, expecting to increse to a value > 3.5 in 2027", style={'textAlign': 'center', 'color': '#94a3b8'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),

    html.Div([
        html.H1("Major Finding #2: K-Means Investment Cluster Analysis", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("Using K-Means clustering (k=2) on NHCCI component percentage changes and macroeconomic indicators, we uncovered two investment patterns: 'Low Investment' and 'High Investment' clusters. These clusters captured periods of significantly different spending levels, as reflected in NHCCI and associated cost components. Findings support our hypothesis that economic signals and component volatility can stratify time periods into investment typologies.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),

        html.P("Cluster Insights:", style={'fontSize': '1.2rem', 'color': '#fbbf24', 'marginTop': '20px'}),
        html.Ul([
            html.Li("Low Investment periods showed lower NHCCI values and were associated with higher unemployment rates and lower PPI."),
            html.Li("High Investment periods were tied to stronger macroeconomic conditions, such as higher construction employment and GDP."),
            html.Li("The clustering highlighted component categories like Asphalt and Bridge as major contributors to cost differentiation between clusters."),
            html.Li("Normalizing cluster centers showed clear divergence across key economic drivers including UNRATE, PPIACO, and GDP."),
            html.Li("This clustering methodology can be used for policy assessment and forecasting infrastructure investment cycles.")
        ], style={
            'color': '#e5e7eb',
            'fontSize': '1.1rem',
            'lineHeight': '1.8',
            'maxWidth': '900px',
            'margin': '20px auto'
        })
    ], style={
        'background': 'linear-gradient(to right, #0f172a, #1e3a8a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H3("Key Visualizations", style={'color': '#fbbf24', 'textAlign': 'center'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_cluster_per_quarter_assignment.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 1: K-Means Cluster Assignment Per Quarter", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Each quarter was assigned to a spending cluster. High and Low Investment periods alternate based on macroeconomic signals and NHCCI changes.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_distribution_by_cluster_investment.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 2: NHCCI Distribution by Investment Cluster", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("High Investment periods show higher median NHCCI values compared to Low Investment periods.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        dcc.Graph(figure=fig_kmeans_line),
        html.P("Figure 3: NHCCI Over Time by Investment Cluster (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Time series visualization highlights the duration and magnitude of investment phases.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        dcc.Graph(figure=fig_cluster_centers),
        html.P("Figure 4: Normalized Cluster Center Comparison Across Economic Indicators (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Normalization highlights key macroeconomic differences between investment phases (e.g., employment, inflation, GDP).", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_top10_component_investment_clusters.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 5: Top 10 Components Driving Cluster Differences", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Asphalt and Bridge were the most influential components distinguishing between High and Low Investment clusters.", style={'textAlign': 'center', 'color': '#94a3b8'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
        html.H1("Major Finding #3: LassoCV Macro Regression", style={
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
        
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_lassocv_coefficients.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 1: NHCCI Distribution by Investment Cluster", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("The bar chart shows the coefficients selected by the LassoCV model, highlighting the most influential macroeconomic predictors of NHCCI. PPIACO (Producer Price Index) and TTLCONS (Total Construction Spending) emerged as the only retained features, indicating their strong predictive relationship with construction costs.", style={'textAlign': 'center', 'color': '#94a3b8'}),

                
        dcc.Graph(figure=fig_lasso),
        html.P("Figure 2: Actual vs Predicted NHCCI using LassoCV (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Clear alignment between predicted and actual NHCCI confirms model’s generalization and explanatory power.", style={'textAlign': 'center', 'color': '#94a3b8'}),
        
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_lassoCV_heatmap_corr.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 3: NHCCI Distribution by Investment Cluster", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("The heatmap displays the correlation between the selected predictors and NHCCI. Both PPIACO_lag1 and TTLCONS_lag1 exhibit high positive correlations (0.92 and 0.90 respectively) with NHCCI, supporting their inclusion in the model and reinforcing the strength of their linear association with highway construction costs.", style={'textAlign': 'center', 'color': '#94a3b8'})

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])
