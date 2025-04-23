from dash import register_page, html

register_page(__name__, path="/findings", name="Major Findings")

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

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_price_forecast_CI.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 4: SARIMA Forecast with 95% Confidence Interval", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("The SARIMA model accurately captures future increases with tight forecast bounds. This supports its reliability for infrastructure cost projections.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_forecast_residuals_SARIMA.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 5: Forecast Residuals – Actual vs. Predicted NHCCI", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Residuals fluctuate around zero, indicating low bias. The pattern also lacks severe autocorrelation, confirming a good model fit.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_price_forecast.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 6: Train, Test, and Forecasted NHCCI", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("The visual split of training/testing shows SARIMA adapting to changing trends, particularly the steep climb from 2020–2024.", style={'textAlign': 'center', 'color': '#94a3b8'})
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

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_high_low_investment_lineplot.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 3: NHCCI Over Time by Investment Cluster", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Time series visualization highlights the duration and magnitude of investment phases.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_normalized_cluster_economic.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 4: Normalized Cluster Center Comparison Across Economic Indicators", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Normalization highlights key macroeconomic differences between investment phases (e.g., employment, inflation, GDP).", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_top10_component_investment_clusters.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 5: Top 10 Components Driving Cluster Differences", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Asphalt and Bridge were the most influential components distinguishing between High and Low Investment clusters.", style={'textAlign': 'center', 'color': '#94a3b8'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])
