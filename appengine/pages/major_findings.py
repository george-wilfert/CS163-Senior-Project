from dash import register_page, html, dcc, Input, Output
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
ridge_cap_top_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_ridge_top_coefs.csv")
econ_time_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_economic_indicators_time_series.csv")

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

# Helper: function to split segments by cluster
def split_segments(df, cluster_label):
    segments = []
    temp_segment = []
    
    for i in range(len(df)):
        if df.iloc[i]["Spending Cluster Label"] == cluster_label:
            temp_segment.append(df.iloc[i])
        else:
            if temp_segment:
                segments.append(pd.DataFrame(temp_segment))
                temp_segment = []
    if temp_segment:
        segments.append(pd.DataFrame(temp_segment))
    return segments

# Create figure
fig_segmented = go.Figure()

# Plot High Investment segments
high_segments = split_segments(cluster_line_df, "High Investment")
for seg in high_segments:
    fig_segmented.add_trace(go.Scatter(
        x=seg["datetime"],
        y=seg["NHCCI-Seasonally-Adjusted"],
        mode="lines+markers",
        name="High Investment",
        line=dict(color="blue"),
        marker=dict(size=4),
        showlegend=False  # Avoid repeating legend
    ))

# Plot Low Investment segments
low_segments = split_segments(cluster_line_df, "Low Investment")
for seg in low_segments:
    fig_segmented.add_trace(go.Scatter(
        x=seg["datetime"],
        y=seg["NHCCI-Seasonally-Adjusted"],
        mode="lines+markers",
        name="Low Investment",
        line=dict(color="red"),
        marker=dict(size=4),
        showlegend=False
    ))

# Add manual legends (since we suppressed repeating them)
fig_segmented.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(color="blue"),
    name="High Investment"
))
fig_segmented.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(color="red"),
    name="Low Investment"
))

fig_segmented.update_layout(
    title="NHCCI Over Time Segmented by Investment Cluster",
    title_x=0.5,
    xaxis_title="Year",
    yaxis_title="NHCCI (Seasonally Adjusted)",
    legend_title_text="Spending Cluster Label",
    xaxis_tickangle=45
)

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

# Major Findings #5, High-Impact Interaction Coeffecients for Captial Spending
fig_ridge_capital_top = px.bar(
    ridge_cap_top_df.sort_values("Coefficient", ascending=False),
    x="Coefficient",
    y="Feature",
    orientation="h",
    title="High-Impact Interaction Coefficients (Capital Spending)"
)
fig_ridge_capital_top.update_layout(title_x=0.5)

# Create figure based on dropdown selection
def create_highway_econ_figure(indicator="TTLCONS"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=econ_time_df["Year"],
        y=econ_time_df["Chained Spending"],
        mode="lines+markers",
        name="Highway Spending",
        yaxis="y1"
    ))

    fig.add_trace(go.Scatter(
        x=econ_time_df["Year"],
        y=econ_time_df[indicator],
        mode="lines+markers",
        name=indicator,
        yaxis="y2"
    ))

    fig.update_layout(
        title="Highway Spending vs Economic Indicator",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Highway Spending (Chained)", side="left"),
        yaxis2=dict(title=indicator, overlaying="y", side="right"),
        legend=dict(x=0.5, y=1.1, orientation="h", xanchor="center"),
        title_x=0.5,
        margin=dict(l=40, r=40, t=80, b=40),
        template="plotly_dark"
    )

    return fig

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

        dcc.Graph(figure=fig_segmented),
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

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
        html.H1("Major Finding #4: Regression on GMM Results", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        
        html.P("", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.P("Our first regression analysis is a focus on predicting expenditures using just the cluster probabilities that were mentioned in previous sections. This model gave us a surprising result of .614 r squared value. This results in a justification for my original goal which was to prove that natural gas usage can be a signal for infrastructure in general. The graph below shows the feature importance for the regression anayalsis. Further analysis will use this clam to build on top to answer our hypothesis.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/EDA_Graphs/Feature%20Importance%20for%20Predicting%20Energy%20Expenditures.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.P("This regression analysis tries to answer our first hypothesis which is if there is a correlation between economic metrics and infrastructure. Using our previous conclusion that we can use these natural resources and local energy infrastructure to abstract a larger metric for overall infrastructure development we tested to see the impact of on the GDP as a whole. Assuming GDP is very dependent on other factors we were not confident in the success of this model but wanted to see some sort of correlation. This we did with a r squared of .231 which shows some sort of help but not alot. We will expand this analysis in the next section.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/EDA_Graphs/Feature%20Importance%20for%20GDP%20Prediction%201.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.P("Given our subpar results with just the GMM cluster we added a couple more features to hopefully include different factors of the economy such as population and productivity rate. This made an instant improvement. We got an R squared of .67 which was very good for predicting economic metrics. Our conclusion is these local energy infrastructure metrics and natural resource use is a good indicator of total economy metrics which proves our first hypothesis.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/EDA_Graphs/Feature%20Importance%20for%20GDP%20Prediction%202.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Lastly for the energy aspect of the project we wanted to help give evidence to our second hypothesis that our signals previously mentioned helped to predict and impact urbanization. This is a statewide analysis so there is higher amounts of variation compared to looking at local structures but still good for concept. We did a regression on urbanization rating found and averaged from governmental sources of states based on the same features as our second GDP regression. We were also successful in getting correlations between urbanization and our indicators so it helps answer our second hypothesis that urbanization is correlated with infrastructure.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/EDA_Graphs/Feature%20Importance%20for%20Predicting%20Urbanization%20Metric.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),



    ], style={
        'background': 'linear-gradient(to right, #0f172a, #1e3a8a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),
    
    html.Div([
        html.H1("Major Finding #5: Ridge Regression on TPFS Capital Spending", style={
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

        dcc.Graph(figure=fig_ridge_capital_top),
        html.P("Figure 1: High-Impact Capital Coefficients (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Interactive bar plot revealing the most influential mode-government interactions driving capital spending, such as federal investments in Water infrastructure and Highway projects.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_cap_vs_noncap_spending.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 2: Average Log Spending by Capital vs Non-Capital", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Comparative bar chart illustrating that capital projects account for the majority of infrastructure investments across key transportation modes.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_ridge_capital_lineplot.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 3: Actual vs Predicted Log Spending (Capital)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Line plot comparing predicted and actual capital expenditures. Close alignment along the 45° reference line validates model accuracy.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_ridge_residual.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 4: Residual Plot for Capital Model", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Residuals scattered symmetrically around zero with little bias/heteroskedasticity, supporting strong generalization of our Ridge model.", style={'textAlign': 'center', 'color': '#94a3b8'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
        html.H1("Major Finding #6: Infrastructure Spending Time Series Trends", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("We performed time series decomposition and cross-correlation analysis on federal, state/local, and total infrastructure spending. This revealed key long-term trends, and showed that spending leads major economic indicators like GDP and PPI by approximately 2 years.", style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
        html.P("Findings Summary:", style={'fontSize': '1.2rem', 'color': '#fbbf24', 'marginTop': '20px'}),
        html.Ul([
            html.Li("Spending series show strong trend components with minimal seasonality."),
            html.Li("Cross-correlation results suggest infrastructure spending peaks precede GDP and PPI growth by ~2 years."),
            html.Li("Highways spending aligns closely with Total Construction Spending (TTLCONS) and Producer Price Index (PPIACO) movements."),
            html.Li("Insights can inform lagged forecasting models for economic growth based on public infrastructure investments.")
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
    
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_decomp_fed.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 1: Time Series Decomposition - Federal Infrastructure Spending", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Decomposition reveals that federal spending trends upward post-2020 without notable seasonality.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_decomp_statelocal.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 2: Time Series Decomposition - State and Local Infrastructure Spending", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("State and local spending followed a gradual rise until 2020, before slight declines due to COVID-related budgetary constraints.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_decomp_total.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 3: Time Series Decomposition - Total Infrastructure Spending", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Total spending remained stable with noticeable boosts during federal stimulus periods.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_GDP_spending.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 4: Cross-Correlation of Spending Leading GDP", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Spending leads GDP growth, with strongest positive correlation observed at a 2-year lag.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        dcc.Dropdown(
            id="indicator_dropdown",
            options=[
                {"label": "Total Construction Spending (TTLCONS)", "value": "TTLCONS"},
                {"label": "Producer Price Index (PPIACO)", "value": "PPIACO"},
                {"label": "Gross Domestic Product (GDP)", "value": "GDP"}
            ],
            value="TTLCONS",
            style={'width': '60%', 'margin': '20px auto'}
        ),
        dcc.Graph(id="highway_econ_graph"),
        html.P("Figure 5: Highways Spending vs Selected Economic Metric (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("User can select an economic indicator (GDP, PPI, or Total Construction) to visualize its relationship with highway spending over time.", style={'textAlign': 'center', 'color': '#94a3b8'}),

        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_PPI_spending.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),
        html.P("Figure 6: Cross-Correlation of Spending Leading PPI", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Spending peaks lead PPI inflation by about 2 years, emphasizing infrastructure investment’s influence on construction-related inflation trends.", style={'textAlign': 'center', 'color': '#94a3b8'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
])

from dash import callback 

@callback(
    Output("highway_econ_graph", "figure"),
    Input("indicator_dropdown", "value")
)
def update_highway_econ_graph(selected_indicator):
    return create_highway_econ_figure(selected_indicator)