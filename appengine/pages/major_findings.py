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
econ_time_df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_economic_indicators_time_series.csv")

# Major Finding #1, Plot 4 - Creating SARIMA (forecast vs actual) plot
# ------------------------------------------------------------------- #
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

# Adding confidence interval
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
# ---------------------------------------------------------------- #

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

# Add legend manually
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
# ----------------------------------------------------- #
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

# Major Findings # 4, Dropdown selection with highway spending vs select economic metric 
def create_highway_econ_figure(indicator="TTLCONS"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=econ_time_df["year"],
        y=econ_time_df["chained_value"],
        mode="lines+markers",
        name="Transportation Spending",
        yaxis="y1"
    ))
    
    indicator_map = {
    "GDP": "gdp",
    "PPIACO": "ppi",
    "TTLCONS": "total_construction_spending"
}

    fig.add_trace(go.Scatter(
        x=econ_time_df["year"],
        y=econ_time_df[indicator_map[indicator]],
        mode="lines+markers",
        name=indicator,
        yaxis="y2"
    ))

    fig.update_layout(
        title="Transportation Spending vs Economic Indicator",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Transportation Spending (Chained)", side="left"),
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
        
        
        html.P("Figure 1: NHCCI Seasonally Adjusted Value Over Time", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This plot reveals a noticeable acceleration in highway construction costs post-2020, suggesting inflationary pressures or increased infrastructure investments during the recovery phase after COVID-19.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_simple_cost_over_time.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 2: Differenced NHCCI Series (Stationarity Confirmed)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Differencing the series confirmed statistical stationarity, allowing reliable SARIMA modeling by neutralizing trend and variance drift.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_stationary_differenced.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 3: ACF and PACF Plots to Guide Parameter Selection", style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.P(
            "These plots were used to identify suitable SARIMA parameters, revealing a strong autocorrelation at lag 1 and partial autocorrelation at lag 2. The ACF plot suggests that past values have a significant influence on the current value, especially at short lags, while the PACF plot helps pinpoint the exact lags with direct impact. This pattern supports including 1–2 autoregressive terms in the SARIMA model to capture the data's memory structure.",
            style={
                'color': '#cbd5e0',
                'fontSize': '1rem',
                'textAlign': 'center',
                'maxWidth': '900px',
                'margin': '20px auto',
                'marginBottom': '20px'
        }),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_ACF_PACF.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 4: Forecast vs Actual NHCCI with 95% Confidence Interval (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This SARIMA forecast shows how predicted NHCCI values align closely with actual data, with 95% confidence intervals shaded in orange. This alignment supports model accuracy, and the widening intervals indicate increasing uncertainty further into the future. This visualization helps forecast infrastructure costs and plan for volatility.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '20px'}),
        dcc.Graph(figure=fig_forecast),

        html.P("Figure 5: Forecast Residuals – Actual vs. Predicted NHCCI (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P(
            "This time series residual plot includes a horizontal reference line at zero to assess forecast bias. Residuals above zero indicate the model underpredicted actual values, while residuals below zero reflect overpredictions. The line helps track how prediction errors evolve over time and confirms that the model maintains low bias without strong autocorrelation.",
            style={
                'color': '#cbd5e0',
                'fontSize': '1rem',
                'textAlign': 'center',
                'marginTop': '20px',
                'maxWidth': '900px',
                'margin': '20px auto',
                'marginBottom': '16px'
        }),
        dcc.Graph(figure=fig_residuals),

        html.P("Figure 6: Forecasted NHCCI Using Recent Trends", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Using only recent NHCCI data, this forecast projects a steady upward trend in construction costs. The widening confidence band reflects growing uncertainty but points to sustained inflation. This projection is crucial for budget planning and highlights the long-term financial implications of current cost momentum in the National Highway Cost Construction Index.", 
               style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_extended_forecast.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'})

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),

    html.Div([
        html.H1("Major Finding #2: NHCCI K-Means Investment Cluster Analysis", style={
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

        html.P("Figure 1: K-Means Cluster Assignment Per Quarter", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Each quarter was assigned to a spending cluster. High and Low Investment periods alternate based on macroeconomic signals and NHCCI changes.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_cluster_per_quarter_assignment.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 2: NHCCI Distribution by Investment Cluster", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This boxplot compares NHCCI values across periods classified as High vs. Low Investment. High Investment periods exhibit a greater median NHCCI and wider range, suggesting increased construction activity and higher costs. The implication is that intense investment phases coincide with greater market volatility and cost escalation in highway projects.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '20px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_distribution_by_cluster_investment.png", style={'width': '70%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 3: NHCCI Over Time by Investment Cluster (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This time series shows how NHCCI values evolved over time, segmented by spending clusters. Blue lines (high investment) align with upward trends in the cost index, while red (low investment) often coincide with plateaus or dips. This segmentation suggests that strategic investment decisions may drive or respond to shifts in construction cost inflation.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '20px'}),
        dcc.Graph(figure=fig_segmented),

        html.P("Figure 4: Normalized Cluster Center Comparison Across Economic Indicators (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This line plot compares normalized macroeconomic indicators between High and Low Investment clusters. High Investment periods correlate with higher GDP and construction employment but lower unemployment (UNRATE), implying a link between public infrastructure spending and stronger economic performance. It underscores how economic context influences spending behavior.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        dcc.Graph(figure=fig_cluster_centers),


        html.P("Figure 5: Top 10 Components Driving Cluster Differences", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Asphalt and Bridge were the most influential components distinguishing between High and Low Investment clusters.", 
               style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/NHCCI_Plots/nhcci_top10_component_investment_clusters.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
        
    html.Div([
        html.H1("Major Finding #3: Regression on GMM Results", style={
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
        'padding': '60px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
    }),
        
    html.Div([
        html.H1("Major Finding #4: TPFS - Transportation Infrastructure Spending Time-Series Trends", style={
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
        'background': 'linear-gradient(to right, #0f172a, #1e3a8a)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    html.Div([
        html.H3("Key Visualizations", style={'color': '#fbbf24', 'textAlign': 'center'}),
        
        html.P("Figure 1: Transportation Spending vs Selected Economic Metric (Interactive)", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("User can select an economic indicator (GDP, PPI, or Total Construction) to visualize its relationship with highway spending over time.", 
               style={'textAlign': 'center', 'color': '#cbd5e0'}),
        dcc.Dropdown(
            id="indicator_dropdown",
            options=[
                {"label": "total_construction_spending", "value": "TTLCONS"},
                {"label": "ppi", "value": "PPIACO"},
                {"label": "gdp", "value": "GDP"}
            ],
            value="TTLCONS",
            style={'width': '60%', 'margin': '20px auto'}
        ),
        dcc.Graph(id="highway_econ_graph"),
    
        html.P("Figure 2: Time Series Decomposition - Federal Transportation Infrastructure Spending", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("Decomposition reveals that federal spending trends upward post-2020 without notable seasonality.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_decomp_fed.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 3: Time Series Decomposition - State and Local Transportation Spending", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("State and local spending followed a gradual rise until 2020, before slight declines due to COVID-related budgetary constraints.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_decomp_statelocal.png", style={'width': '90%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 4: Cross-Correlation of Spending Leading GDP", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This plot reveals that transportation infrastructure spending is positively correlated with GDP growth, particularly at a 2-year lag. It suggests that public investment stimulates economic output over time lagged periods.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_GDP_spending.png", style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),

        html.P("Figure 5: Cross-Correlation of Spending Leading PPI", style={'textAlign': 'center', 'color': '#cbd5e0'}),
        html.P("This cross-correlation plot shows a peak at lag 2, indicating that increases in transportation infrastructure spending precede rises in the Producer Price Index (PPI) by roughly two years. This suggests that public investment has a delayed but measurable inflationary effect on construction materials/services/etc.", 
               style={'textAlign': 'center', 'color': '#cbd5e0', 'marginBottom': '16px'}),
        html.Img(src="https://storage.googleapis.com/databucket_seniorproj/TPFS_Plots/TPFS_time_series_PPI_spending.png", style={'width': '80%', 'margin': '30px auto', 'display': 'block'}),

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
])
