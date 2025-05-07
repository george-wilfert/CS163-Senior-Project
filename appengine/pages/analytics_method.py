import dash
from dash import html, dcc, Input, Output, State, register_page, callback_context
import dash_bootstrap_components as dbc

# Register the page
register_page(__name__, path="/analytical", name="Analytical Methods")

# Load images function
def load_images(year_range, group):
    # Proper base filename (strip the _Clustering part for filename)
    base_filename = group.replace('_Clustering', '')
    base_url = f"https://storage.googleapis.com/databucket_seniorproj/{group}/{base_filename}_"
    images = []
    for year in range(year_range[0], year_range[1] + 1):
        image_url = f"{base_url}{year}.png"
        images.append(image_url)


    return images

# Layout for the page
layout = html.Div([
    # Header
    html.Div([
        html.H1("Analytical Methods", style={
            'fontSize': '3rem',
            'color': '#38bdf8',
            'marginBottom': '20px'
        }),
        html.P("The data-driven techniques behind our infrastructure insights.",
               style={'fontSize': '1.3rem', 'color': '#cbd5e0'}),
    ], style={
        'background': 'linear-gradient(to right, #0f172a, #1e293b)',
        'padding': '100px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
    }),

    # Energy Infrastructure Section
html.Div([
    html.H2("Energy Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center', 'textDecoration': 'underline'}),
    html.H3("Method: GMM Model", style={'color': '#38bdf8', 'textAlign': 'center'}),

    html.P(
            "A Gaussian Mixture Model (GMM) is a way to group data into clusters based on the idea that the data comes from a mix of several normal distributions. Each cluster is represented by one of these distributions, and the model tries to figure out the best fit. Unlike methods like k-means, which assigns each data point to one cluster, GMM gives a probability for each point to belong to multiple clusters, which makes it a bit more flexible for more complex patterns in the data. Below are three sliders to cluster all the EIA data points into clusters for three different categories of energy in order to use this data as a signal for total infrastructure spending.",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),
    # First Group
    html.H4("Natural Gas Infrastructure", style={'color': '#38bdf8', 'textAlign': 'center'}),
    dcc.RangeSlider(
        id='year-range-slider-1',
        min=2000,
        max=2021,
        step=1,
        marks={year: str(year) for year in range(2000, 2022, 2)},
        value=[2000, 2005],
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    html.Div(id='slideshow-container-1', style={'textAlign': 'center', 'margin': '30px 0'}),
    html.Div([
        dbc.Button("Previous", id="prev-button-1", color="primary", className="me-2", n_clicks=0),
        dbc.Button("Next", id="next-button-1", color="primary", n_clicks=0),
    ], style={'textAlign': 'center'}),
    dcc.Store(id='image-index-1', data=0),
    dcc.Store(id='image-list-1', data=load_images([2000, 2005],group="Natural_Gas_Clustering")),  

    html.Hr(style={'margin': '40px 0'}),

    # Second Group
    html.H4("Coal Infastructure", style={'color': '#38bdf8', 'textAlign': 'center'}),
    dcc.RangeSlider(
        id='year-range-slider-2',
        min=2000,
        max=2021,
        step=1,
        marks={year: str(year) for year in range(2000, 2022, 2)},
        value=[2000, 2005],
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    html.Div(id='slideshow-container-2', style={'textAlign': 'center', 'margin': '30px 0'}),
    html.Div([
        dbc.Button("Previous", id="prev-button-2", color="primary", className="me-2", n_clicks=0),
        dbc.Button("Next", id="next-button-2", color="primary", n_clicks=0),
    ], style={'textAlign': 'center'}),
    dcc.Store(id='image-index-2', data=0),
    dcc.Store(id='image-list-2', data=load_images([2000, 2005],group="Coal_Clustering")),  

    html.Hr(style={'margin': '40px 0'}),

    # Third Group (Electricity)
        html.H4("Electricity Infrastructure", style={'color': '#38bdf8', 'textAlign': 'center'}),
        dcc.RangeSlider(
            id='year-range-slider-3',
            min=2000,
            max=2021,
            step=1,
            marks={year: str(year) for year in range(2000, 2022, 2)},
            value=[2000, 2005],
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id='slideshow-container-3', style={'textAlign': 'center', 'margin': '30px 0'}),
        html.Div([
            dbc.Button("Previous", id="prev-button-3", color="primary", className="me-2", n_clicks=0),
            dbc.Button("Next", id="next-button-3", color="primary", n_clicks=0),
        ], style={'textAlign': 'center'}),
        dcc.Store(id='image-index-3', data=0),
        dcc.Store(id='image-list-3', data=load_images([2000, 2005], group="Electricity_Clustering")),

    ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'}),
    #Regression Anyalsis
    html.Div([
        html.H3("Regression Analysis", style={'color': '#38bdf8', 'textAlign': 'center'}),
        html.P(
            "As mentioned above the GMM can give us a lot of very important information that can be used in future types such as regression analysis. The biggest goal of the regression at first is to justify the assumption that I made originally that the energy consumption can be used as a signal for infrastructure. This was done through our first regression analysis using a Random Forest Regressor that predicts the expenditures on infrastructure in a state. After using this assumption we extrapolated out to other factors to justify our hypothesis that infrastructure is correlated with GDP and Urbanization metrics. Using the regression we also get some feature importance graphs that tell us which classifications are important to determining high expenditure and GDP metrics.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),
    
    
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
    # Highway Infrastructure Section
    html.Div([
        html.H2("Highway Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center', 'textDecoration': 'underline'}),
        html.H3("Method 1: NHCCI SARIMAX Model", style={'color' : '#38bdf8', 'textAlign': 'center'}),
    
        html.P(
            "To better understand construction cost trends in highway infrastructure, we utilized a Seasonal AutoRegressive Integrated Moving Average with eXogenous factors (SARIMAX) model on the National Highway Construction Cost Index (NHCCI).",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),
    
        html.P(
            "The SARIMA model was selected for its ability to capture temporal autocorrelation, seasonality (quarterly), and the influence of external economic variables like GDP and total construction spending. Prior to modeling, we conducted stationarity testing using the Augmented Dickey-Fuller (ADF) test, applied first-order differencing, and validated the stationarity of the transformed series.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),
    
        html.P(
            "Model diagnostics included visualization of residuals, forecasting with confidence intervals, and evaluation using RMSE (Root Mean Squared Error). ACF and PACF plots were used to guide the selection of model parameters, while standardization of exogenous variables helped ensure stability in parameter estimation. The model showed strong fit characteristics, capturing the seasonal patterns and general trend in NHCCI.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),
    
        html.P(
            "References: Box & Jenkins (1970), 'Time Series Analysis: Forecasting and Control'; Hyndman & Athanasopoulos (2018), 'Forecasting: Principles and Practice'.",
            style={'color': '#9ca3af', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        )
    ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'}),    
    
    html.Div([
        html.H2("Method 2: NHCCI Spending Cluster Analysis", style={'color': '#38bdf8', 'textAlign': 'center'}),

        html.P(
            "To analyze patterns in highway construction cost changes and their relationship with macroeconomic factors, we applied K-Means clustering on quarterly NHCCI data. Each data point was represented by a feature set that included percentage changes in construction components (e.g., asphalt, grading, bridge), as well as key economic indicators such as GDP, unemployment rate (UNRATE), PPI (PPIACO), and construction employment.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),

        html.P(
            "Data preprocessing included handling missing values and standardizing the selected features using `StandardScaler` to ensure fair distance-based clustering. K-Means with two clusters (k=2) was chosen based on interpretability, capturing distinct 'High Investment' and 'Low Investment' spending periods across quarters.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "We assigned each quarter to a cluster and examined differences in NHCCI trends and construction cost components. To further understand the characteristics of each cluster, we visualized their centers (in original and normalized economic scale) and identified the top ten construction components that showed the greatest variance across clusters.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "References: Jain & Dubes (1988), 'Algorithms for Clustering Data'; Han et al. (2011), 'Data Mining: Concepts and Techniques'; Scikit-learn Documentation on KMeans Clustering.",
            style={'color': '#9ca3af', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        )
    ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
        html.H2("Transportation Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center', 'textDecoration': 'underline'}),
        html.H3("Method: TPFS Time Series Decomposition & Cross-Correlation Analysis", style={'color': '#38bdf8', 'textAlign': 'center'}),

        html.P(
            "To better understand long-term trends, seasonal patterns, and cyclical behavior in transportation infrastructure spending, we applied classical time series decomposition (additive model) to the TPFS dataset. Spending was grouped by government level and mode (e.g., Highways, Transit) to separate trend, seasonal, and residual components.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),

        html.P(
            "Following decomposition, we conducted lagged cross-correlation analysis between highway spending and macroeconomic indicators (GDP, PPI, total construction spending). This analysis measured how changes in infrastructure investment aligned with or preceded shifts in broader economic conditions. Positive leading correlations would suggest infrastructure spending could act as a predictor of future economic health.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "Visualization included multi-axis time series plots comparing highway spending against economic metrics and lagged correlation bar charts to detect temporal dynamics. Insights from decomposition and correlation guided further modeling strategies.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "References: Cleveland et al. (1990), 'STL: A Seasonal-Trend Decomposition Procedure Based on Loess'; Hyndman & Athanasopoulos (2018), 'Forecasting: Principles and Practice'; Box & Jenkins (1970), 'Time Series Analysis: Forecasting and Control'.",
            style={'color': '#9ca3af', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        )
    ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
        html.H2("Additional Transportation & Highway Methods from NHCCI & TPFS (less relevant)", 
                style={'color': '#fbbf24', 'textAlign': 'center', 'textDecoration': 'underline'}),
        html.H3("Additional Method #1: TPFS Multivariate Regression w/ Ridge Regularization", style={'color': '#f8f9fa', 'textAlign': 'center'}),
    
        html.P(
            "To explore interactions between government levels, modes of transportation, and their influence on transportation infrastructure spending patterns, we implemented Ridge Regression (L2 regularization). Ridge regression was selected due to its ability to handle multicollinearity and stabilize coefficient estimates, especially when interaction terms are introduced.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),

        html.P(
            "The model used log-transformed spending values (log(1 + x)) as the target variable and incorporated interaction terms between 'gov_level' and 'mode' through formula-based feature engineering (`patsy`). We standardized predictors to ensure comparability before applying Ridge regularization. Separate models were trained for Capital and Non-Capital expenditures.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "Model performance was evaluated using out-of-sample R² scores, and residual plots were examined for patterns or heteroskedasticity. Coefficient analysis identified key government-mode interactions associated with higher or lower log spending, supporting the hypothesis that expenditure drivers vary across governance structures and transportation modes.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "References: Hoerl & Kennard (1970), 'Ridge Regression: Biased Estimation for Nonorthogonal Problems'; James et al. (2013), 'An Introduction to Statistical Learning'; Scikit-learn Ridge Regression Documentation.",
            style={'color': '#9ca3af', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        )
    ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'}),

    html.Div([
        html.H2("Additional Method #2: TPFS Anomaly Detection using Isolation Forest", style={'color': '#f8f9fa', 'textAlign': 'center'}),

        html.P(
            "To identify unusual transportation infrastructure spending behavior, we applied an Isolation Forest model on the TPFS dataset. Isolation Forest is an unsupervised anomaly detection method that works by isolating observations through recursive partitioning. Observations that are easier to isolate are considered anomalies.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),

        html.P(
            "The model was run separately across each combination of transportation mode and government level to account for differences in spending patterns. Spending data was first filtered to remove missing values and then grouped accordingly. The model flagged approximately 10% of entries as anomalous, which were then analyzed to detect outliers in inflation-adjusted ('chained') values.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "We visualized the frequency of anomalies using heatmaps across time and spending categories, and produced a ranked list of the most anomalous entries using the associated project descriptions. These results help uncover potential data errors, overspending patterns, or policy shocks within specific modes and jurisdictions.",
            style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        ),

        html.P(
            "References: Liu et al. (2008), 'Isolation Forest'; Breunig et al. (2000), 'LOF: Identifying Density-Based Local Outliers'; Scikit-learn IsolationForest Documentation.",
            style={'color': '#9ca3af', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}
        )
    ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
        html.H2("Additional Method #3: Multivariate Regression with Lasso Regularization", style={'color': '#f8f9fa', 'textAlign': 'center'}),
        html.P("To isolate the most impactful macroeconomic predictors for NHCCI changes, we implemented a multivariate regression model using LassoCV (L1 regularization). The Lasso method is well-suited for high-dimensional data, promoting sparse solutions by shrinking less relevant coefficients to zero.", style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}),
        html.P("We focused on lagged versions of PPI (PPIACO) and Total Construction Spending (TTLCONS) as inputs to align with our hypothesis that prior economic activity forecasts construction cost trends. The model employed time-aware cross-validation (TimeSeriesSplit) and `StandardScaler` normalization to ensure compatibility with the penalty function.", style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}),
        html.P("The model returned optimal alpha selection using cross-validation, reported out-of-sample R² for test set accuracy, and allowed for visualization of retained macroeconomic variables. Selected variables were validated using a correlation matrix and interpretation of the Lasso coefficients.", style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}),
        html.P("References: Tibshirani (1996), 'Regression Shrinkage and Selection via the Lasso'; Hastie, Tibshirani, Friedman (2009), 'The Elements of Statistical Learning'; Scikit-learn LassoCV Documentation.", style={'color': '#9ca3af', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'})
        ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'})
    
    ], style={'padding': '30px 20px', 'backgroundColor': '#1f2937'})

# Helper function to create similar callbacks with dynamic group
def create_callbacks(slider_id, prev_id, next_id, image_list_id, image_index_id, slideshow_container_id, group):
    @dash.callback(
        Output(image_list_id, 'data'),
        Output(image_index_id, 'data'),
        Input(slider_id, 'value'),
        Input(prev_id, 'n_clicks'),
        Input(next_id, 'n_clicks'),
        State(image_index_id, 'data'),
        State(image_list_id, 'data')
    )
    def update_images(year_range, prev_clicks, next_clicks, current_index, image_list):
        ctx = callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Call the load_images function with the group parameter
        if trigger_id == slider_id:
            new_images = load_images(year_range, group)  # Passing the group here
            return new_images, 0

        if not image_list:
            return dash.no_update, 0

        if trigger_id == next_id:
            current_index = (current_index + 1) % len(image_list)
        elif trigger_id == prev_id:
            current_index = (current_index - 1) % len(image_list)

        return dash.no_update, current_index

    @dash.callback(
        Output(slideshow_container_id, 'children'),
        Input(image_index_id, 'data'),
        State(image_list_id, 'data')
    )
    def display_image(index, image_list):
        if not image_list:
            return html.P("No images found for the selected range.", style={'color': 'red'})
        return html.Img(src=image_list[index], style={'width': '60%', 'height': 'auto'})


# Create the 3 callback sets with dynamic group names
create_callbacks('year-range-slider-1', 'prev-button-1', 'next-button-1', 'image-list-1', 'image-index-1', 'slideshow-container-1', "Natural_Gas_Clustering")
create_callbacks('year-range-slider-2', 'prev-button-2', 'next-button-2', 'image-list-2', 'image-index-2', 'slideshow-container-2', "Coal_Clustering")
create_callbacks('year-range-slider-3', 'prev-button-3', 'next-button-3', 'image-list-3', 'image-index-3', 'slideshow-container-3', "Electricity_Clustering")
