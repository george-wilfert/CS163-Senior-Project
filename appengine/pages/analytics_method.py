import dash
from dash import html, dcc, Input, Output, State, register_page, callback_context
import dash_bootstrap_components as dbc

# Register the page
register_page(__name__, path="/analytical", name="Analytical Methods")

# Load images function
def load_images(year_range):
    base_url = "https://storage.googleapis.com/databucket_seniorproj/Natural_Gas_Clustering/Natural_Gas_"
    images = []
    for year in range(year_range[0], year_range[1] + 1):
        image_url = f"{base_url}{year}.png"
        images.append(image_url)
    print("Loaded images:", images)
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
        html.H2("Energy Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center'}),
        dcc.RangeSlider(
            id='year-range-slider',
            min=2000,
            max=2021,
            step=1,
            marks={year: str(year) for year in range(2000, 2022, 2)},
            value=[2000, 2005],
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id='slideshow-container', style={'textAlign': 'center', 'margin': '30px 0'}),
        html.Div([
            dbc.Button("Previous", id="prev-button", color="primary", className="me-2", n_clicks=0),
            dbc.Button("Next", id="next-button", color="primary", n_clicks=0),
        ], style={'textAlign': 'center'}),
        dcc.Store(id='image-index', data=0),
        dcc.Store(id='image-list', data=load_images([2000, 2005]))

    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
    # Highway Infrastructure Section
    html.Div([
        html.H2("Highway Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center'}),
    
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
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),

    # Transportation Infrastructure Section (Placeholder)
    html.Div([
        html.H2("Transportation Infrastructure", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P("Coming soon...", style={'color': '#cbd5e0', 'textAlign': 'center'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
    html.Div([
    html.H2("Spending Cluster Analysis", style={'color': '#fbbf24', 'textAlign': 'center'}),

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
    ),
    
    html.Div([
        html.H2("Multivariate Regression with Lasso Regularization", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P("To isolate the most impactful macroeconomic predictors for NHCCI changes, we implemented a multivariate regression model using LassoCV (L1 regularization). The Lasso method is well-suited for high-dimensional data, promoting sparse solutions by shrinking less relevant coefficients to zero.", style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}),
        html.P("We focused on lagged versions of PPI (PPIACO) and Total Construction Spending (TTLCONS) as inputs to align with our hypothesis that prior economic activity forecasts construction cost trends. The model employed time-aware cross-validation (TimeSeriesSplit) and `StandardScaler` normalization to ensure compatibility with the penalty function.", style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}),
        html.P("The model returned optimal alpha selection using cross-validation, reported out-of-sample R² for test set accuracy, and allowed for visualization of retained macroeconomic variables. Selected variables were validated using a correlation matrix and interpretation of the Lasso coefficients.", style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'}),
        html.P("References: Tibshirani (1996), 'Regression Shrinkage and Selection via the Lasso'; Hastie, Tibshirani, Friedman (2009), 'The Elements of Statistical Learning'; Scikit-learn LassoCV Documentation.", style={'color': '#9ca3af', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '20px auto'})
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
    
], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    
])

# Combined callback for both year range and buttons
@dash.callback(
    Output('image-list', 'data'),
    Output('image-index', 'data'),
    Input('year-range-slider', 'value'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    State('image-index', 'data'),
    State('image-list', 'data')
)
def update_images(year_range, prev_clicks, next_clicks, current_index, image_list):
    ctx = callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Range slider changed: reload images
    if trigger_id == 'year-range-slider':
        new_images = load_images(year_range)
        return new_images, 0

    if not image_list:
        return dash.no_update, 0

    # Button navigation
    if trigger_id == 'next-button':
        current_index = (current_index + 1) % len(image_list)
    elif trigger_id == 'prev-button':
        current_index = (current_index - 1) % len(image_list)

    return dash.no_update, current_index

# Callback to display current image
@dash.callback(
    Output('slideshow-container', 'children'),
    Input('image-index', 'data'),
    State('image-list', 'data')
)
def display_image(index, image_list):
    if not image_list:
        return html.P("No images found for the selected range.", style={'color': 'red'})
    return html.Img(src=image_list[index], style={'width': '60%', 'height': 'auto'})
