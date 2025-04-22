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
            "Energy plays such a crucial role in every aspect of infrastructure. When looking from a top down level there is energy in everything we do. Infrastructure in a large sense is reliant on energy so diving deeper into a region's energy charastics gives us the ability to explore inferences on not only their infrastructure as a whole but also the impacts of the infrastructure such as GDP, employment, and urbanization. When you think of highly urbanized areas with highly developed infrastructure there is a large amount of energy required to fulfill demands of whether that be manufacturing or everyday use of lights in school buildings. Using an API from the Energy Information Administration we can collect all types of data from states over years to inference about. Below is an example using the API which gives us data about the electricity per state consumed.",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),
        html.Img(
                src="https://storage.googleapis.com/databucket_seniorproj/EDA_Graphs/choropleth_map.png",
                style={
                    "width": "60%",          
                    "maxWidth": "600px",     
                    "display": "block",      
                    "margin": "auto",        
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
                }
            ),
            html.Img(
                src="https://storage.googleapis.com/databucket_seniorproj/EDA_Graphs/barplot2022.png",
                style={
                    "width": "60%",          
                    "maxWidth": "600px",     
                    "display": "block",      
                    "margin": "auto",        
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
                }
            ),
        html.P(
            "Our analysis on energy looks at the impacts on energy categories, such as natural gas, coal, and electricity. Using these categories we can cluster each state into similar groups which can further help our analysis. Some examples of some of the data points we collected are “Natural Gas Used by The Electrical Grid” and “Natural Gas Pipeline and Distribution Use”. This can help answer hypothesis 1 by correlating factors like GDP, urbanization, and productivity to our clusters. Using our clusters we can see which features are important to determining the previously mentioned factors. Using randomforestregressor models we can see correlation and feature importance.  From a sourcing standpoint, being able to tell similarities is important because we can see where other states are similar conditions which can be used to source projects based on key features. Also looking at the change over years gives an important variable that helps predict future costs and thus future effects on our hypothesis 1 factors. For hypothesis 2 we did a regression analysis on expenditures and used urbanization as a feature along with the hidden states in our clusters containing information about it to see how that urbanization plays a role in infrastructure,",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        ),
        
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'}),
    html.Div([
        html.H2("Transportation Infastrucuture", style={'color': '#fbbf24', 'textAlign': 'center'}),
        html.P(
            "",style={'color': '#cbd5e0', 'fontSize': '1.1rem', 'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto'}
        )
    ], style={'padding': '60px 20px', 'backgroundColor': '#1f2937'})
    
])