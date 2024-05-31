from dash import Dash, callback, html, dcc
import dash_bootstrap_components as dbc
import dash
import gunicorn                     #whilst your local machine's webserver doesn't need this, Heroku's linux webserver (i.e. dyno) does. I.e. This is your HTTP server
from whitenoise import WhiteNoise   #for serving static files on Heroku

# Instantiate dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Reference the underlying flask app (Used by gunicorn webserver in Heroku production deployment)
server = app.server 

# Enable Whitenoise for serving static files from Heroku (the /static folder is seen as root by Heroku) 
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/') 


app = Dash(__name__, assets_folder='./assets', use_pages=True, pages_folder="./src/pages",
           suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Nav([
        html.Div(className="nav-container", children=[
            html.Div(className="nav-logo", children='ATI Results Overview'),
            html.Div(className="nav-links", children=
            [
                html.Div(
                    dcc.Link(f"{page['name']}", href=page["relative_path"])
                )
                for page in dash.page_registry.values()
            ]
                     ),
        ]),
    ]),
    html.Div(id="overlay", className="page-content", children=[
        html.Div(className="text", children= ['Running...']),
    ]),
    dash.page_container
])


# Run flask app
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
