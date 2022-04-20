import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pickle
import json
from sklearn import metrics
from dash.dependencies import Input, Output, State

########### Define your variables ######
myheading1='Predicting Mortgage Loan Approval'
image1='assets/rocauc.html'
tabtitle = 'Loan Prediction'
sourceurl = 'https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/'
githublink = 'https://github.com/austinlasseter/simple-ml-apps'

########### open the json file ######
with open('assets/rocauc.json', 'r') as f:
    fig=json.load(f)

########### open the pickle file ######
filename = open('analysis/loan_approval_logistic_model.pkl', 'rb')
unpickled_model = pickle.load(filename)
filename.close()

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),

    html.Div([
        html.Div(
            [dcc.Graph(figure=fig, id='fig1')
            ], className='six columns'),
        html.Div([
                html.H3("Features"),
                html.Div('Credit History:'),
                dcc.Input(id='Credit_History', value=1, type='number', min=0, max=1, step=1),
                html.Div('Loan Amount (in thousands):'),
                dcc.Input(id='LoanAmount', value=130, type='number', min=10, max=800, step=10),
                html.Div('Term (in months)'),
                dcc.Input(id='Loan_Amount_Term', value=360, type='number', min=120, max=480, step=10),
                html.Div('Applicant Income (in dollars)'),
                dcc.Input(id='ApplicantIncome', value=5000, type='number', min=0, max=100000, step=500),
                html.Div('Probability Threshold for Loan Approval'),
                dcc.Input(id='Threshold', value=50, type='number', min=0, max=100, step=1),

            ], className='three columns'),
            html.Div([
                html.H3('Predictions'),
                html.Div('Predicted Status:'),
                html.Div(id='PredResults'),
                html.Br(),
                html.Div('Probability of Approval:'),
                html.Div(id='ApprovalProb'),
                html.Br(),
                html.Div('Probability of Denial:'),
                html.Div(id='DenialProb')
            ], className='three columns')
        ], className='twelve columns',
    ),


    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
    ]
)


######### Define Callback
@app.callback(
    [Output(component_id='PredResults', component_property='children'),
     Output(component_id='ApprovalProb', component_property='children'),
     Output(component_id='DenialProb', component_property='children'),
    ],
    [Input(component_id='Credit_History', component_property='value'),
     Input(component_id='LoanAmount', component_property='value'),
     Input(component_id='Loan_Amount_Term', component_property='value'),
     Input(component_id='ApplicantIncome', component_property='value'),
     Input(component_id='Threshold', component_property='value')
    ])
def prediction_function(Credit_History, LoanAmount, Loan_Amount_Term, ApplicantIncome, Threshold):
    try:
        data = [[Credit_History, LoanAmount, Loan_Amount_Term, ApplicantIncome]]
        rawprob=100*unpickled_model.predict_proba(data)[0][1]
        func = lambda y: 'Approved' if int(rawprob)>Threshold else 'Denied'
        formatted_y = func(rawprob)
        deny_prob=unpickled_model.predict_proba(data)[0][0]*100
        formatted_deny_prob = "{:,.2f}%".format(deny_prob)
        app_prob=unpickled_model.predict_proba(data)[0][1]*100
        formatted_app_prob = "{:,.2f}%".format(app_prob)
        return formatted_y, formatted_app_prob, formatted_deny_prob
    except:
        return "inadequate inputs", "inadequate inputs"





############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
