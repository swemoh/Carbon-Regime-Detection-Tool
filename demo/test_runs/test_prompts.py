import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Checkbox Example"),
    dcc.Checklist(
        id='checkboxes',
        options=[
            {'label': 'Option 1', 'value': 'opt1'},
            {'label': 'Option 2', 'value': 'opt2'},
            {'label': 'Option 3', 'value': 'opt3'}
        ],
        value=['opt1'],  # Set the default selected values
        labelStyle={'display': 'block'}  # Display the labels in a new line
    ),
    html.Div(id='output')
])

# Create a callback to update the output based on the checkbox selection
@app.callback(
    Output('output', 'children'),
    [Input('checkboxes', 'value')]
)
def update_output(selected_values):
    return f"You have selected: {', '.join(selected_values)}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
