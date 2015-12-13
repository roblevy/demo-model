"""
A web server which will listen for requested changes via
POST and GET requests, and respond with CSV data summarising
the state of a model following the requested changes.
"""
from flask import Flask, render_template, request, make_response
from flask.ext.cors import CORS
from functools import update_wrapper
import pandas as pd
from demo_model.global_demo_model import GlobalDemoModel as gdm
from demo_model.tools import sectors, dataframe, metrics
import demo_model.trademodels.exportness as exportness
import json
import time
import StringIO
import os

app = Flask(__name__)
CORS(app) # enable cors for all routes
model = None
DEBUG = False # Enables auto-restart when this file is saved
CURRENT_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURRENT_PATH, '..', 'models/')
RESOURCE_PATH = os.path.join(CURRENT_PATH, '..', 'resources/')
SECTOR_ID = {
    'all':0,
    'primary':1,
    'secondary':2,
    'raw':3,
    'trade':4,
    'services':5,
    'transport':6,
    'public':7
}

def load_model(model_file='model2005_estimates.gdm', model_path=None,
        regression_file='exportness_regression.csv', regression_path=None):
    """
    Load a model from a specified pickle and stored in
    global variable `model`.
    If no model_path is specified, reverts to MODEL_PATH
    """
    global model, exportness
    print "Loading model..."
    if model_path is None:
        model_path = MODEL_PATH
    if regression_path is None:
        regression_path = RESOURCE_PATH
    model = gdm.from_pickle(model_path + model_file)
    print "Initialising..."
    model = exportness.initialise_model_with_exportness(model, regression_path + regression_file)
    model.original_exportness = model.exportness.copy()
    model.original_import_propensities = model._import_propensities.copy()
    print "Model loaded"
    return model

def response(response_text):
    r = make_response(response_text)
    #r.headers['Access-Control-Allow-Origin'] = '*'
    #r.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    #r.headers["Access-Control-Max-Age"] = "1000"
    #r.headers["Access-Control-Allow-Headers"] = "*"
    return r

def print_args(args):
    print args

@app.route('/')
def show_index():
    return "Model year: %s" % str(model.year)
    #return response(render_template('index.html'))

@app.route('/reset', methods=['GET'])
def reset_model():
    print "Model is being reset..."
    load_model()
    print "Model reset."
    return "Model reset at %s" % time.ctime()

@app.route('/flows', methods=['GET'])
def get_flows():
    print "getting flows..."
    #model_output = model.flows_to_json(model.trade_flows(countries, sectors))
    #return response(model_output)
    args = request.args
    print_args(args)
    param_from = parse_comma_separated_param(args.get('from'))
    param_to = parse_comma_separated_param(args.get('to'))
    param_relation = args.get('relation')
    flows = model.trade_flows(from_country=param_from, to_country=param_to)
    flows = sectors.aggregate_to_supersectors(flows)
    flows = replace_sectors_with_ids(flows)
    return to_string_csv(flows, header=True)

@app.route('/attributes', methods=['GET'])
def get_attributes():
    print "getting attributes..."
    args = request.args
    print_args(args)
    param_country = parse_comma_separated_param(args.get('country'))
    countries = [model.countries[c] for c in param_country]
    return to_string_csv(gdp_dataframe(countries), index=False)

@app.route('/change/', methods=['POST'])
def change_model():
    print "model change requested"
    form = dict_from_multidict(request.form)
    print_args(form)
    return route_change_request(form)

def route_change_request(form):
    """
    `form` is a dict of POST key/value pairs
    """
    change_type = form['change_type']
    if str(change_type) == "1":
        change_export_attractiveness(**form)
    elif str(change_type) == "2":
        change_trade_relationship(**form)
    else:
        return "error"
    return "ok"

def gdp_dataframe(countries):
    """
    A DataFrame of GDP figures from `model`, restricted to `countries`
    """
    gdp = {c.name:metrics.country_metrics(c)['tva'] for c in countries}
    gdp_df = dict_to_df(gdp, columns=['country', 'value'])
    gdp_df['key'] = 'GDP'
    return gdp_df

def change_export_attractiveness(country1, slider_value, sector_id=None, **kwargs):
    """
    Change the export attractiveness according to `value` which has
    arbitrary units between -1 and 1.

    Optionally change only the export attractiveness of the sector
    specified by `sector_id`
    """
    print "Changing export attractiveness"
    print "country: %s, value: %s, sector: %s" % (country1, slider_value, sector_id)
    model.exportness[country1] = new_exportness(model, country1, slider_value)
    replace_import_propensities(model, sector_id)
    print "Recalculating world..."
    model.recalculate_world()
    return "ok"

def new_exportness(model, country1, slider_value):
    """
    Split the difference between current exportness of `country1` and the
    min and max values across `model`
    """
    slider_value = float(slider_value)
    min_exportness = model.exportness.min()
    max_exportness = model.exportness.max()
    current_exportness = model.exportness[country1]
    if slider_value > 0:
        endpoint = max_exportness
    elif slider_value < 0:
        endpoint = min_exportness
    else:
        endpoint = current_exportness
    return current_exportness + ((endpoint - current_exportness) * slider_value)

def replace_import_propensities(model, sector_id):
    """
    Replace either all import propensities or just those relating to `sector_id`

    Uses `model.exportness`. `sector_id` is either a single integer or a list of integers.
    """
    new_import_propensities = exportness.estimate_import_propensities(exportness=model.exportness)
    if sector_id is None:
        model._import_propensities = new_import_propensities
    else:
        sectors = sector_names_from_id(sector_id)
        for s in sectors:
            model._import_propensities[s] = new_import_propensities[s]

def sector_names_from_id(sector_id, get_supersectors=False):
    """
    A list of sector names (either supersectors or normal sectors) given a sector id
    or list of sector_ids.
    """
    if sector_id is None:
        supersectors = SECTOR_ID.keys()
    else:
        if not isinstance(sector_id, (list, tuple)):
            sector_id = [int(sector_id)]
        sector_id = [int(sid) for sid in sector_id]
        supersectors = [k for k, v in SECTOR_ID.iteritems() if v in sector_id]
    if get_supersectors:
        return supersectors
    else:
        return sectors.sectors_from_supersector(supersectors)

def change_trade_relationship(country1, country2, slider_value, sector_id=None, **kwargs):
    """
    """
    print "Changing trade relationship"
    print "country1: %s, country2: %s, value: %s, sector: %s" % (country1, country2, slider_value, sector_id)

def parse_comma_separated_param(param):
    """
    Return a list of single parameters
    """
    return str(param).split(",")

def dict_from_multidict(multidict):
    """
    Create a simple dict from an import ImmutableMultiDict
    """
    return {k:v for k, v in multidict.iteritems()}

def to_string_csv(df_or_series, **kwargs):
    """
    A string representation of a CSV-format file
    """
    s = StringIO.StringIO()
    df_or_series.to_csv(s, **kwargs)
    return s.getvalue()

def dict_to_df(dictionary, columns):
    """
    Convert a dictionary into a two-column DataFrame with
    column names specified by `columns`
    """
    tuples = [(k, v) for k, v in dictionary.iteritems()]
    return pd.DataFrame(tuples, columns=columns)

def replace_sectors_with_ids(df_or_series, level='sector'):
    """
    Use `SECTOR_ID` to replace sector names with IDs in the MultiIndex level
    specified by `level`
    """
    idx_vals = df_or_series.index.get_level_values(level)
    new_vals = [SECTOR_ID[v] for v in idx_vals]
    retval = dataframe.set_index_values(df_or_series, new_vals, level)
    return retval

def start_server():
    load_model()
    app.run(debug=DEBUG, host='0.0.0.0')

if __name__ == "__main__":
    start_server()

