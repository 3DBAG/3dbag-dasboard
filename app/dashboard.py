import os
from pathlib import Path
import logging
import io
import base64

from dash import Dash, html, dcc, dash_table
import plotly.graph_objects as go
import flask
import json
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import plotly.io as pio
import dash_bootstrap_components as dbc
import seaborn as sns

sns.set_palette("muted")
pio.templates.default = "seaborn"

logging.basicConfig(encoding='utf-8', level=logging.INFO)
log = logging.getLogger(__name__)

# --- Server and App

server = flask.Flask(__name__)  # define flask app.server

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           server=server,
           url_base_pathname='/dashboard/'
           )  # call flask server

# --- Prep input

data_dir = Path(os.environ.get("BAG3D_DATA_DIR", "data"))
log.info(f"{data_dir=}")

## Lineage
metadata_json = data_dir / "metadata.json"
with open(metadata_json, "r") as fo:
    metadata = json.load(fo)
log.info(f"Loaded {metadata_json}")

df_software = pd.DataFrame.from_records(
    metadata["dataQualityInfo"]["lineage"]["software"])
df_software.loc[df_software["name"] == "geoflow-bundle", "version"] = \
    df_software.loc[df_software["name"] == "geoflow-bundle", "version"].values[
        0].replace(
        " Plugins:", "\nPlugins:").replace(" >", "\n >")
df_software["repository"] = df_software.repository.apply(
    lambda x: f"[link]({x})").astype(str)

df_steps = pd.DataFrame.from_records(
    metadata["dataQualityInfo"]["lineage"]["processStep"]).rename(
    columns={"name": "step"})
df_steps = df_steps[["step", "featureCount", "dataVersion", "dateTime", "runId"]]
count_reconstruction_input = int(df_steps.loc[df_steps[
                                                  "step"] == "input.reconstruction_input", "featureCount"].values[
                                     0])

## Compressed files
validate_compressed_files_parquet = data_dir / "validate_compressed_files.parquet"
validated_compressed = pd.read_parquet(validate_compressed_files_parquet,
                                       engine="pyarrow")
log.info(f"Loaded {validate_compressed_files_parquet}")

## Reconstruction results
reconstructed_features_parquet = data_dir / "reconstructed_features.parquet"
reconstructed_features = pd.read_parquet(reconstructed_features_parquet,
                                         engine="pyarrow")
count_not_reconstructed = reconstructed_features.loc[(
        (reconstructed_features.lod_12 == 0) & (
        reconstructed_features.lod_13 == 0) & (
                reconstructed_features.lod_13 == 0) & (
                reconstructed_features.lod_0 == 0)), "identificatie"].nunique()
log.info(f"Loaded {reconstructed_features_parquet}")

## Output formats

df_val3dity_params = pd.DataFrame.from_records([
    {
        "parameter": "[planarity_d2p_tol](https://val3dity.readthedocs.io/en/latest/usage/#planarity-d2p-tol)",
        "description": "Tolerance for planarity based on a distance to a plane",
        "value": 0.001},
    {
        "parameter": "[planarity_n_tol](https://val3dity.readthedocs.io/en/latest/usage/#planarity-n-tol)",
        "description": "Tolerance for planarity based on normals deviation",
        "value": 20.0},
])
df_val3dity_params["parameter"] = df_val3dity_params.parameter.astype(str)

invalid_cj = validated_compressed.cj_nr_invalid.sum()
total_cj = validated_compressed.cj_nr_features.sum()
total_obj = validated_compressed.obj_nr_features.sum()
invalid_obj = sum(itertools.chain.from_iterable(
    eval(e) for e in validated_compressed.obj_nr_invalid if
    isinstance(e, str) and e != "[]"))
total_gpkg = validated_compressed.gpkg_nr_features.sum()
df_format_counts = pd.DataFrame.from_records([
    {"format": "CityJSON",
     "object count": total_cj,
     "invalid geometry": f"{invalid_cj} ({round(invalid_cj / total_cj * 100, 2)}%)"},
    {"format": "OBJ",
     "object count": total_obj,
     "invalid geometry": f"{invalid_obj} ({round(invalid_obj / total_obj * 100, 2)}%)"},
    {"format": "GPKG",
     "object count": total_gpkg,
     "invalid geometry": "-"},
])

# --- Plots

log.info("Preparing dataframes for the plots")
PIE_HOLE = 0.4
rf_pw_bron_counts = reconstructed_features.b3_pw_bron.value_counts(dropna=False)
rf_pw_selectie_counts = reconstructed_features.b3_pw_selectie_reden.value_counts(
    dropna=False)
rf_pw_fraction = pd.DataFrame({
    'AHN3': reconstructed_features[reconstructed_features[
        "b3_nodata_fractie_ahn3"].notna()].b3_nodata_fractie_ahn3,
    'AHN4': reconstructed_features[
        reconstructed_features["b3_nodata_fractie_ahn4"].notna()].b3_nodata_fractie_ahn4
}).melt(var_name="pc_source", value_name="fraction")
rf_pw_radius = pd.DataFrame({
    'AHN3': reconstructed_features[
        reconstructed_features["b3_nodata_radius_ahn3"].notna()].b3_nodata_radius_ahn3,
    'AHN4': reconstructed_features[
        reconstructed_features["b3_nodata_radius_ahn4"].notna()].b3_nodata_radius_ahn4}
).melt(var_name="pc_source", value_name="fraction")
rf_pw_density = pd.DataFrame({
    'AHN3': reconstructed_features[
        reconstructed_features["b3_puntdichtheid_ahn3"].notna()].b3_puntdichtheid_ahn3,
    'AHN4': reconstructed_features[
        reconstructed_features["b3_puntdichtheid_ahn4"].notna()].b3_puntdichtheid_ahn4}
).melt(var_name="pc_source", value_name="fraction")
del reconstructed_features
log.info("End Preparing dataframes for the plots")


def plot_pc_version(rf_pw_bron_counts):
    return go.Figure(data=[go.Pie(labels=rf_pw_bron_counts.index,
                                  values=rf_pw_bron_counts.values, hole=PIE_HOLE)])


def plot_pc_selection_reason(rf_pw_selectie_counts):
    return go.Figure(data=[go.Pie(labels=rf_pw_selectie_counts.index,
                                  values=rf_pw_selectie_counts.values, hole=PIE_HOLE)])


def plot_pc_nodata_fraction(rf_pw_fraction):
    buf = io.BytesIO()
    p = sns.displot(rf_pw_fraction, x="fraction", hue="pc_source", binwidth=0.05,
                    stat="probability", common_norm=False)
    p.set_axis_labels("Fraction of the footprint")
    p.legend.set_title("Point cloud")
    plt.savefig(buf, format="svg")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    buf.close()
    return "data:image/svg+xml;base64,{}".format(data)


def plot_pc_nodata_radius(rf_pw_radius):
    buf = io.BytesIO()
    x_range = (0, 5)
    p = sns.displot(rf_pw_radius, x="fraction", hue="pc_source", binwidth=0.1,
                    stat="probability", common_norm=False)
    p.set_axis_labels(f"Radius [m]. Range limited to {x_range}")
    p.legend.set_title("Point cloud")
    plt.xlim(*x_range)
    plt.savefig(buf, format="svg")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    buf.close()
    return "data:image/svg+xml;base64,{}".format(data)


def plot_pc_density(rf_pw_density):
    buf = io.BytesIO()
    x_range = (0, 100)
    p = sns.displot(rf_pw_density, x="fraction", hue="pc_source", binwidth=5,
                    stat="probability", common_norm=False)
    p.set_axis_labels(f"Point density [pt/m2]. Range limited to {x_range}")
    p.legend.set_title("Point cloud")
    plt.xlim(*x_range)
    plt.savefig(buf, format="svg")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    buf.close()
    return "data:image/svg+xml;base64,{}".format(data)


def plot_validity_cityjson(validated_compressed):
    _ae = itertools.chain.from_iterable(
        eval(e) for e in validated_compressed.cj_all_errors if e != "[]")
    all_errors = list(_ae)
    fig = go.Figure(data=[go.Pie(labels=all_errors, hole=PIE_HOLE)])
    fig.update_layout(
        annotations=[dict(text='CityJSON', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig


def plot_validity_obj(validated_compressed):
    _ae = itertools.chain.from_iterable(
        eval(e) for e in validated_compressed.obj_all_errors if
        isinstance(e, str) and e != "[]")
    all_errors = list(_ae)
    fig = go.Figure(data=[go.Pie(labels=all_errors, values=all_errors, hole=PIE_HOLE)])
    fig.update_layout(
        annotations=[dict(text='OBJ', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig


log.info("Plotting figures")
fig_pc_version = plot_pc_version(rf_pw_bron_counts)
fig_pc_selection_reason = plot_pc_selection_reason(rf_pw_selectie_counts)
fig_pc_nodata_fraction = plot_pc_nodata_fraction(rf_pw_fraction)
fig_pc_nodata_radius = plot_pc_nodata_radius(rf_pw_radius)
fig_pc_density = plot_pc_density(rf_pw_density)

# --- Layout
log.info("Building layout")

card_pc_version = dbc.Card(
    dbc.CardBody([
        html.H4("Point cloud source"),
        html.P(["Attribute: ", html.A("b3_pw_bron",
                                      href="https://docs.3dbag.nl/en/schema/attributes/#b3_pw_bron")]),
        dcc.Graph(figure=fig_pc_version)
    ])
)

card_pc_selection = dbc.Card(
    dbc.CardBody([
        html.H4("Point cloud selection"),
        html.P(["Reason for the selected point cloud source. Attribute: ",
                html.A("b3_pw_selectie_reden",
                       href="https://docs.3dbag.nl/en/schema/attributes/#b3_pw_selectie_reden")]),
        dcc.Graph(figure=fig_pc_selection_reason)
    ])
)

card_pc_nodata_fraction = dbc.Card(
    dbc.CardBody([
        html.H4("No-data fraction"),
        html.P([
            "Fraction of the footprint area that has no point data in the AHN point cloud. Only points classified as building or ground are considered. Attribute: ",
            html.A("b3_nodata_fractie_*",
                   href="https://docs.3dbag.nl/en/schema/attributes/#b3_nodata_fractie_ahn3")
        ]),
        dbc.CardImg(src=fig_pc_nodata_fraction),
    ])
)

card_pc_nodata_radius = dbc.Card(
    dbc.CardBody([
        html.H4("No-data radius"),
        html.P([
            "Radius of the largest circle inside the BAG polygon without any AHN points. Only points classified as building or ground are considered. Attribute: ",
            html.A("b3_nodata_radius_*",
                   href="https://docs.3dbag.nl/en/schema/attributes/#b3_nodata_radius_ahn3")
        ]),
        dbc.CardImg(src=fig_pc_nodata_radius)
    ])
)

card_pc_density = dbc.Card(
    dbc.CardBody([
        html.H4("Point density"),
        html.P([
            "Density of the AHN point cloud on BAG polygon. Only points classified as building or ground are considered. Attribute: ",
            html.A("b3_puntdichtheid_*",
                   href="https://docs.3dbag.nl/en/schema/attributes/#b3_puntdichtheid_ahn3")
        ]),
        dbc.CardImg(src=fig_pc_density)
    ])
)

card_validity_cityjson = dbc.Card(
    dbc.CardBody([
        html.H4("CityJSON geometric errors"),
        dcc.Graph(figure=plot_validity_cityjson(validated_compressed))
    ])
)

card_validity_obj = dbc.Card(
    dbc.CardBody([
        html.H4("OBJ geometric errors"),
        dcc.Graph(figure=plot_validity_obj(validated_compressed))
    ])
)

log.info("Creating app layout")
app.layout = dbc.Container([
    html.H3(f"3DBAG version {metadata['identificationInfo']['citation']['edition']}",
            className="title is-3"),
    html.H3("Software versions", className="title is-3"),
    dbc.Row([
        dash_table.DataTable(data=df_software.to_dict('records'),
                             page_size=len(df_software),
                             style_cell={'whiteSpace': 'pre-line', 'textAlign': 'left'},
                             columns=[
                                 {"name": "name", "id": "name"},
                                 {"name": "description", "id": "description"},
                                 {"name": "repository", "id": "repository",
                                  "presentation": "markdown"},
                                 {"name": "version", "id": "version"}
                             ]),
    ], align="center", className="table"),
    html.H3("Input processing", className="title is-3"),
    dbc.Row([
        dash_table.DataTable(data=df_steps.to_dict('records'),
                             page_size=len(df_steps),
                             style_cell={'textAlign': 'left'},
                             ),
    ], align="center", className="table"),
    html.H4("Point clouds", className="title is-4"),
    html.Div([
        dbc.Row([
            dbc.Col(card_pc_version),
            dbc.Col(card_pc_selection),
        ]),
        dbc.Row([
            dbc.Col(card_pc_nodata_fraction),
            dbc.Col(card_pc_nodata_radius),
            dbc.Col(card_pc_density),
        ], style={'display': 'flex', 'flexDirection': 'row'}),
    ]),
    html.H3("Reconstruction", className="title is-3"),
    html.P(
        f"Reconstruction rate: {100.0 - round(count_not_reconstructed / count_reconstruction_input * 100, 1)}%"),
    html.P(
        f"Number failed: {count_not_reconstructed} (out of {count_reconstruction_input})"),
    html.H3("Output formats", className="title is-3"),
    dbc.Row([
        html.H4("CityJSON", className="title is-4"),
        dcc.Markdown(f"""
        - Tile IDs with an invalid ZIP file: {', '.join(validated_compressed.loc[validated_compressed["cj_zip_ok"] == False, "tile_id"].sort_values(ascending=True))}
        - Tile IDs with an invalid CityJSON schema: {', '.join(validated_compressed.loc[validated_compressed["cj_schema_valid"] == False, "tile_id"].sort_values(ascending=True))}
        - Tile IDs that do not contain all LoDs (0, 1.2, 1.3, 2.2): {', '.join(validated_compressed.loc[validated_compressed["cj_lod"] != "['0', '1.2', '1.3', '2.2']", "tile_id"].sort_values(ascending=True))}
        """, className="content"),

        html.H4("OBJ", className="title is-4"),
        dcc.Markdown(f"""
        - Tile IDs with an invalid ZIP file: {', '.join(validated_compressed.loc[validated_compressed["obj_zip_ok"] == False, "tile_id"].sort_values(ascending=True))}
        """, className="content"),

        html.H4("GeoPackage", className="title is-4"),
        dcc.Markdown(f"""
        - Tile IDs with an invalid ZIP file: {', '.join(validated_compressed.loc[validated_compressed["gpkg_zip_ok"] == False, "tile_id"].sort_values(ascending=True))}
        - Tile IDs with invalid GeoPackage: {', '.join(validated_compressed.loc[validated_compressed["gpkg_ok"] == False, "tile_id"].sort_values(ascending=True))}
        """, className="content"),
    ]),
    html.P("", className="content"),
    html.H3("Geometric errors", className="title is-3"),
    dbc.Row([
        dash_table.DataTable(data=df_format_counts.to_dict('records'),
                             page_size=len(df_format_counts),
                             style_cell={'whiteSpace': 'pre-line',
                                         'textAlign': 'left'}),
    ], align="center", className="table"),
    dcc.Markdown(
        "Distribution of geometric errors. For the meaning of error codes see the [val3dity errors](https://val3dity.readthedocs.io/en/latest/errors/)."),
    dbc.Row([dash_table.DataTable(data=df_val3dity_params.to_dict('records'),
                                  style_cell={'whiteSpace': 'pre-line',
                                              'textAlign': 'left'},
                                  columns=[
                                      {"name": "parameter", "id": "parameter",
                                       "presentation": "markdown"},
                                      {"name": "description", "id": "description"},
                                      {"name": "value", "id": "value"}
                                  ])
             ], align="center", className="table"),
    dbc.Row([
        dbc.Col(card_validity_cityjson),
        dbc.Col(card_validity_obj),
    ]),
], fluid=True, className="section content")

log.info("Running the dashboard")
