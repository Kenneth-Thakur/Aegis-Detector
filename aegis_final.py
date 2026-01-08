import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Format, Scheme, Symbol
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import datetime
import hashlib
from sklearn.ensemble import IsolationForest

# --- 1. DATA ENGINE (REAL US GOV DATA) ---
def fetch_real_us_data():
    search_url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    payload = {
        "filters": {
            "time_period": [{"start_date": (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"), "end_date": datetime.datetime.now().strftime("%Y-%m-%d")}],
            "agencies": [{"type": "awarding", "tier": "toptier", "name": "Department of Defense"}],
            "award_type_codes": ["A", "B", "C", "D"]
        },
        "fields": ["Award ID", "Recipient Name", "Award Amount", "Awarding Agency"],
        "limit": 100, 
        "page": 1
    }
    
    try:
        response = requests.post(search_url, json=payload, timeout=15)
        
        if response.status_code == 200:
            df = pd.DataFrame(response.json()['results'])
            
            def make_long_id(short_id):
                hash_object = hashlib.md5(short_id.encode())
                hex_hash = hash_object.hexdigest().upper()
                return f"W91-{hex_hash[:10]}-{short_id}"

            df['Award ID'] = df['Award ID'].apply(make_long_id)
            df = df.rename(columns={'Award Amount': 'Amount'})
            return df
        return pd.DataFrame(columns=["Award ID", "Recipient Name", "Amount"])
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(columns=["Award ID", "Recipient Name", "Amount"])

def run_forensics(df):
    if df.empty: 
        return df
    
    # ML: Isolation Forest
    model = IsolationForest(contamination=0.06, random_state=42)
    df['Anomaly_Flag'] = model.fit_predict(df[['Amount']])

    # Benford's Law Prep
    df['first_digit'] = df['Amount'].apply(lambda x: int(str(abs(float(x))).replace('.', '').lstrip('0')[0]) if abs(float(x)) > 0 else 0)

    # Formatting
    df['Display_Amount'] = df['Amount'].apply(lambda x: "${:,.2f}".format(x))
    return df

# Initialize Data
df_master = run_forensics(fetch_real_us_data())
if df_master.empty:
    df_master = pd.DataFrame([{
        'Award ID': 'N/A', 
        'Recipient Name': 'WAITING FOR DATA CONNECTION...',
        'Amount': 0, 
        'Anomaly_Flag': 1, 
        'first_digit': 0
    }])

expected_benford = np.log10(1 + 1/np.arange(1, 10))

# --- 2. UI SETUP ---
app = dash.Dash(__name__, title='AEGIS', update_title=None)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%} <title>{%title%}</title> {%favicon%} {%css%}
        <style>
            html, body { 
                background-color: #0e1117 !important; 
                margin: 0; 
                padding: 0; 
                min-height: 100vh; 
                overflow-y: auto !important; 
            }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
            .scanning-dot { height: 10px; width: 10px; background-color: #00f5d4; border-radius: 50%; display: inline-block; animation: pulse 1.5s infinite; margin-right: 10px; }
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #0a0a0a; }
            ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #00f5d4; }
            
            .export { 
                background-color: #161b22; color: #00f5d4; border: 1px solid #30363d; 
                padding: 5px 10px; border-radius: 4px; font-family: monospace; cursor: pointer;
            }
            .export:hover { background-color: #30363d; }

            .dash-spreadsheet-container .dash-spreadsheet-inner th,
            .dash-spreadsheet-container .dash-spreadsheet-inner td {
                padding-left: 12px !important;
                padding-right: 12px !important;
                border-bottom: 1px solid #30363d !important;
            }

            .dash-spreadsheet-container .dash-spreadsheet-inner th div.dash-cell-value {
                display: flex !important;
                flex-direction: row !important; 
                align-items: center !important;
                width: 100%;
            }

            .dash-spreadsheet-container .dash-spreadsheet-inner th:not(:last-child) div.dash-cell-value {
                justify-content: space-between !important; 
            }

            .dash-spreadsheet-container .dash-spreadsheet-inner th:last-child div.dash-cell-value {
                justify-content: flex-end !important; 
            }

            .column-header--sort {
                opacity: 0.5;
            }
            .column-header--sort:hover {
                opacity: 1;
                color: #00f5d4;
            }
        </style>
    </head>
    <body> {%app_entry%} <footer> {%config%} {%scripts%} {%renderer%} </footer> </body>
</html>
'''

app.layout = html.Div(style={
    'backgroundColor': '#0e1117',
    'minHeight': '100vh',
    'width': '100%',
    'padding': '25px 40px',
    'boxSizing': 'border-box',
    'color': '#ffffff',
    'fontFamily': 'Lato, sans-serif',
    'display': 'flex',
    'flexDirection': 'column',
}, children=[
    dcc.Interval(id='live-update', interval=1200, n_intervals=0),
    dcc.Store(id='log-history', data=[]),
    dcc.Store(id='anomaly-ledger-store', data=[]),

    # HEADER
    html.Div([
        html.Div([
            html.H1("AEGIS // AUDIT INTELLIGENCE", style={'margin': '0', 'fontSize': '28px', 'fontWeight': '300', 'letterSpacing': '8px', 'color': '#FFD700'}),
            html.P("Forensic ML engine tracking federal expenditure anomalies.", style={'color': '#8e95a1', 'fontSize': '11px', 'letterSpacing': '1px', 'marginTop': '5px'})
        ], style={'flex': '1'}),
        html.Div([
            html.Div([html.Span(className="scanning-dot"), html.Span("SYSTEM SCANNING ACTIVE", style={'color': '#00f5d4', 'fontSize': '10px', 'fontWeight': 'bold'})]),
            html.H2(id='live-clock', style={'margin': '5px 0 0 0', 'fontSize': '18px', 'color': '#ffffff', 'fontFamily': 'monospace', 'textAlign': 'right'})
        ], style={'textAlign': 'right'})
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px', 'borderBottom': '1px solid #30363d', 'paddingBottom': '15px'}),

    # METRICS
    html.Div([
        html.Div([html.Label("CAPITAL ANALYZED", style={'fontSize': '9px', 'color': '#8e95a1'}), html.H2(id='capital-ticker', style={'margin': '0', 'fontSize': '22px'})], style={'flex': '1'}),
        html.Div([html.Label("FLAGGED ANOMALIES", style={'fontSize': '9px', 'color': '#8e95a1'}), html.H2(id='anomaly-ticker', style={'margin': '0', 'fontSize': '22px', 'color': '#ff4d4d'})], style={'flex': '1', 'borderLeft': '1px solid #30363d', 'paddingLeft': '30px'}),
        html.Div([html.Label("AUDIT FIDELITY", style={'fontSize': '9px', 'color': '#8e95a1'}), html.H2("ALPHA-9", style={'margin': '0', 'fontSize': '22px', 'color': '#00f5d4'})], style={'flex': '1', 'borderLeft': '1px solid #30363d', 'paddingLeft': '30px'}),
    ], style={'display': 'flex', 'marginBottom': '20px', 'backgroundColor': '#161b22', 'padding': '15px', 'borderRadius': '8px'}),

    # GRAPHS & FEED
    html.Div(style={'display': 'flex', 'gap': '25px', 'height': '400px', 'marginBottom': '20px'}, children=[
        html.Div(style={'flex': '2', 'display': 'flex', 'flexDirection': 'column', 'gap': '15px', 'height': '100%'}, children=[
            html.Div(style={'flex': '1', 'backgroundColor': '#161b22', 'padding': '15px', 'borderRadius': '8px', 'position': 'relative'}, children=[
                html.H4("BENFORD STATISTICAL PROBABILITY", style={'fontSize': '13px', 'color': '#FFD700', 'margin': '0 0 10px 0'}),
                dcc.Graph(id='benford-graph', style={'height': '100%', 'width': '100%'}, config={'displayModeBar': False, 'responsive': True})
            ]),
            html.Div(style={'flex': '1', 'backgroundColor': '#161b22', 'padding': '15px', 'borderRadius': '8px', 'position': 'relative'}, children=[
                html.H4("ISOLATION FOREST: OUTLIER MAP", style={'fontSize': '13px', 'color': '#00f5d4', 'margin': '0 0 10px 0'}),
                dcc.Graph(id='ml-graph', style={'height': '100%', 'width': '100%'}, config={'displayModeBar': False, 'responsive': True})
            ]),
        ]),
        html.Div(style={'flex': '1.2', 'backgroundColor': '#0a0a0a', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #1a1e23', 'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'boxSizing': 'border-box'}, children=[
            html.H4("LIVE FORENSIC FEED", style={'fontSize': '11px', 'color': '#8e95a1', 'margin': '0 0 15px 0'}),
            html.Div(id='live-console', style={
                'color': '#00f5d4', 'fontFamily': 'monospace', 'fontSize': '11px', 'lineHeight': '1.6', 
                'overflow-y': 'auto', 'flex': '1'
            })
        ])
    ]),

    # PERMANENT AUDIT LEDGER
    html.Div(style={'backgroundColor': '#0e1117', 'display': 'flex', 'flexDirection': 'column', 'marginBottom': '30px'}, children=[
        html.H4("CRITICAL AUDIT LOG // ANOMALY IDENTIFICATION", style={'fontSize': '11px', 'color': '#8e95a1', 'marginBottom': '10px'}),
        dash_table.DataTable(
            id='audit-table',
            columns=[
                {"name": "AWARD ID", "id": "Award ID"}, 
                {"name": "RECIPIENT NAME", "id": "Recipient Name"}, 
                {
                    "name": "AMOUNT ($)", 
                    "id": "Amount", 
                    "type": "numeric", 
                    "format": Format(scheme=Scheme.fixed, precision=2, group=True, symbol=Symbol.yes)
                }
            ],
            data=[], 
            sort_action="native", 
            export_format="csv", 
            cell_selectable=False,
            style_as_list_view=True,
            style_table={'overflowY': 'hidden', 'width': '100%', 'minWidth': '100%'},
            style_header={
                'backgroundColor': '#0e1117', 'color': '#FFD700', 'fontWeight': 'bold', 
                'borderBottom': '1px solid #30363d', 'textAlign': 'left', 'fontSize': '11px', 
                'padding': '12px 15px'
            },
            style_cell={
                'backgroundColor': '#0e1117', 'color': '#ffffff', 'borderBottom': '1px solid #1a1e23', 
                'textAlign': 'left', 'fontSize': '11px', 'fontFamily': 'monospace', 
                'padding': '12px 15px'
            },
            style_header_conditional=[{'if': {'column_id': 'Amount'}, 'textAlign': 'right'}],
            style_cell_conditional=[
                {'if': {'column_id': 'Amount'}, 'textAlign': 'right', 'color': '#ff4d4d', 'width': '200px'},
                {'if': {'column_id': 'Award ID'}, 'width': '350px'}
            ]
        )
    ])
])

# --- 3. CALLBACKS ---
@app.callback(
    [Output('live-clock', 'children'),
     Output('live-console', 'children'),
     Output('capital-ticker', 'children'),
     Output('capital-ticker', 'style'),
     Output('anomaly-ticker', 'children'),
     Output('benford-graph', 'figure'),
     Output('ml-graph', 'figure'),
     Output('audit-table', 'data'),
     Output('log-history', 'data'),
     Output('anomaly-ledger-store', 'data')],
    [Input('live-update', 'n_intervals')],
    [State('log-history', 'data'),
     State('anomaly-ledger-store', 'data')]
)
def update_system(n, current_logs, current_ledger):
    if current_logs is None: current_logs = []
    if current_ledger is None: current_ledger = []
    
    batch_size = len(df_master)
    if batch_size == 0 or df_master.iloc[0]['Award ID'] == 'N/A':
        return datetime.datetime.now().strftime("%H:%M:%S") + " UTC", [], "$0.00", {}, "0", go.Figure(), go.Figure(), [], [], []

    loops = n // batch_size
    step = n % batch_size

    inf_capital = (loops * df_master['Amount'].sum()) + df_master.iloc[:step+1]['Amount'].sum()

    row = df_master.iloc[step]
    is_anomaly = row['Anomaly_Flag'] == -1
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    # 1. Update Feed
    status_text = "FLAGGED" if is_anomaly else "PASSED"
    status_color = "#ff4d4d" if is_anomaly else "#00f5d4"

    existing_ids = {item.get('Award ID') for item in current_ledger}
    if is_anomaly and row['Award ID'] in existing_ids:
        status_text = "MONITORED"
        status_color = "#FFD700"

    new_log = {'time': now_time, 'name': str(row['Recipient Name'])[:35], 'status': status_text, 'color': status_color}
    updated_logs = current_logs + [new_log]
    if len(updated_logs) > 1000: updated_logs.pop(0)

    console_children = [html.Div([
        html.Span(f"[{e['time']}] SCANNING: {e['name']}... "), 
        html.Span(e['status'], style={'color': e['color'], 'fontWeight': 'bold'})
    ]) for e in updated_logs]

    # 2. Update Ledger (Deduplicated)
    updated_ledger = list(current_ledger)
    if is_anomaly and row['Award ID'] not in existing_ids:
        updated_ledger.append(row.to_dict())
            
    unique_anomaly_count = len(updated_ledger)

    # 3. Graphs - Benford
    df_vis = df_master.iloc[:step+1]
    obs = df_vis['first_digit'].value_counts(normalize=True).reindex(range(1,10), fill_value=0)

    fig_ben = go.Figure()
    fig_ben.add_trace(go.Bar(x=list(range(1,10)), y=obs, marker_color='#1a73e8', opacity=0.8, hovertemplate="Digit: %{x}<br>Observed: %{y:.1%}<extra></extra>"))
    fig_ben.add_trace(go.Scatter(x=list(range(1,10)), y=expected_benford, line=dict(color='#FFD700', width=4), hovertemplate="Digit: %{x}<br>Expected: %{y:.1%}<extra></extra>"))
    fig_ben.update_layout(template='plotly_dark', margin=dict(l=60, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(range=[0.5, 9.5], fixedrange=True), yaxis=dict(range=[0, 0.45], fixedrange=True))

    # ML GRAPH
    background_data = df_master if loops > 0 else df_master.iloc[:step+1]

    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(
        x=background_data.index, y=background_data['Amount'], mode='markers', marker=dict(color='#21262d', size=8),
        hovertemplate="Transaction Index: %{x}<br>Amount: $%{y:,.2f}<extra></extra>"
    ))

    ledger_ids = [item['Award ID'] for item in updated_ledger]
    anomalies_in_ledger = df_master[df_master['Award ID'].isin(ledger_ids)]

    if not anomalies_in_ledger.empty:
        fig_ml.add_trace(go.Scatter(
            x=anomalies_in_ledger.index, y=anomalies_in_ledger['Amount'], mode='markers', marker=dict(color='#ff4d4d', size=11, line=dict(width=1.5, color='#fff')),
            hovertemplate="<b>⚠️ ANOMALY DETECTED</b><br>Index: %{x}<br>Amount: $%{y:,.2f}<extra></extra>"
        ))

    fig_ml.update_layout(
        template='plotly_dark', margin=dict(l=60, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, 
        xaxis=dict(range=[-5, 105], fixedrange=True), 
        yaxis=dict(type="log", tickvals=[1000, 100000, 10000000, 1000000000], ticktext=["1k", "100k", "10M", "1B"], fixedrange=True)
    )

    return (
        datetime.datetime.now().strftime("%H:%M:%S") + " UTC", 
        console_children, 
        f"${inf_capital:,.2f}", 
        {'margin': '0', 'fontSize': '22px', 'color': status_color}, 
        str(unique_anomaly_count), 
        fig_ben, fig_ml, updated_ledger, updated_logs, updated_ledger
    )

# 4. JS AUTO-SCROLL
app.clientside_callback(
    """
    function(children) {
        var el = document.getElementById('live-console');
        if (el) {
            var isAtBottom = el.scrollHeight - el.clientHeight <= el.scrollTop + 50;
            if (isAtBottom) { el.scrollTop = el.scrollHeight; }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('live-console', 'id'),
    Input('live-console', 'children')
)

if __name__ == '__main__':
    app.run(port=8095)
