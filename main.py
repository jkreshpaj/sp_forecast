import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────────
LAT, LON = 40.7948342, 19.4022414
APIKEY = '6poS7GKcuWLzClbE'
TZ = 'Europe/Tirane'
TILT = 20            # degrees
AZIMUTH = 180        # south
SYS_CAPACITY = 62.2  # MW
REF_TEMP = 25        # °C
LOSS_COEFF = -0.0026 # per °C
SYSTEM_LOSSES = 0.08 # 8% system losses
DATA_PATH = 'output.csv'
# ── END CONFIG ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Solar Forecast Dashboard", layout="wide")

# Force light theme background
st.markdown(
    """
    <style>
        .reportview-container .main, .css-18e3th9 {
            background-color: white !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Load data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.reset_index().rename(columns={'index': 'time'})
    return df

df = load_data(DATA_PATH)

# 1) Physics-based Power chart
st.subheader("Physics-based Power Output (MW)")
fig1 = px.line(
    df, x='time', y='power_physics',
    labels={'time': 'Time', 'power_physics': 'Power (MW)'},
    title='Power Forecast',
    template='plotly_white'
)
fig1.update_traces(
    line=dict(color='white'),
    hovertemplate='Time: %{x}<br>Power: %{y:.2f} MW'
)
fig1.update_layout(hovermode='x unified', showlegend=False)
st.plotly_chart(fig1, use_container_width=True)
st.markdown("---")

# 2) Dual-axis chart for GHI vs other variables
def plot_dual_axis(df, options):
    fig = go.Figure()
    # Primary y-axis: GHI fill
    if 'ghi' in options:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['ghi'], name='GHI', fill='tozeroy', mode='none', fillcolor='yellow',
            hovertemplate='Time: %{x}<br>GHI: %{y:.1f} W/m²'
        ))
    # Secondary y-axis: other variables
    color_map = {'module_temperature':'blue','temperature':'red','humidity':'green','wind_speed':'purple','pressure':'brown'}
    for var in options:
        if var == 'ghi': continue
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[var], name=var.replace('_',' ').title(),
            mode='lines', line=dict(color=color_map.get(var, 'gray')),
            yaxis='y2',
            hovertemplate=f'Time: %{{x}}<br>{var.replace("_"," ").title()}: %{{y:.2f}}'
        ))
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(title='Time'),
        yaxis=dict(title='GHI (W/m²)', showgrid=True),
        yaxis2=dict(title='Other Variables', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    return fig

# Chart controls
options = st.multiselect(
    "Select variables to plot:",
    ['ghi', 'module_temperature', 'temperature', 'humidity', 'wind_speed', 'pressure'],
    default=['ghi', 'module_temperature']
)

if options:
    st.subheader("Irradiance & Other Variables Over Time")
    fig2 = plot_dual_axis(df, options)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

# Raw data table
if st.checkbox("Show raw data table"):
    st.subheader("Raw Data")
    st.dataframe(df)

# Configuration on main page
st.header("Configuration")
st.write(f"**Latitude:** {LAT}")
st.write(f"**Longitude:** {LON}")
st.write(f"**Timezone:** {TZ}")
st.write(f"**Tilt (°):** {TILT}")
st.write(f"**Azimuth (°):** {AZIMUTH}")
st.write(f"**System Capacity (MW):** {SYS_CAPACITY}")
st.write(f"**Reference Temp (°C):** {REF_TEMP}")
st.write(f"**Loss Coeff:** {LOSS_COEFF}")
st.write(f"**System Losses:** {SYSTEM_LOSSES}")
st.markdown("---")
