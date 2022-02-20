import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("PoC - Tesi - Mario Alessandro Napoli")

@st.cache
def get_data(filepath):
    data = pd.read_csv(filepath)
    data.elevation = data.elevation.astype(float)
    data.timestamp = pd.to_datetime(data.timestamp)
    return data

data = get_data("amrc_extracted_data.csv")

viz1 = data.groupby("station").agg({"latitude": "mean", "longitude": "mean", "elevation": "first"}).reset_index()
fig1 = go.Figure(go.Scattergeo(lat=viz1.latitude, lon=viz1.longitude, hovertext=viz1.station))
fig1.update_geos(projection={"type": "stereographic", "rotation": {"lat": -90, "lon": 0}, "scale": 4})
fig1.update_layout(height=600, margin={"r": 0, "t": 0, "l": 0, "b": 0})

st.plotly_chart(fig1)

station_name = st.selectbox("Select a Station", options=data.station.unique())

viz2 = data[data.station == station_name]

viz2 = viz2.groupby(viz2.timestamp.dt.date).agg({'pressure': 'mean', 'temperature': 'mean', 'wind_speed': 'mean'}).reset_index()
fig2 = px.line(viz2, x="timestamp", y=viz2.columns[1:],
              hover_data={"timestamp": "|%B %d, %Y"})
fig2.update_layout(width=1600)#, margin={"r": 0, "t": 0, "l": 0, "b": 0})

st.plotly_chart(fig2)


#
# import pydeck as pdk
# import pandas as pd
#
# SCREEN_GRID_LAYER_DATA = (
#     "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/sf-bike-parking.json"  # noqa
# )
# df = pd.read_json(SCREEN_GRID_LAYER_DATA)
#
# # Define a layer to display on a map
# layer = pdk.Layer(
#     "ScreenGridLayer",
#     df,
#     pickable=False,
#     opacity=0.5,
#     cell_size_pixels=10,
#     color_range=[
#         [0, 25, 0, 25],
#         [0, 85, 0, 85],
#         [0, 127, 0, 127],
#         [0, 170, 0, 170],
#         [0, 190, 0, 190],
#         [0, 255, 0, 255],
#     ],
#     get_position="COORDINATES",
#     get_weight="SPACES",
# )
#
# # Set the viewport location
# view_state = pdk.ViewState(latitude=37.7749295, longitude=-122.4194155, zoom=11, bearing=0, pitch=0)
#
# # Render
# r = pdk.Deck(layers=[layer], initial_view_state=view_state)
#
# st.pydeck_chart(r)