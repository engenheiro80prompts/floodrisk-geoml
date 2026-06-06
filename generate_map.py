import pandas as pd
import folium
from folium.plugins import HeatMap, Fullscreen
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "processed" / "dataset_completo_com_score.csv"
OUT  = ROOT / "dashboard" / "mapa_risco.html"
OUT.parent.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

mapa = folium.Map(location=[-23.62, -46.54], zoom_start=10, tiles=None)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attr="CartoDB", name="Dark Matter"
).add_to(mapa)
Fullscreen().add_to(mapa)

def cor_risco(p):
    if p >= 0.80: return "#d73027"
    elif p >= 0.65: return "#fc8d59"
    elif p >= 0.50: return "#fee08b"
    elif p >= 0.35: return "#d9ef8b"
    else: return "#1a9850"

layer = folium.FeatureGroup(name="Setores Censitários", show=True)
for _, row in df.iterrows():
    prob = row["prob_risco_alto"]
    cor  = cor_risco(prob)
    popup = f"""<div style="font-family:Arial;font-size:12px;min-width:200px">
      <b>{row['bairro']}</b><br><span style="color:#888">{row['municipio']}</span><hr>
      Prob. Risco Alto: <b style="color:{cor}">{prob:.1%}</b><br>
      Ocorrências (5 anos): {int(row['ocorrencias_alagamento_5anos'])}<br>
      Declividade: {row['declividade_graus']:.1f}°<br>
      Dist. drenagem: {row['dist_drenagem_m']:.0f} m
    </div>"""
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=max(4, prob*10), color=cor, fill=True,
        fill_color=cor, fill_opacity=0.65, weight=0.8,
        popup=folium.Popup(popup, max_width=260),
        tooltip=f"{row['bairro']} | Risco: {prob:.0%}"
    ).add_to(layer)
layer.add_to(mapa)

heat_data = df[["latitude","longitude","prob_risco_alto"]].values.tolist()
layer_heat = folium.FeatureGroup(name="Mapa de Calor", show=False)
HeatMap(heat_data, min_opacity=0.3, radius=18, blur=15,
        gradient={"0.4":"#1a9850","0.65":"#fee08b","0.80":"#fc8d59","1.0":"#d73027"}
).add_to(layer_heat)
layer_heat.add_to(mapa)

criticos = (df["prob_risco_alto"] >= 0.80).sum()
altos    = ((df["prob_risco_alto"] >= 0.65) & (df["prob_risco_alto"] < 0.80)).sum()

legenda = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
background:rgba(13,17,23,0.92);border:1px solid #30363d;border-radius:8px;
padding:14px 18px;font-family:Arial;color:#e6edf3;min-width:185px">
<div style="font-weight:bold;font-size:13px;margin-bottom:10px;color:#58a6ff">Nível de Risco</div>
<div><span style="background:#d73027;width:14px;height:14px;border-radius:50%;display:inline-block;margin-right:8px"></span>Crítico >= 80%</div>
<div><span style="background:#fc8d59;width:14px;height:14px;border-radius:50%;display:inline-block;margin-right:8px"></span>Alto 65-80%</div>
<div><span style="background:#fee08b;width:14px;height:14px;border-radius:50%;display:inline-block;margin-right:8px"></span>Moderado 50-65%</div>
<div><span style="background:#1a9850;width:14px;height:14px;border-radius:50%;display:inline-block;margin-right:8px"></span>Baixo < 50%</div>
<hr style="border-color:#30363d;margin:10px 0">
<div style="font-size:10px;color:#8b949e">Random Forest | 500 setores | RMSP</div>
</div>"""
mapa.get_root().html.add_child(folium.Element(legenda))

stats = f"""<div style="position:fixed;top:20px;right:20px;z-index:9999;
background:rgba(13,17,23,0.92);border:1px solid #30363d;border-radius:8px;
padding:14px 18px;font-family:Arial;color:#e6edf3;min-width:200px">
<div style="font-weight:bold;font-size:13px;margin-bottom:10px;color:#58a6ff">Resumo RMSP</div>
<div style="display:flex;justify-content:space-between"><span style="color:#d73027">Crítico</span><b>{criticos} setores</b></div>
<div style="display:flex;justify-content:space-between"><span style="color:#fc8d59">Alto</span><b>{altos} setores</b></div>
<hr style="border-color:#30363d;margin:8px 0">
<div style="font-size:10px;color:#8b949e">FloodRisk GeoML v1.0</div>
</div>"""
mapa.get_root().html.add_child(folium.Element(stats))

folium.LayerControl(collapsed=False).add_to(mapa)
mapa.save(str(OUT))
print(f"Mapa gerado: {OUT}")
