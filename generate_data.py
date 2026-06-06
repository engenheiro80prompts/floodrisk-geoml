import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 500

declividade = np.random.gamma(shape=2.0, scale=3.5, size=N)
dist_drenagem = np.random.exponential(scale=400, size=N).clip(20, 2000)
impermeabilizacao = np.random.beta(a=4, b=2, size=N) * 100
altitude_relativa = np.random.gamma(shape=1.5, scale=8, size=N)
precipitacao_max_24h = np.random.normal(loc=85, scale=25, size=N).clip(20, 180)
capacidade_drenagem = np.random.beta(a=2, b=3, size=N)
densidade_pop = np.random.lognormal(mean=8.5, sigma=0.8, size=N)
tempo_concentracao = np.random.normal(loc=45, scale=15, size=N).clip(10, 120)
cobertura_vegetal = np.clip(100 - impermeabilizacao + np.random.normal(0, 5, N), 0, 80)
manut_microdrenagem = np.random.beta(a=2, b=4, size=N)

from sklearn.preprocessing import MinMaxScaler
feats = np.column_stack([declividade, dist_drenagem, impermeabilizacao,
                          altitude_relativa, precipitacao_max_24h,
                          capacidade_drenagem, manut_microdrenagem])
s = MinMaxScaler().fit_transform(feats)
risco = -0.30*s[:,0] - 0.25*s[:,1] + 0.25*s[:,2] - 0.20*s[:,3] + 0.10*s[:,4] - 0.15*s[:,5] - 0.15*s[:,6]
risco_norm = (risco - risco.min()) / (risco.max() - risco.min())
ocorrencias = np.random.poisson(lam=risco_norm * 8).astype(int)

import random
random.seed(42)
bairros = ["Jardim Primavera","Vila Nova","Parque Industrial","Centro","Jardim das Flores",
           "Vila Operária","Parque São Jorge","Cohab","Jardim Modelo","Vila Esperança"]
nomes_bairros = [random.choice(bairros) + f" {chr(65+i%26)}{i//26}" for i in range(N)]
municipios = np.random.choice(
    ["São Paulo","Guarulhos","Osasco","Santo André","São Bernardo do Campo",
     "Mauá","Diadema","Taboão da Serra","Carapicuíba"],
    size=N, p=[0.35,0.15,0.10,0.08,0.08,0.06,0.06,0.06,0.06]
)
lat = np.random.uniform(-23.85, -23.40, N)
lon = np.random.uniform(-46.80, -46.30, N)

df = pd.DataFrame({
    "setor_id": [f"SP{str(i).zfill(5)}" for i in range(N)],
    "bairro": nomes_bairros, "municipio": municipios,
    "latitude": lat.round(6), "longitude": lon.round(6),
    "declividade_graus": declividade.round(2),
    "dist_drenagem_m": dist_drenagem.round(1),
    "impermeabilizacao_pct": impermeabilizacao.round(1),
    "altitude_relativa_m": altitude_relativa.round(2),
    "precipitacao_max_24h_mm": precipitacao_max_24h.round(1),
    "capacidade_drenagem_idx": capacidade_drenagem.round(3),
    "densidade_pop_hab_km2": densidade_pop.round(0).astype(int),
    "tempo_concentracao_min": tempo_concentracao.round(1),
    "cobertura_vegetal_pct": cobertura_vegetal.round(1),
    "manut_microdrenagem_idx": manut_microdrenagem.round(3),
    "ocorrencias_alagamento_5anos": ocorrencias,
    "risco_alto": (ocorrencias >= 3).astype(int),
})

out = Path(__file__).parent.parent / "data" / "raw" / "setores_censitarios.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Dataset gerado: {len(df)} setores | {df['risco_alto'].mean():.1%} alto risco")
