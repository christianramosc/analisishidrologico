# -*- coding: utf-8 -*-
"""Pruebas del motor hidrológico.  Ejecutar:  pytest"""
import io
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import hidrologia as hd  # noqa: E402


def diario_sintetico(anios=range(1960, 2000), semilla=1, faltan=()):
    """Lluvia diaria con un máximo anual conocido el 15 de septiembre."""
    rng = np.random.default_rng(semilla)
    maximos = stats.gumbel_r(80, 25).rvs(len(list(anios)), random_state=rng)
    filas = []
    for a, mx in zip(anios, maximos):
        for f in pd.date_range(f"{a}-01-01", f"{a}-12-31"):
            hp = float(rng.uniform(0, 10)) if f.month in (7, 8, 9) else 0.0
            if f.month == 9 and f.day == 15:
                hp = float(mx)
            if (a, f.month) in faltan:
                hp = np.nan
            filas.append((f.month, f.day, f.year, hp))
    return pd.DataFrame(filas, columns=["Mes", "Dia", "Año", "Datos"]), dict(zip(anios, maximos))


# ---------------- lectura ----------------
def test_normalizar_tabla_formatos():
    crudo, _ = diario_sintetico(range(1990, 1992))
    t = hd.normalizar_tabla(crudo)
    assert list(t.columns) == ["fecha", "hp"] and len(t) == 730
    alt = pd.DataFrame({"FECHA": t["fecha"].dt.strftime("%d/%m/%Y"), "Precip": t["hp"]})
    assert hd.normalizar_tabla(alt)["hp"].sum() == pytest.approx(t["hp"].sum())
    crudo.loc[3, "Datos"] = -99.0                      # código de dato faltante
    assert np.isnan(hd.normalizar_tabla(crudo)["hp"].iloc[3])
    with pytest.raises(hd.ErrorDatos):
        hd.normalizar_tabla(pd.DataFrame({"a": [1], "b": [2]}))


def test_leer_smn():
    texto = """CONAGUA
ESTACION  : 25046
NOMBRE    : JAINA
ESTADO    : SINALOA
LATITUD   : 25.9 °

FECHA        PRECIP  EVAP   TMAX   TMIN
             (mm)    (mm)   (°C)   (°C)
01/01/1942   0       Nulo   Nulo   Nulo
02/01/1942   0.7     5.1    25     10
03/01/1942   Nulo    Nulo   Nulo   Nulo
"""
    meta, t = hd.leer_smn(texto)
    assert meta["nombre"] == "JAINA" and meta["estacion"] == "25046"
    assert len(t) == 3 and t["hp"].iloc[1] == 0.7 and np.isnan(t["hp"].iloc[2])
    assert t["fecha"].iloc[1] == pd.Timestamp("1942-01-02")
    est = hd.leer_archivo(io.BytesIO(texto.encode("utf-8")), "25046.txt")
    assert list(est) == ["Jaina"]


# ---------------- serie anual ----------------
def test_tabla_anual_y_criterio_de_meses():
    crudo, maximos = diario_sintetico(range(1980, 1990), faltan={(1984, m) for m in (1, 2, 3, 4)} | {(1986, 9)})
    t = hd.tabla_anual(hd.normalizar_tabla(crudo), min_meses=9)
    assert not t.set_index("Año").loc[1984, "Válido"]            # solo 8 meses con dato
    fila = t.set_index("Año").loc[1981]
    assert fila["Hpmax (mm)"] == pytest.approx(maximos[1981]) and str(fila["Fecha del máximo"]) == "1981-09-15"
    assert t.set_index("Año").loc[1986, "Válido"]                # 11 meses: se acepta
    assert t.set_index("Año").loc[1986, "% temporada de lluvias"] < 80
    t2 = hd.tabla_anual(hd.normalizar_tabla(crudo), min_meses=9, min_cobertura_lluvias=0.9)
    assert not t2.set_index("Año").loc[1986, "Válido"]           # le falta septiembre, mes lluvioso


# ---------------- pruebas ----------------
def test_mann_kendall_detecta_tendencia():
    rng = np.random.default_rng(0)
    ruido = rng.normal(0, 5, 60)
    assert hd.prueba_mann_kendall(100 + ruido)["pasa"]
    con = hd.prueba_mann_kendall(100 + ruido + 1.0 * np.arange(60))
    assert not con["pasa"] and con["pendiente"] == pytest.approx(1.0, abs=0.2)


def test_homogeneidad_detecta_salto():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(80, 8, 30), rng.normal(120, 8, 30)])
    assert not hd.prueba_t_student(x)["pasa"]
    assert not hd.prueba_cramer(x)["pasa"]
    p = hd.prueba_pettitt(x, np.arange(1950, 2010))
    assert not p["pasa"] and abs(p["anio"] - 1979) <= 2
    assert not hd.prueba_helmert(x)["pasa"]


def test_grubbs_beck_y_anderson():
    rng = np.random.default_rng(3)
    x = stats.gumbel_r(80, 15).rvs(50, random_state=rng)
    assert hd.prueba_anderson(x)["pasa"]
    x2 = x.copy()
    x2[10] = 900.0
    gb = hd.prueba_grubbs_beck(x2, np.arange(1950, 2000))
    assert not gb["pasa"] and gb["altos"][0][0] == 1960
    kn50 = -0.9043 + 3.345 * math.sqrt(math.log10(50)) - 0.4046 * math.log10(50)
    assert gb["estadistico"] == pytest.approx(kn50) and kn50 == pytest.approx(2.768, abs=0.01)   # tabla 17B


# ---------------- distribuciones ----------------
def test_gumbel_momentos_coincide_con_la_formula():
    x = np.array([119.4, 239.4, 56.0, 61.7, 89.0, 110.0, 93.0, 105.0, 106.0, 93.5, 70.2, 84.0])
    aj = hd.ajustar(x, ["gumbel"])[0]["gumbel"]
    s, m = x.std(ddof=1), x.mean()
    alfa, beta = 1.2825 / s, m - 0.45 * s                         # fórmulas del reporte original
    for tr in (2, 50, 1000):
        esperado = beta - 1 / alfa * math.log(-math.log((tr - 1) / tr))
        assert float(aj.hp(tr)) == pytest.approx(esperado, rel=2e-3)


@pytest.mark.parametrize("clave, dist", [("gumbel_mv", stats.gumbel_r(80, 20)), ("gev", stats.genextreme(-0.1, 80, 20)),
                                         ("lognormal", stats.lognorm(0.35, scale=85)), ("normal", stats.norm(90, 15)),
                                         ("gamma", stats.gamma(9, scale=10)), ("pearson3", stats.pearson3(0.8, 90, 25)),
                                         ("gumbel", stats.gumbel_r(80, 20))])
def test_recupera_cuantiles_de_muestra_grande(clave, dist):
    x = dist.rvs(4000, random_state=np.random.default_rng(7))
    aj = hd.ajustar(x, [clave])[0][clave]
    for tr in (10, 100):
        assert float(aj.hp(tr)) == pytest.approx(float(dist.ppf(1 - 1 / tr)), rel=0.05)
    F = np.array([0.1, 0.5, 0.9, 0.99])
    assert np.allclose(aj.cdf(aj.ppf(F)), F, atol=1e-6)           # ppf y cdf son inversas


def test_log_pearson3_y_doble_gumbel_son_coherentes():
    rng = np.random.default_rng(11)
    x = np.concatenate([stats.gumbel_r(70, 12).rvs(160, random_state=rng), stats.gumbel_r(170, 30).rvs(40, random_state=rng)])
    aj, avisos = hd.ajustar(x, ["logpearson3", "doble_gumbel"])
    assert not avisos
    F = np.array([0.2, 0.5, 0.9, 0.99, 0.999])
    for a in aj.values():
        assert np.allclose(a.cdf(a.ppf(F)), F, atol=1e-6)
        v = np.linspace(x.min() * 0.6, x.max() * 2.5, 6000)
        assert np.trapezoid(np.exp(a.logpdf(v)), v) == pytest.approx(1.0, abs=0.02)   # la densidad integra a 1
    th = aj["doble_gumbel"].theta
    assert th[4] > th[2] and 0.6 < th[0] < 0.95                    # separa las dos poblaciones
    b = hd.bondad_de_ajuste(x, {**aj, **hd.ajustar(x, ["gumbel"])[0]})
    assert b.iloc[0]["clave"] == "doble_gumbel"                    # gana a la Gumbel simple en datos mezclados


def test_bondad_y_eleccion():
    x = stats.gumbel_r(80, 20).rvs(60, random_state=np.random.default_rng(5))
    aj, _ = hd.ajustar(x)
    b = hd.bondad_de_ajuste(x, aj)
    assert len(b) == len(aj) == 9 and list(b["Lugar"]) == list(range(1, 10))
    assert b["EEA (mm)"].is_monotonic_increasing
    fila = b.set_index("clave").loc["gumbel"]
    pos = hd.posiciones_weibull(x)
    calc = aj["gumbel"].ppf(pos["P no excedencia"].to_numpy())
    assert fila["EEA (mm)"] == pytest.approx(math.sqrt(np.sum((pos["Hpmax ordenada (mm)"] - calc) ** 2) / (60 - 2)))
    assert fila["KS D"] == pytest.approx(stats.kstest(x, aj["gumbel"].cdf).statistic)
    assert hd.elegir_mejor(b) in aj
    assert pos["Tr (años)"].iloc[0] == pytest.approx(61.0) and pos["Hpmax ordenada (mm)"].iloc[0] == x.max()


def test_intervalo_de_confianza_contiene_al_estimado():
    x = stats.gumbel_r(80, 20).rvs(50, random_state=np.random.default_rng(9))
    trs = [2, 10, 100]
    li, ls = hd.intervalo_confianza(x, "gumbel", trs, B=200)
    hp = hd.ajustar(x, ["gumbel"])[0]["gumbel"].hp(trs)
    assert np.all(li < hp) and np.all(hp < ls)
    assert (ls - li)[2] > (ls - li)[0]                            # se abre al crecer Tr


# ---------------- IDF y hietograma ----------------
def test_idf_dick_peschke():
    hp24 = {2: 80.0, 10: 120.0, 100: 180.0}
    idf = hd.curvas_idf(hp24)
    assert idf["lluvia"].loc[1440, "Tr 10"] == pytest.approx(120.0)
    assert idf["lluvia"].loc[60, "Tr 100"] == pytest.approx(180 * (60 / 1440) ** 0.25)
    assert idf["intensidad"].loc[30, "Tr 2"] == pytest.approx(80 * (30 / 1440) ** 0.25 / 0.5)
    assert idf["n"] == pytest.approx(0.75, abs=1e-9) and idf["r2"] > 0.98
    assert (idf["intensidad"].diff().dropna() < 0).all().all()     # la intensidad baja con la duración


def test_idf_bell():
    hp24 = {2: 80.0, 10: 120.0, 100: 180.0, 500: 220.0}
    idf = hd.curvas_idf(hp24, metodo="bell", R=0.5)
    assert idf["trs"] == [2, 10, 100] and max(idf["duraciones"]) == 120
    p60_10 = 0.5 * 120
    assert idf["lluvia"].loc[60, "Tr 10"] == pytest.approx((0.21 * math.log(10) + 0.52) * (0.54 * 60 ** 0.25 - 0.5) * p60_10)
    assert idf["lluvia"].loc[60, "Tr 10"] == pytest.approx(p60_10, rel=0.01)   # Bell reproduce P60 de Tr 10
    with pytest.raises(ValueError):
        hd.curvas_idf({2: 80.0, 100: 180.0}, metodo="bell", R=0.5)


def test_hietograma_bloques_alternos():
    h = hd.hietograma(150.0, duracion_min=360, dt_min=30)
    assert len(h) == 12
    assert h["Lluvia (mm)"].sum() == pytest.approx(150 * (360 / 1440) ** 0.25)
    assert int(h["Lluvia (mm)"].idxmax()) == 5                    # el bloque mayor al centro
    assert h["Lluvia (mm)"].iloc[:6].is_monotonic_increasing and h["Lluvia (mm)"].iloc[5:].is_monotonic_decreasing


# ---------------- análisis completo ----------------
def test_analisis_completo_y_archivos():
    crudo, maximos = diario_sintetico(range(1960, 2000), faltan={(1970, m) for m in range(1, 7)})
    res = hd.analizar_estacion(hd.normalizar_tabla(crudo), "Prueba", bootstrap=60)
    assert res["n"] == 39 and 1970 not in res["anios"]
    assert res["x"].max() == pytest.approx(max(v for a, v in maximos.items() if a != 1970))
    sel = res["diseno_sel"]
    assert (sel["hp diaria (mm)"].diff().dropna() > 0).all()
    assert np.allclose(sel["hp 24 h (mm)"], 1.13 * sel["hp diaria (mm)"])
    real = stats.gumbel_r(80, 25).ppf(0.99)
    assert float(sel.loc[sel["Tr (años)"] == 100, "hp diaria (mm)"].iloc[0]) == pytest.approx(real, rel=0.30)
    forzada = hd.analizar_estacion(hd.normalizar_tabla(crudo), "Prueba", distribucion="normal", bootstrap=0,
                                   factor_24h=1.0, excluir_anios=[1961])
    assert forzada["elegida"] == "normal" and forzada["n"] == 38
    xls, pdf = io.BytesIO(), io.BytesIO()
    hd.exportar_excel(res, xls)
    hd.generar_pdf(res, pdf)
    assert pdf.getvalue()[:4] == b"%PDF"
    hojas = pd.read_excel(io.BytesIO(xls.getvalue()), sheet_name=None)
    assert {"Resumen", "Serie anual", "Pruebas", "Bondad de ajuste", "Lluvias de diseño", "Hietograma"} <= set(hojas)
    assert len(hojas["Serie anual"]) == 39
    assert len(hd.resumen_estaciones({"A": res, "B": forzada})) == 2


def test_registro_corto_da_error():
    crudo, _ = diario_sintetico(range(2000, 2006))
    with pytest.raises(hd.ErrorDatos):
        hd.analizar_estacion(hd.normalizar_tabla(crudo), "Corta")


@pytest.mark.skipif(not os.path.exists(os.path.join(RAIZ, "Estaciones24h.xlsx")), reason="sin archivo de ejemplo")
def test_estaciones_de_ejemplo():
    est = hd.leer_archivo(os.path.join(RAIZ, "Estaciones24h.xlsx"))
    assert set(est) == {"Jaina", "Yecorato"}
    res = hd.analizar_estacion(est["Jaina"], "Jaina", bootstrap=0)
    assert res["n"] == 73                                          # los 73 años del reporte original
    assert res["estadisticos"]["Máximo"] == pytest.approx(248.6)
    s = res["serie"].set_index("Año")["Hpmax (mm)"]
    assert s[1942] == pytest.approx(119.4) and s[1943] == pytest.approx(239.4) and 1945 not in s.index
