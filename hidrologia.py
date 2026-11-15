# -*- coding: utf-8 -*-
"""
ANÁLISIS HIDROLÓGICO DE PRECIPITACIONES MÁXIMAS CON ESTACIONES CLIMATOLÓGICAS
==============================================================================
@author: Ramos Castillo Christian

A partir de la lluvia diaria de una o varias estaciones (SMN / CICESE-CLICOM):

  1. Arma la serie de precipitación máxima anual en 24 h y revisa la calidad del registro.
  2. Prueba la serie: datos dudosos (Grubbs-Beck), independencia (Anderson), tendencia
     (Mann-Kendall) y homogeneidad (Helmert, t de Student, Cramer, Pettitt).
  3. Ajusta 9 distribuciones de probabilidad: Normal, Log-Normal, Gumbel (momentos y máxima
     verosimilitud), Gamma, Pearson III, Log-Pearson III, GEV y Doble Gumbel.
  4. Elige la de mejor ajuste (error estándar de ajuste, Kolmogorov-Smirnov, AIC/BIC).
  5. Calcula las precipitaciones de diseño por periodo de retorno, con intervalos de confianza.
  6. Estima curvas intensidad-duración-periodo de retorno (IDF) y un hietograma de diseño.
  7. Entrega un Excel, un PDF y las gráficas de cada estación.

Uso rápido (Colab o terminal):  coloca tu archivo junto a este script y ejecuta
    python hidrologia.py                      # usa Estaciones24h.xlsx
    python hidrologia.py MiArchivo.xlsx

Desde otro programa:
    import hidrologia as hd
    estaciones = hd.leer_archivo("Estaciones24h.xlsx")        # {nombre: tabla diaria}
    res = hd.analizar_estacion(estaciones["Jaina"], "Jaina")
    hd.exportar_excel(res, "Jaina.xlsx");  hd.generar_pdf(res, "Jaina.pdf")

Formato de entrada: una hoja por estación con las columnas  Mes | Dia | Año | Datos  (mm),
o bien  Fecha | Datos.  También lee el archivo de texto de datos diarios del SMN.
"""
import io
import math
import os
import re
import sys
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gamma as _gamma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TR_DEF = (2, 5, 10, 20, 25, 50, 100, 200, 500, 1000)
DUR_DEF = (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 360, 720, 1440)      # minutos
FACTOR_24H = 1.13          # de lluvia diaria (lectura a hora fija) a lluvia máxima en 24 h
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


class ErrorDatos(Exception):
    pass


# =====================================================================
# 1) LECTURA DE DATOS
# =====================================================================
def _sin_acentos(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def normalizar_tabla(df):
    """Devuelve una tabla diaria con columnas [fecha, hp] a partir de Mes/Dia/Año/Datos o Fecha/Datos."""
    cols = {_sin_acentos(c): c for c in df.columns}
    busca = lambda *ops: next((cols[o] for o in ops if o in cols), None)
    c_hp = busca("datos", "dato", "precip", "precipitacion", "hp", "lluvia", "pp", "prec", "precip (mm)")
    c_f = busca("fecha", "date")
    c_m, c_d, c_a = busca("mes", "month"), busca("dia", "day"), busca("ano", "anio", "year", "año")
    if c_hp is None:
        raise ErrorDatos("No encontré la columna de lluvia. Debe llamarse 'Datos' (o Precip, Lluvia, hp).")
    if c_f is not None:
        fecha = pd.to_datetime(df[c_f], errors="coerce", dayfirst=True)
    elif None not in (c_m, c_d, c_a):
        fecha = pd.to_datetime(dict(year=pd.to_numeric(df[c_a], errors="coerce"),
                                    month=pd.to_numeric(df[c_m], errors="coerce"),
                                    day=pd.to_numeric(df[c_d], errors="coerce")), errors="coerce")
    else:
        raise ErrorDatos("Faltan columnas de fecha: usa 'Mes', 'Dia', 'Año' o una columna 'Fecha'.")
    out = pd.DataFrame({"fecha": fecha, "hp": pd.to_numeric(df[c_hp], errors="coerce")})
    out = out.dropna(subset=["fecha"])
    out.loc[out["hp"] < 0, "hp"] = np.nan              # -99, -9999 y similares = dato faltante
    out = out.sort_values("fecha").drop_duplicates("fecha", keep="first").reset_index(drop=True)
    if out["hp"].notna().sum() == 0:
        raise ErrorDatos("La tabla no tiene ningún dato de lluvia válido.")
    return out


def leer_smn(texto):
    """Lee el archivo de texto de datos diarios del SMN (columnas FECHA, PRECIP, EVAP, TMAX, TMIN).
    Devuelve (metadatos, tabla diaria)."""
    meta, filas = {}, []
    for linea in str(texto).splitlines():
        m = re.match(r"\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s+(\S+)", linea)
        if m:
            d, mes, a, v = m.groups()
            try:
                v = float(v)
            except ValueError:
                v = np.nan                                 # "Nulo"
            filas.append((f"{a}-{int(mes):02d}-{int(d):02d}", v))
            continue
        m = re.match(r"\s*([A-Za-zÁÉÍÓÚÑáéíóúñ\- ]{3,20}?)\s*:\s*(.+?)\s*$", linea)
        if m and not filas:
            meta[_sin_acentos(m.group(1))] = m.group(2)
    if not filas:
        raise ErrorDatos("No encontré renglones de datos diarios con formato dd/mm/aaaa.")
    df = pd.DataFrame(filas, columns=["Fecha", "Datos"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return meta, normalizar_tabla(df)


def leer_archivo(origen, nombre=None):
    """Lee .xlsx (una hoja por estación), .csv o .txt del SMN. origen: ruta o archivo en memoria.
    Devuelve {nombre_estación: tabla diaria [fecha, hp]}."""
    nombre = nombre or (origen if isinstance(origen, str) else getattr(origen, "name", "estacion"))
    base = os.path.splitext(os.path.basename(str(nombre)))[0]
    ext = os.path.splitext(str(nombre))[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        hojas = pd.read_excel(origen, sheet_name=None)
        out, errores = {}, []
        for h, df in hojas.items():
            try:
                out[str(h)] = normalizar_tabla(df)
            except ErrorDatos as e:
                errores.append(f"{h}: {e}")
        if not out:
            raise ErrorDatos("Ninguna hoja tiene el formato esperado. " + " | ".join(errores))
        return out
    datos = origen.read() if hasattr(origen, "read") else open(origen, "rb").read()
    texto = datos.decode("utf-8", errors="ignore") if isinstance(datos, bytes) else datos
    if "�" in texto or not texto.strip():
        texto = datos.decode("latin-1", errors="ignore")
    if ext == ".csv":
        df = pd.read_csv(io.StringIO(texto), sep=None, engine="python")
        return {base: normalizar_tabla(df)}
    meta, tabla = leer_smn(texto)
    tabla.attrs["meta"] = meta
    return {str(meta.get("nombre", base)).title() if meta.get("nombre") else base: tabla}


# =====================================================================
# 2) SERIE DE MÁXIMOS ANUALES Y CALIDAD DEL REGISTRO
# =====================================================================
def regimen_mensual(diario):
    """Lluvia media mensual (mm), días con lluvia y máxima histórica en 24 h por mes."""
    d = diario.dropna(subset=["hp"]).copy()
    d["a"], d["m"] = d["fecha"].dt.year, d["fecha"].dt.month
    tot = d.groupby(["a", "m"])["hp"].agg(["sum", "count"]).reset_index()
    tot = tot[tot["count"] >= 25]                          # meses prácticamente completos
    out = pd.DataFrame({"Mes": MESES}, index=range(1, 13))
    out["Lluvia media (mm)"] = tot.groupby("m")["sum"].mean()
    out["Máxima en 24 h (mm)"] = d.groupby("m")["hp"].max()
    out["Días con lluvia (media)"] = d[d["hp"] >= 0.1].groupby(["a", "m"]).size().groupby("m").mean()
    return out.fillna(0.0).reset_index(drop=True)


def tabla_anual(diario, min_meses=9, min_cobertura_lluvias=0.0):
    """Una fila por año: máxima en 24 h, su fecha, lluvia anual, días con dato y validez del año.

    Un año es válido si tiene datos en al menos `min_meses` meses y si la fracción de días con
    dato en la temporada de lluvias es al menos `min_cobertura_lluvias` (0 = no exigir)."""
    d = diario.copy()
    d["a"], d["m"] = d["fecha"].dt.year, d["fecha"].dt.month
    v = d.dropna(subset=["hp"])
    reg = regimen_mensual(diario)
    media = reg["Lluvia media (mm)"].to_numpy()
    lluviosos = [k + 1 for k in range(12) if media[k] >= media.sum() / 12] or list(range(1, 13))
    filas = []
    for a in range(int(d["a"].min()), int(d["a"].max()) + 1):
        va = v[v["a"] == a]
        dias_ano = 366 if (a % 4 == 0 and (a % 100 != 0 or a % 400 == 0)) else 365
        dias_ll = sum(pd.Period(f"{a}-{m:02d}").days_in_month for m in lluviosos)
        fila = {"Año": a, "Días con dato": len(va), "% del año": 100 * len(va) / dias_ano,
                "Meses con dato": va["m"].nunique(),
                "% temporada de lluvias": 100 * len(va[va["m"].isin(lluviosos)]) / dias_ll}
        if len(va):
            k = va["hp"].idxmax()
            fila.update({"Hpmax (mm)": float(va.loc[k, "hp"]), "Fecha del máximo": va.loc[k, "fecha"].date(),
                         "Lluvia anual (mm)": float(va["hp"].sum()),
                         "Días con lluvia": int((va["hp"] >= 0.1).sum())})
        fila["Válido"] = bool(len(va) and fila["Meses con dato"] >= min_meses
                              and fila["% temporada de lluvias"] >= 100 * min_cobertura_lluvias)
        filas.append(fila)
    t = pd.DataFrame(filas)
    t.attrs["meses_lluviosos"] = lluviosos
    return t


def estadisticos(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = {"n": n, "Media": x.mean(), "Mediana": float(np.median(x)), "Desviación estándar": x.std(ddof=1),
           "Mínimo": x.min(), "Máximo": x.max()}
    out["Coeficiente de variación"] = out["Desviación estándar"] / out["Media"]
    out["Coeficiente de asimetría"] = float(stats.skew(x, bias=False)) if n > 2 else np.nan
    out["Curtosis (exceso)"] = float(stats.kurtosis(x, bias=False)) if n > 3 else np.nan
    return out


def tabla_frecuencias(x):
    """Tabla de frecuencias con el número de clases de Sturges."""
    x = np.asarray(x, dtype=float)
    k = int(round(1 + 3.322 * math.log10(len(x))))
    cuentas, bordes = np.histogram(x, bins=k)
    t = pd.DataFrame({"Clase": range(1, k + 1), "Límite inferior (mm)": bordes[:-1], "Límite superior (mm)": bordes[1:],
                      "Marca de clase (mm)": (bordes[:-1] + bordes[1:]) / 2, "Eventos": cuentas})
    t["Frecuencia relativa"] = t["Eventos"] / len(x)
    t["Frecuencia acumulada"] = t["Frecuencia relativa"].cumsum()
    return t


# =====================================================================
# 3) PRUEBAS A LA SERIE
# =====================================================================
def prueba_grubbs_beck(x, anios=None):
    """Datos dudosos altos y bajos (Bulletin 17B, nivel 10 %)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.log10(x[x > 0])
    kn = -0.9043 + 3.345 * math.sqrt(math.log10(n)) - 0.4046 * math.log10(n)
    alto, bajo = 10 ** (y.mean() + kn * y.std(ddof=1)), 10 ** (y.mean() - kn * y.std(ddof=1))
    anios = np.asarray(anios if anios is not None else range(1, n + 1))
    altos = [(int(a), float(v)) for a, v in zip(anios, x) if v > alto]
    bajos = [(int(a), float(v)) for a, v in zip(anios, x) if v < bajo]
    txt = "Sin datos dudosos." if not (altos or bajos) else \
        "Datos dudosos: " + ", ".join(f"{a} ({v:.1f} mm)" for a, v in altos + bajos) + \
        ". Verifícalos con estaciones vecinas o registros de ciclones antes de descartarlos."
    return dict(nombre="Grubbs-Beck (datos dudosos)", estadistico=kn, critico=None, umbral_alto=alto,
                umbral_bajo=bajo, altos=altos, bajos=bajos, pasa=not (altos or bajos), conclusion=txt,
                detalle=f"Kn = {kn:.3f}; umbral alto = {alto:.1f} mm; umbral bajo = {bajo:.1f} mm")


def prueba_anderson(x):
    """Independencia: correlograma con límites de Anderson al 95 %."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    ks = np.arange(1, max(2, n // 3) + 1)
    r = np.array([np.sum(d[:-k] * d[k:]) / np.sum(d ** 2) for k in ks])
    sup = (-1 + 1.96 * np.sqrt(n - ks - 1)) / (n - ks)
    inf = (-1 - 1.96 * np.sqrt(n - ks - 1)) / (n - ks)
    fuera = int(np.sum((r > sup) | (r < inf)))
    pasa = fuera <= 0.10 * len(ks)
    return dict(nombre="Anderson (independencia)", estadistico=float(r[0]), critico=float(sup[0]), pasa=pasa,
                k=ks, r=r, sup=sup, inf=inf,
                detalle=f"r1 = {r[0]:.3f}; {fuera} de {len(ks)} coeficientes fuera de los límites",
                conclusion="La serie es independiente (aleatoria)." if pasa else
                "La serie muestra dependencia entre años: los ajustes deben tomarse con reserva.")


def prueba_mann_kendall(x, anios=None):
    """Tendencia monótona (Mann-Kendall) y pendiente de Sen (mm por año)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    t = np.asarray(anios if anios is not None else np.arange(n), dtype=float)
    i, j = np.triu_indices(n, 1)
    s = float(np.sum(np.sign(x[j] - x[i])))
    _, cuentas = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(cuentas * (cuentas - 1) * (2 * cuentas + 5))) / 18.0
    z = 0.0 if s == 0 else (s - np.sign(s)) / math.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    sen = float(np.median((x[j] - x[i]) / (t[j] - t[i])))
    pasa = p >= 0.05
    return dict(nombre="Mann-Kendall (tendencia)", estadistico=z, critico=1.96, p=p, pendiente=sen, pasa=pasa,
                detalle=f"Z = {z:.3f}; p = {p:.3f}; pendiente de Sen = {sen:+.3f} mm/año",
                conclusion="Sin tendencia significativa." if pasa else
                f"Tendencia {'creciente' if z > 0 else 'decreciente'} significativa: la serie no es estacionaria.")


def prueba_helmert(x):
    x = np.asarray(x, dtype=float)
    sg = np.sign(x - x.mean())
    sg = sg[sg != 0]
    sec = int(np.sum(sg[1:] == sg[:-1]))
    cam = int(np.sum(sg[1:] != sg[:-1]))
    lim = math.sqrt(len(x) - 1)
    pasa = abs(sec - cam) <= lim
    return dict(nombre="Helmert (homogeneidad)", estadistico=sec - cam, critico=lim, pasa=pasa,
                detalle=f"Secuencias = {sec}; cambios = {cam}; |S - C| = {abs(sec - cam)}; límite = {lim:.2f}",
                conclusion="Serie homogénea." if pasa else "Serie no homogénea.")


def prueba_t_student(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    n1 = n // 2
    a, b = x[:n1], x[n1:]
    n2 = len(b)
    td = (a.mean() - b.mean()) / math.sqrt((n1 * a.var() + n2 * b.var()) / (n1 + n2 - 2) * (1 / n1 + 1 / n2))
    tc = float(stats.t.ppf(0.975, n1 + n2 - 2))
    pasa = abs(td) <= tc
    return dict(nombre="t de Student (homogeneidad de la media)", estadistico=td, critico=tc, pasa=pasa,
                detalle=f"Media 1a mitad = {a.mean():.1f} mm; media 2a mitad = {b.mean():.1f} mm; "
                        f"|t| = {abs(td):.3f}; t crítico = {tc:.3f}",
                conclusion="Serie homogénea." if pasa else "La media cambia entre la primera y la segunda mitad.")


def prueba_cramer(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = x.std(ddof=1)
    tc = float(stats.t.ppf(0.975, n - 2))
    ts = []
    for frac in (0.60, 0.30):
        nk = max(2, int(round(frac * n)))
        tau = (x[-nk:].mean() - x.mean()) / s
        ts.append(math.sqrt(nk * (n - 2) / (n - nk * (1 + tau ** 2))) * tau)
    pasa = all(abs(t) <= tc for t in ts)
    return dict(nombre="Cramer (homogeneidad)", estadistico=max(ts, key=abs), critico=tc, pasa=pasa,
                detalle=f"t60 = {ts[0]:.3f}; t30 = {ts[1]:.3f}; t crítico = {tc:.3f}",
                conclusion="Serie homogénea." if pasa else "Serie no homogénea en el último tramo del registro.")


def prueba_pettitt(x, anios=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    sg = np.sign(x[:, None] - x[None, :])
    u = np.array([sg[:t + 1, t + 1:].sum() for t in range(n - 1)])
    k = int(np.argmax(np.abs(u)))
    K = float(abs(u[k]))
    p = min(1.0, 2 * math.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
    anio = int(np.asarray(anios)[k]) if anios is not None else k + 1
    pasa = p >= 0.05
    return dict(nombre="Pettitt (punto de cambio)", estadistico=K, critico=None, p=p, pasa=pasa, anio=anio,
                detalle=f"K = {K:.0f}; p = {p:.3f}; posible cambio después de {anio}",
                conclusion="Sin punto de cambio significativo." if pasa else
                f"Cambio significativo en la serie después de {anio}.")


def pruebas_serie(x, anios):
    return [prueba_grubbs_beck(x, anios), prueba_anderson(x), prueba_mann_kendall(x, anios), prueba_helmert(x),
            prueba_t_student(x), prueba_cramer(x), prueba_pettitt(x, anios)]


# =====================================================================
# 4) DISTRIBUCIONES DE PROBABILIDAD
# =====================================================================
class Ajuste:
    """Distribución ajustada: ppf(F), cdf(x), logpdf(x), parámetros y número de parámetros k."""

    def __init__(self, clave, nombre, metodo, parametros, k, ppf, cdf, logpdf):
        self.clave, self.nombre, self.metodo, self.parametros, self.k = clave, nombre, metodo, parametros, k
        self.ppf, self.cdf, self.logpdf = ppf, cdf, logpdf

    def hp(self, tr):
        """Precipitación para el periodo de retorno tr (años)."""
        tr = np.asarray(tr, dtype=float)
        return self.ppf(1 - 1 / tr)

    def texto_parametros(self):
        return ", ".join(f"{k} = {v:.4g}" for k, v in self.parametros.items())


def _scipy(clave, nombre, metodo, dist, parametros, k):
    return Ajuste(clave, nombre, metodo, parametros, k, dist.ppf, dist.cdf, dist.logpdf)


def _ajusta_normal(x, **_):
    m, s = x.mean(), x.std(ddof=1)
    return _scipy("normal", "Normal", "momentos", stats.norm(m, s), {"media": m, "desv": s}, 2)


def _ajusta_lognormal(x, **_):
    y = np.log(x)
    m, s = y.mean(), y.std(ddof=1)
    return _scipy("lognormal", "Log-Normal", "momentos de ln(x)", stats.lognorm(s, scale=math.exp(m)),
                  {"media ln": m, "desv ln": s}, 2)


def _ajusta_gumbel_mom(x, **_):
    s = x.std(ddof=1)
    a = math.sqrt(6) / math.pi * s
    u = x.mean() - 0.5772156649 * a
    return _scipy("gumbel", "Gumbel", "momentos", stats.gumbel_r(u, a), {"ubicación": u, "escala": a}, 2)


def _ajusta_gumbel_mv(x, **_):
    u, a = stats.gumbel_r.fit(x)
    return _scipy("gumbel_mv", "Gumbel (MV)", "máxima verosimilitud", stats.gumbel_r(u, a),
                  {"ubicación": u, "escala": a}, 2)


def _ajusta_gamma(x, **_):
    k, _, th = stats.gamma.fit(x, floc=0)
    return _scipy("gamma", "Gamma 2P", "máxima verosimilitud", stats.gamma(k, scale=th), {"forma": k, "escala": th}, 2)


def _ajusta_pearson3(x, **_):
    m, s, g = x.mean(), x.std(ddof=1), float(stats.skew(x, bias=False))
    return _scipy("pearson3", "Pearson III", "momentos", stats.pearson3(g, loc=m, scale=s),
                  {"media": m, "desv": s, "asimetría": g}, 3)


def _ajusta_logpearson3(x, **_):
    y = np.log10(x)
    m, s, g = y.mean(), y.std(ddof=1), float(stats.skew(y, bias=False))
    d = stats.pearson3(g, loc=m, scale=s)
    ln10 = math.log(10)
    return Ajuste("logpearson3", "Log-Pearson III", "momentos de log10(x)",
                  {"media log": m, "desv log": s, "asimetría log": g}, 3,
                  lambda F: 10 ** d.ppf(F), lambda v: d.cdf(np.log10(v)),
                  lambda v: d.logpdf(np.log10(v)) - np.log(np.asarray(v, dtype=float) * ln10))


def momentos_l(x):
    """Momentos L muestrales (l1, l2, t3)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    return l1, l2, l3 / l2


def _ajusta_gev(x, **_):
    l1, l2, t3 = momentos_l(x)
    c = 2 / (3 + t3) - math.log(2) / math.log(3)
    k = 7.8590 * c + 2.9554 * c ** 2
    if abs(k) < 1e-6:
        k = 1e-6
    a = l2 * k / ((1 - 2 ** (-k)) * _gamma(1 + k))
    u = l1 - a * (1 - _gamma(1 + k)) / k
    return _scipy("gev", "GEV", "momentos L", stats.genextreme(k, loc=u, scale=a),
                  {"ubicación": u, "escala": a, "forma k": k}, 3)


def _doble_gumbel(theta):
    p, a1, b1, a2, b2 = theta

    def cdf(v):
        v = np.asarray(v, dtype=float)
        return p * np.exp(-np.exp(-(v - b1) / a1)) + (1 - p) * np.exp(-np.exp(-(v - b2) / a2))

    def logpdf(v):
        v = np.asarray(v, dtype=float)
        z1, z2 = (v - b1) / a1, (v - b2) / a2
        f = p / a1 * np.exp(-z1 - np.exp(-z1)) + (1 - p) / a2 * np.exp(-z2 - np.exp(-z2))
        return np.log(np.maximum(f, 1e-300))

    def ppf(F):
        F = np.atleast_1d(np.asarray(F, dtype=float))
        lo = min(b1, b2) - 12 * max(a1, a2)
        out = []
        for f in F:
            hi = max(b1, b2) + max(a1, a2) * (-math.log(-math.log(min(f, 1 - 1e-12))) + 30)
            out.append(optimize.brentq(lambda v: float(cdf(v)) - f, lo, hi, xtol=1e-9))
        return np.array(out)

    return cdf, logpdf, ppf


def _ajusta_doble_gumbel(x, inicio=None, **_):
    """Mezcla de dos poblaciones Gumbel (lluvias ordinarias y ciclónicas) por máxima verosimilitud."""
    n = len(x)
    if n < 20:
        raise ValueError("La Doble Gumbel requiere al menos 20 años de registro.")
    xs = np.sort(x)
    s = x.std(ddof=1)

    def gum(v):
        a = max(math.sqrt(6) / math.pi * v.std(ddof=1), 0.05 * s)
        return a, v.mean() - 0.5772 * a

    inicios = [inicio] if inicio is not None else []
    for frac in (0.10, 0.20, 0.30):
        nc = max(3, int(round(frac * n)))
        a1, b1 = gum(xs[:-nc])
        a2, b2 = gum(xs[-nc:])
        inicios.append([1 - nc / n, a1, b1, a2, b2])

    def nll(th):
        return -float(np.sum(_doble_gumbel(th)[1](x)))

    cotas = [(0.5, 0.98), (0.05 * s, 10 * s), (xs[0] - 5 * s, xs[-1]), (0.05 * s, 10 * s), (xs[0], xs[-1] + 5 * s)]
    mejor = None
    for th0 in inicios:
        try:
            r = optimize.minimize(nll, th0, method="L-BFGS-B", bounds=cotas)
        except Exception:
            continue
        if np.isfinite(r.fun) and (mejor is None or r.fun < mejor.fun):
            mejor = r
    if mejor is None:
        raise ValueError("No convergió el ajuste de la Doble Gumbel.")
    th = list(mejor.x)
    if th[4] < th[2]:                                    # la segunda población es la de eventos mayores
        th = [1 - th[0], th[3], th[4], th[1], th[2]]
    cdf, logpdf, ppf = _doble_gumbel(th)
    aj = Ajuste("doble_gumbel", "Doble Gumbel", "máxima verosimilitud",
                {"p (no ciclónica)": th[0], "escala 1": th[1], "ubicación 1": th[2], "escala 2": th[3],
                 "ubicación 2": th[4]}, 5, ppf, cdf, logpdf)
    aj.theta = th
    return aj


DISTRIBUCIONES = {
    "normal": _ajusta_normal, "lognormal": _ajusta_lognormal, "gumbel": _ajusta_gumbel_mom,
    "gumbel_mv": _ajusta_gumbel_mv, "gamma": _ajusta_gamma, "pearson3": _ajusta_pearson3,
    "logpearson3": _ajusta_logpearson3, "gev": _ajusta_gev, "doble_gumbel": _ajusta_doble_gumbel,
}


def ajustar(x, claves=None):
    """Ajusta las distribuciones pedidas. Devuelve ({clave: Ajuste}, [avisos])."""
    x = np.asarray(x, dtype=float)
    out, avisos = {}, []
    for clave in (claves or DISTRIBUCIONES):
        try:
            if clave in ("lognormal", "logpearson3", "gamma") and np.any(x <= 0):
                raise ValueError("hay máximos anuales iguales a cero")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aj = DISTRIBUCIONES[clave](x)
                prueba = np.asarray(aj.ppf(np.array([0.5, 0.99, 0.999])), dtype=float)
            if not np.all(np.isfinite(prueba)) or np.any(np.diff(prueba) <= 0):
                raise ValueError("el ajuste no dio cuantiles válidos")
            out[clave] = aj
        except Exception as e:
            avisos.append(f"No se ajustó {clave}: {e}.")
    return out, avisos


def posiciones_weibull(x):
    """Serie ordenada de mayor a menor con m, Tr = (n+1)/m y probabilidades (Weibull)."""
    xs = np.sort(np.asarray(x, dtype=float))[::-1]
    n = len(xs)
    m = np.arange(1, n + 1)
    t = pd.DataFrame({"m": m, "Hpmax ordenada (mm)": xs, "Tr (años)": (n + 1) / m})
    t["P excedencia"] = 1 / t["Tr (años)"]
    t["P no excedencia"] = 1 - t["P excedencia"]
    return t


def bondad_de_ajuste(x, ajustes):
    """Error estándar de ajuste (Kite), Kolmogorov-Smirnov, AIC, BIC y R² por distribución."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pos = posiciones_weibull(x)
    obs, F = pos["Hpmax ordenada (mm)"].to_numpy(), pos["P no excedencia"].to_numpy()
    xa = np.sort(x)
    i = np.arange(1, n + 1)
    dcrit = float(stats.kstwo.ppf(0.95, n))
    filas = []
    for clave, aj in ajustes.items():
        calc = np.asarray(aj.ppf(F), dtype=float)
        sse = float(np.sum((obs - calc) ** 2))
        Fx = np.asarray(aj.cdf(xa), dtype=float)
        D = float(max(np.max(np.abs(Fx - i / n)), np.max(np.abs(Fx - (i - 1) / n))))
        ll = float(np.sum(aj.logpdf(x)))
        if not np.isfinite(ll):                          # algún dato cae fuera del dominio de la distribución
            ll = np.nan
        filas.append({"clave": clave, "Distribución": aj.nombre, "Método": aj.metodo, "Parámetros": aj.k,
                      "EEA (mm)": math.sqrt(sse / (n - aj.k)) if n > aj.k else np.nan,
                      "RMSE (mm)": math.sqrt(sse / n), "R²": 1 - sse / float(np.sum((obs - obs.mean()) ** 2)),
                      "KS D": D, "KS D crítico": dcrit, "KS": "Acepta" if D <= dcrit else "Rechaza",
                      "AIC": 2 * aj.k - 2 * ll, "BIC": aj.k * math.log(n) - 2 * ll})
    t = pd.DataFrame(filas).sort_values("EEA (mm)").reset_index(drop=True)
    t.insert(0, "Lugar", range(1, len(t) + 1))
    return t


def elegir_mejor(bondad):
    """Menor error estándar de ajuste entre las distribuciones que acepta Kolmogorov-Smirnov."""
    ok = bondad[bondad["KS"] == "Acepta"]
    return str((ok if len(ok) else bondad).iloc[0]["clave"])


def intervalo_confianza(x, clave, trs, nivel=0.90, B=400, semilla=12345):
    """Intervalo de confianza de hp(Tr) por remuestreo paramétrico (bootstrap)."""
    x = np.asarray(x, dtype=float)
    base = DISTRIBUCIONES[clave](x)
    rng = np.random.default_rng(semilla)
    F = 1 - 1 / np.asarray(trs, dtype=float)
    if clave == "doble_gumbel":
        B = min(B, 150)
    sims = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(B):
            u = rng.uniform(1e-6, 1 - 1e-6, len(x))
            try:
                muestra = np.asarray(base.ppf(u), dtype=float)
                aj = DISTRIBUCIONES[clave](muestra, inicio=getattr(base, "theta", None))
                q = np.asarray(aj.ppf(F), dtype=float)
                if np.all(np.isfinite(q)):
                    sims.append(q)
            except Exception:
                continue
    if len(sims) < 30:
        return np.full(len(F), np.nan), np.full(len(F), np.nan)
    sims = np.array(sims)
    a = (1 - nivel) / 2
    return np.quantile(sims, a, axis=0), np.quantile(sims, 1 - a, axis=0)


# =====================================================================
# 5) CURVAS IDF Y HIETOGRAMA
# =====================================================================
def lluvia_duracion(hp24, d_min, metodo="dick_peschke", tr=None, hp24_tr10=None, R=None):
    """Lluvia (mm) para la duración d_min a partir de la lluvia en 24 h.

    dick_peschke: P_d = P_24 (d/1440)^0.25
    bell: P_d^T = (0.21 ln T + 0.52)(0.54 d^0.25 - 0.50) P_60^10, con P_60^10 = R * P_24^10
          (válida para 5 <= d <= 120 min y 2 <= T <= 100 años; R = cociente lluvia 1 h / 24 h)."""
    d = np.asarray(d_min, dtype=float)
    if metodo == "bell":
        if R is None or hp24_tr10 is None or tr is None:
            raise ValueError("El método de Bell requiere R (lluvia 1 h / lluvia 24 h).")
        return (0.21 * math.log(tr) + 0.52) * (0.54 * d ** 0.25 - 0.50) * R * hp24_tr10
    return hp24 * (d / 1440.0) ** 0.25


def curvas_idf(hp24_por_tr, duraciones=DUR_DEF, metodo="dick_peschke", R=None):
    """Tablas de lluvia (mm) e intensidad (mm/h) por duración y periodo de retorno, y la ecuación
    i = K * Tr^m / d^n ajustada por regresión (i en mm/h, d en minutos, Tr en años)."""
    trs = sorted(hp24_por_tr)
    dur = [d for d in duraciones]
    if metodo == "bell":
        if 10 not in hp24_por_tr:
            raise ValueError("El método de Bell necesita la lluvia de 24 h para Tr = 10 años.")
        trs = [t for t in trs if 2 <= t <= 100]
        dur = [d for d in dur if 5 <= d <= 120]
    P = pd.DataFrame({t: lluvia_duracion(hp24_por_tr[t], dur, metodo, tr=t, hp24_tr10=hp24_por_tr.get(10), R=R)
                      for t in trs}, index=dur)
    I = P.div(np.array(dur) / 60.0, axis=0)
    T, D = np.meshgrid(trs, dur)
    A = np.column_stack([np.ones(T.size), np.log(T.ravel()), -np.log(D.ravel())])
    y = np.log(I.to_numpy().ravel())
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    P.index.name = I.index.name = "Duración (min)"
    P.columns = I.columns = [f"Tr {t}" for t in trs]
    return dict(lluvia=P, intensidad=I, K=float(math.exp(coef[0])), m=float(coef[1]), n=float(coef[2]), r2=float(r2),
                metodo=metodo, R=R, trs=trs, duraciones=dur)


def hietograma(hp24, duracion_min=360, dt_min=30, metodo="dick_peschke", tr=None, hp24_tr10=None, R=None):
    """Hietograma de diseño por bloques alternos (el bloque mayor al centro)."""
    nb = int(round(duracion_min / dt_min))
    if nb < 2:
        raise ValueError("La duración debe ser al menos dos veces el intervalo.")
    d = dt_min * np.arange(1, nb + 1)
    acum = lluvia_duracion(hp24, d, metodo, tr=tr, hp24_tr10=hp24_tr10, R=R)
    inc = np.sort(np.diff(np.concatenate([[0.0], acum])))[::-1]
    orden = np.zeros(nb)
    centro = (nb - 1) // 2
    izq, der = centro - 1, centro + 1
    orden[centro] = inc[0]
    for k, v in enumerate(inc[1:]):
        if (k % 2 == 0 and der < nb) or izq < 0:
            orden[der] = v
            der += 1
        else:
            orden[izq] = v
            izq -= 1
    t = pd.DataFrame({"Inicio (min)": d - dt_min, "Fin (min)": d, "Lluvia (mm)": orden})
    t["Intensidad (mm/h)"] = t["Lluvia (mm)"] / (dt_min / 60.0)
    t["Lluvia acumulada (mm)"] = t["Lluvia (mm)"].cumsum()
    return t


# =====================================================================
# 6) ANÁLISIS COMPLETO DE UNA ESTACIÓN
# =====================================================================
def analizar_estacion(diario, nombre="Estación", min_meses=9, min_cobertura_lluvias=0.0, trs=TR_DEF,
                      factor_24h=FACTOR_24H, distribucion=None, distribuciones=None, nivel_confianza=0.90,
                      bootstrap=400, duraciones=DUR_DEF, metodo_idf="dick_peschke", R=None, excluir_anios=()):
    """Análisis de frecuencias de lluvias máximas de una estación.

    diario: tabla [fecha, hp] (de leer_archivo).      min_meses: meses con dato para aceptar un año.
    trs: periodos de retorno.     factor_24h: factor de lluvia diaria a 24 h (1.13; usa 1.0 para no corregir).
    distribucion: clave para forzar la distribución de diseño (None = la de mejor ajuste).
    metodo_idf: "dick_peschke" o "bell" (este último necesita R = lluvia 1 h / lluvia 24 h).
    excluir_anios: años a quitar de la serie (por ejemplo, datos dudosos ya verificados)."""
    avisos = []
    anual = tabla_anual(diario, min_meses, min_cobertura_lluvias)
    anual.loc[anual["Año"].isin(list(excluir_anios)), "Válido"] = False
    serie = anual[anual["Válido"]][["Año", "Hpmax (mm)", "Fecha del máximo"]].reset_index(drop=True)
    n = len(serie)
    if n < 10:
        raise ErrorDatos(f"{nombre}: solo hay {n} años válidos; se necesitan al menos 10 para un análisis de frecuencias.")
    x, anios = serie["Hpmax (mm)"].to_numpy(dtype=float), serie["Año"].to_numpy()
    if n < 25:
        avisos.append(f"El registro tiene {n} años válidos: con menos de 25 las extrapolaciones son poco confiables.")
    descartados = anual[~anual["Válido"]]["Año"].tolist()
    if descartados:
        avisos.append(f"Años descartados por registro incompleto o exclusión: {', '.join(map(str, descartados))}.")
    flojos = anual[anual["Válido"] & (anual["% temporada de lluvias"] < 80)]["Año"].tolist()
    if flojos:
        avisos.append("Años aceptados con menos del 80 % de días en la temporada de lluvias (su máximo podría "
                      f"estar subestimado): {', '.join(map(str, flojos))}.")

    pruebas = pruebas_serie(x, anios)
    ajustes, av = ajustar(x, distribuciones)
    avisos += av
    if not ajustes:
        raise ErrorDatos(f"{nombre}: no se pudo ajustar ninguna distribución.")
    bondad = bondad_de_ajuste(x, ajustes)
    mejor = elegir_mejor(bondad)
    elegida = distribucion if distribucion in ajustes else mejor
    if distribucion and distribucion not in ajustes:
        avisos.append(f"La distribución '{distribucion}' no está disponible; se usó {ajustes[mejor].nombre}.")

    pos = posiciones_weibull(x)
    for clave in bondad["clave"]:
        pos[ajustes[clave].nombre] = np.asarray(ajustes[clave].ppf(pos["P no excedencia"].to_numpy()), dtype=float)

    trs = sorted({float(t) for t in trs if float(t) > 1})
    trs = [int(t) if float(t).is_integer() else t for t in trs]
    diseno = pd.DataFrame({"Tr (años)": trs, "P no excedencia": [1 - 1 / t for t in trs]})
    for clave in bondad["clave"]:
        diseno[ajustes[clave].nombre] = np.asarray(ajustes[clave].hp(trs), dtype=float)
    li, ls = intervalo_confianza(x, elegida, trs, nivel_confianza, bootstrap) if bootstrap else \
        (np.full(len(trs), np.nan), np.full(len(trs), np.nan))
    hp = np.asarray(ajustes[elegida].hp(trs), dtype=float)
    pct = int(round(nivel_confianza * 100))
    sel = pd.DataFrame({"Tr (años)": trs, "P no excedencia": [1 - 1 / t for t in trs], "hp diaria (mm)": hp,
                        f"Límite inferior {pct} % (mm)": li, f"Límite superior {pct} % (mm)": ls,
                        "hp 24 h (mm)": hp * factor_24h})
    if max(trs) > 4 * n:
        avisos.append(f"Los periodos de retorno mayores a unas 4 veces el registro ({4 * n} años) son "
                      "extrapolaciones con mucha incertidumbre: revisa el intervalo de confianza.")
    if not all(p["pasa"] for p in pruebas[1:]):
        avisos.append("La serie no pasó alguna prueba de independencia, tendencia u homogeneidad: "
                      "revisa el registro (cambios de sitio, de observador o de instrumento) antes de usar los resultados.")

    hp24 = {t: float(v) for t, v in zip(trs, sel["hp 24 h (mm)"])}
    try:
        idf = curvas_idf(hp24, duraciones, metodo_idf, R)
    except ValueError as e:
        avisos.append(f"Curvas IDF: {e} Se usó Dick-Peschke.")
        idf = curvas_idf(hp24, duraciones, "dick_peschke")

    v = diario.dropna(subset=["hp"])
    return dict(nombre=nombre, meta=diario.attrs.get("meta", {}), diario=diario, anual=anual, serie=serie, x=x,
                anios=anios, n=n, periodo=(int(v["fecha"].dt.year.min()), int(v["fecha"].dt.year.max())),
                dias_faltantes=int(diario["hp"].isna().sum()), estadisticos=estadisticos(x),
                mensual=regimen_mensual(diario), frecuencias=tabla_frecuencias(x), pruebas=pruebas, ajustes=ajustes,
                bondad=bondad, mejor=mejor, elegida=elegida, posiciones=pos, diseno=diseno, diseno_sel=sel, idf=idf,
                avisos=avisos, opciones=dict(min_meses=min_meses, min_cobertura_lluvias=min_cobertura_lluvias,
                                             factor_24h=factor_24h, nivel_confianza=nivel_confianza,
                                             metodo_idf=idf["metodo"], R=R, excluir_anios=list(excluir_anios)))


def hietograma_de(res, tr=100, duracion_min=360, dt_min=30):
    """Hietograma de diseño para un periodo de retorno, con la distribución elegida de la estación."""
    aj, f = res["ajustes"][res["elegida"]], res["opciones"]["factor_24h"]
    hp24 = float(np.asarray(aj.hp([tr]))[0]) * f
    hp10 = float(np.asarray(aj.hp([10]))[0]) * f
    return hietograma(hp24, duracion_min, dt_min, res["idf"]["metodo"], tr=tr, hp24_tr10=hp10, R=res["idf"]["R"])


def resumen_estaciones(resultados):
    """Tabla comparativa de varias estaciones. resultados = {nombre: res}."""
    filas = []
    for nombre, r in resultados.items():
        e = r["estadisticos"]
        fila = {"Estación": nombre, "Periodo": f"{r['periodo'][0]}-{r['periodo'][1]}", "Años válidos": r["n"],
                "Media (mm)": e["Media"], "Desv. estándar (mm)": e["Desviación estándar"],
                "Asimetría": e["Coeficiente de asimetría"], "Máximo (mm)": e["Máximo"],
                "Distribución": r["ajustes"][r["elegida"]].nombre,
                "EEA (mm)": float(r["bondad"].set_index("clave").loc[r["elegida"], "EEA (mm)"]),
                "Pruebas no superadas": sum(not p["pasa"] for p in r["pruebas"])}
        for t, v in zip(r["diseno_sel"]["Tr (años)"], r["diseno_sel"]["hp 24 h (mm)"]):
            fila[f"hp24 Tr {t} (mm)"] = v
        filas.append(fila)
    return pd.DataFrame(filas)


# =====================================================================
# 7) FIGURAS
# =====================================================================
AZUL, ROJO, GRIS, VERDE = "#1f4e79", "#c0392b", "#7f8c8d", "#1e8449"


def _png(fig, dpi=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def fig_serie(res):
    s, e, gb, mk = res["serie"], res["estadisticos"], res["pruebas"][0], res["pruebas"][2]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(s["Año"], s["Hpmax (mm)"], color=AZUL, width=0.8, label="Máxima anual en 24 h")
    ax.axhline(e["Media"], color=VERDE, ls="--", lw=1.2, label=f"Media = {e['Media']:.1f} mm")
    ax.axhline(gb["umbral_alto"], color=ROJO, ls=":", lw=1.2, label=f"Umbral de dato dudoso = {gb['umbral_alto']:.0f} mm")
    a = s["Año"].to_numpy(dtype=float)
    ax.plot(a, np.median(s["Hpmax (mm)"]) + mk["pendiente"] * (a - np.median(a)), color="#d68910", lw=1.3,
            label=f"Tendencia de Sen = {mk['pendiente']:+.2f} mm/año")
    for an, v in gb["altos"] + gb["bajos"]:
        ax.annotate(str(an), (an, v), ha="center", va="bottom", fontsize=7.5, color=ROJO)
    ax.set_xlabel("Año")
    ax.set_ylabel("hp (mm)")
    ax.set_title(f"Serie de precipitación máxima anual en 24 h - {res['nombre']}", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    return _png(fig)


def fig_regimen(res):
    m = res["mensual"]
    conteo = pd.to_datetime(res["serie"]["Fecha del máximo"]).dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(m["Mes"], m["Lluvia media (mm)"], color=AZUL, label="Lluvia media mensual (mm)")
    ax.set_ylabel("Lluvia media (mm)")
    ax2 = ax.twinx()
    ax2.plot(m["Mes"], conteo.to_numpy(), "o-", color=ROJO, label="Años cuyo máximo cayó en el mes")
    ax2.set_ylabel("Número de máximos anuales")
    ax2.set_ylim(bottom=0)
    ax.set_title(f"Régimen de lluvias y mes de ocurrencia de los máximos - {res['nombre']}", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    return _png(fig)


def fig_histograma(res):
    x, aj = res["x"], res["ajustes"][res["elegida"]]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(x, bins=len(res["frecuencias"]), density=True, color=AZUL, alpha=0.75, edgecolor="white", label="Observado")
    v = np.linspace(max(x.min() * 0.5, 0.1), x.max() * 1.15, 300)
    ax.plot(v, np.exp(aj.logpdf(v)), color=ROJO, lw=1.8, label=f"Densidad {aj.nombre}")
    ax.set_xlabel("hp máxima anual (mm)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Histograma de máximos anuales - {res['nombre']}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _png(fig)


def fig_correlograma(res):
    p = res["pruebas"][1]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(p["k"], p["r"], color=AZUL, width=0.6)
    ax.plot(p["k"], p["sup"], "--", color=ROJO, lw=1.1, label="Límites de Anderson (95 %)")
    ax.plot(p["k"], p["inf"], "--", color=ROJO, lw=1.1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Desfase k (años)")
    ax.set_ylabel("Autocorrelación r(k)")
    ax.set_title(f"Correlograma de la serie - {res['nombre']}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _png(fig)


def fig_frecuencias(res, todas=True):
    """Datos observados (posición de Weibull) contra las distribuciones ajustadas."""
    pos, sel, aj = res["posiciones"], res["diseno_sel"], res["ajustes"][res["elegida"]]
    tmax = max(float(sel["Tr (años)"].max()), float(pos["Tr (años)"].max()))
    T = np.geomspace(1.02, tmax, 200)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if todas:
        for clave, a in res["ajustes"].items():
            if clave != res["elegida"]:
                ax.plot(T, a.hp(T), lw=0.9, alpha=0.75, label=a.nombre)
    li, ls = sel.iloc[:, 3].to_numpy(dtype=float), sel.iloc[:, 4].to_numpy(dtype=float)
    if np.all(np.isfinite(li)):
        ax.fill_between(sel["Tr (años)"], li, ls, color=ROJO, alpha=0.13,
                        label=f"Intervalo de confianza {int(round(res['opciones']['nivel_confianza'] * 100))} %")
    ax.plot(T, aj.hp(T), color=ROJO, lw=2.4, label=f"{aj.nombre} (elegida)")
    ax.scatter(pos["Tr (años)"], pos["Hpmax ordenada (mm)"], color="k", s=16, zorder=5, label="Observado (Weibull)")
    ax.set_xscale("log")
    marcas = [t for t in (1.01, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 10000) if t <= tmax * 1.01]
    ax.set_xticks(marcas)
    ax.set_xticklabels([f"{t:g}" for t in marcas])
    ax.set_xlabel("Periodo de retorno (años)")
    ax.set_ylabel("hp diaria máxima anual (mm)")
    ax.set_title(f"Análisis de frecuencias - {res['nombre']}", fontsize=10)
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=7.2, ncol=2)
    return _png(fig)


def fig_idf(res):
    idf = res["idf"]
    I = idf["intensidad"]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    colores = plt.cm.viridis(np.linspace(0.05, 0.92, len(I.columns)))
    for c, col in zip(colores, I.columns):
        ax.plot(I.index, I[col], "-o", ms=3, lw=1.3, color=c, label=f"{col} años")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(I.index))
    ax.set_xticklabels([f"{d:g}" for d in I.index], fontsize=7.5)
    ax.set_xlabel("Duración (min)")
    ax.set_ylabel("Intensidad (mm/h)")
    ax.set_title(f"Curvas intensidad - duración - periodo de retorno - {res['nombre']}", fontsize=10)
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return _png(fig)


def fig_hietograma(tabla, titulo="Hietograma de diseño"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    dt = float(tabla["Fin (min)"].iloc[0] - tabla["Inicio (min)"].iloc[0])
    ax.bar(tabla["Inicio (min)"] + dt / 2, tabla["Lluvia (mm)"], width=dt * 0.92, color=AZUL)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Lluvia por intervalo (mm)")
    ax.set_title(f"{titulo} (total = {tabla['Lluvia (mm)'].sum():.1f} mm)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    return _png(fig)


# =====================================================================
# 8) EXCEL
# =====================================================================
def _tabla_pruebas(res):
    return pd.DataFrame([{"Prueba": p["nombre"], "Resultado": "Pasa" if p["pasa"] else "No pasa",
                          "Detalle": p["detalle"], "Conclusión": p["conclusion"]} for p in res["pruebas"]])


def _tabla_parametros(res):
    return pd.DataFrame([{"Distribución": a.nombre, "Método": a.metodo, "Parámetros": a.texto_parametros()}
                         for a in res["ajustes"].values()])


def _tabla_resumen(res):
    e, aj = res["estadisticos"], res["ajustes"][res["elegida"]]
    filas = [("Estación", res["nombre"]), ("Periodo de registro", f"{res['periodo'][0]} - {res['periodo'][1]}"),
             ("Años válidos en la serie", res["n"]), ("Días sin dato en el registro", res["dias_faltantes"]),
             ("Meses mínimos con dato por año", res["opciones"]["min_meses"])]
    filas += [(k if k == "n" else f"{k}", (round(v, 4) if isinstance(v, float) else v)) for k, v in e.items() if k != "n"]
    filas += [("Distribución de mejor ajuste", res["ajustes"][res["mejor"]].nombre),
              ("Distribución usada para diseño", aj.nombre), ("Parámetros", aj.texto_parametros()),
              ("Factor de lluvia diaria a 24 h", res["opciones"]["factor_24h"]),
              ("Método de las curvas IDF", "Bell" if res["idf"]["metodo"] == "bell" else "Dick-Peschke"),
              ("Ecuación IDF", f"i = {res['idf']['K']:.3f} * Tr^{res['idf']['m']:.4f} / d^{res['idf']['n']:.4f}  "
                               f"(i mm/h, d min; R2 = {res['idf']['r2']:.4f})")]
    filas += [(f"Aviso {k}", a) for k, a in enumerate(res["avisos"], 1)]
    return pd.DataFrame(filas, columns=["Concepto", "Valor"])


def exportar_excel(res, archivo, hieto=None):
    """Libro de Excel con todas las tablas del análisis. archivo: ruta o BytesIO."""
    hieto = hieto if hieto is not None else hietograma_de(res)
    hojas = {
        "Resumen": _tabla_resumen(res),
        "Serie anual": res["serie"],
        "Calidad del registro": res["anual"],
        "Régimen mensual": res["mensual"],
        "Frecuencias": res["frecuencias"],
        "Pruebas": _tabla_pruebas(res),
        "Parámetros": _tabla_parametros(res),
        "Bondad de ajuste": res["bondad"].drop(columns="clave"),
        "Observado vs calculado": res["posiciones"],
        "hp por distribución": res["diseno"],
        "Lluvias de diseño": res["diseno_sel"],
        "IDF lluvia (mm)": res["idf"]["lluvia"].reset_index(),
        "IDF intensidad (mm-h)": res["idf"]["intensidad"].reset_index(),
        "Hietograma": hieto,
    }
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(archivo, engine="openpyxl") as xw:
        for nombre, df in hojas.items():
            df = df.copy()
            for c in df.columns:
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].round(4)
            df.to_excel(xw, sheet_name=nombre, index=False)
            ws = xw.sheets[nombre]
            for celda in ws[1]:
                celda.font = Font(bold=True, color="FFFFFF")
                celda.fill = PatternFill("solid", fgColor="1F4E79")
                celda.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for k, c in enumerate(df.columns, 1):
                largo = max([len(str(c))] + [len(str(v)) for v in df[c].head(60)])
                ws.column_dimensions[get_column_letter(k)].width = max(10, min(70, largo + 2))
            ws.freeze_panes = "A2"
    return archivo


# =====================================================================
# 9) PDF
# =====================================================================
def generar_pdf(res, archivo, hieto=None, tr_hieto=100):
    """Reporte en PDF con el procedimiento, las tablas y las gráficas. archivo: ruta o BytesIO."""
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor(AZUL))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor(AZUL), spaceBefore=10,
                        keepWithNext=1)
    N = ParagraphStyle("N", parent=ss["Normal"], fontSize=8.8, leading=11.8, spaceAfter=4)
    S = ParagraphStyle("S", parent=ss["Normal"], fontSize=7.4, leading=9.4, textColor=colors.HexColor("#555555"))
    C = ParagraphStyle("C", parent=ss["Normal"], fontSize=7.2, leading=8.8)
    nombre = res["nombre"]
    titulo = f"Análisis hidrológico de lluvias máximas - Estación {nombre}"

    def tabla(df, anchos=None, fs=7.4, dec=2, envolver=()):
        data = [[Paragraph(f"<b><font color='white'>{c}</font></b>", C) for c in df.columns]]
        for r in df.itertuples(index=False):
            fila = []
            for c, v in zip(df.columns, r):
                if isinstance(v, (float, np.floating)) and float(v).is_integer() and c in ("Eventos", "Clase", "Lugar", "Parámetros", "m"):
                    v = int(v)
                if isinstance(v, (float, np.floating)):
                    v = "-" if not np.isfinite(v) else (f"{v:.4f}" if c in ("P no excedencia", "KS D", "KS D crítico", "R²")
                                                         else f"{v:,.{dec}f}")
                fila.append(Paragraph(str(v), C) if c in envolver else str(v))
            data.append(fila)
        t = Table(data, colWidths=anchos, hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(AZUL)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def img(buf, ancho=6.9 * inch):
        w, h = PILImage.open(buf).size
        buf.seek(0)
        return Image(buf, width=ancho, height=ancho * h / w)

    def formula(latex, fs=11):
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${latex}$", fontsize=fs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=250, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)
        buf.seek(0)
        w, h = PILImage.open(buf).size
        buf.seek(0)
        W = min(w * 72 / 250, 6.8 * inch)
        im = Image(buf, width=W, height=W * h / w)
        im.hAlign = "LEFT"
        return im

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(0.7 * inch, 0.45 * inch, titulo)
        canvas.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"Página {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(archivo, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.65 * inch, bottomMargin=0.7 * inch, title=titulo)
    e, aj, op, idf = res["estadisticos"], res["ajustes"][res["elegida"]], res["opciones"], res["idf"]
    pct = int(round(op["nivel_confianza"] * 100))
    st = [Paragraph(titulo, H1)]
    meta = res.get("meta") or {}
    if meta:
        st.append(Paragraph(" | ".join(f"{k.title()}: {v}" for k, v in list(meta.items())[:8]), S))

    # ---- 1
    st.append(Paragraph("1. Registro y serie de máximos anuales", H2))
    st.append(Paragraph(
        f"El registro diario va de <b>{res['periodo'][0]}</b> a <b>{res['periodo'][1]}</b> y tiene "
        f"{res['dias_faltantes']:,} días sin dato. De cada año se toma la mayor lluvia diaria; un año se acepta si tiene "
        f"datos en al menos {op['min_meses']} meses. Quedan <b>{res['n']} años válidos</b>.", N))
    for a in res["avisos"]:
        st.append(Paragraph(f"<font color='{ROJO}'><b>Aviso:</b></font> {a}", N))
    st.append(img(fig_serie(res), 6.0 * inch))
    filas = [(k, f"{v:.4g}" if isinstance(v, float) else str(v)) for k, v in e.items()]
    mitad = (len(filas) + 1) // 2
    doble = [list(filas[k]) + (list(filas[k + mitad]) if k + mitad < len(filas) else ["", ""]) for k in range(mitad)]
    st.append(Spacer(1, 4))
    st.append(tabla(pd.DataFrame(doble, columns=["Estadístico", "Valor", "Estadístico ", "Valor "]),
                    anchos=[1.9 * inch, 1.0 * inch, 1.9 * inch, 1.0 * inch]))
    st.append(Spacer(1, 4))
    st.append(img(fig_regimen(res), 5.6 * inch))

    # ---- 2
    st.append(Paragraph("2. Pruebas a la serie", H2))
    st.append(Paragraph("El análisis de frecuencias supone que los máximos anuales son independientes, no tienen "
                        "tendencia y provienen de una misma población. Estas pruebas (nivel de significancia 5 %, "
                        "Grubbs-Beck 10 %) revisan esos supuestos.", N))
    st.append(tabla(_tabla_pruebas(res), anchos=[1.6 * inch, 0.6 * inch, 2.5 * inch, 2.4 * inch],
                    envolver=("Prueba", "Detalle", "Conclusión")))
    st.append(Spacer(1, 6))
    st.append(img(fig_correlograma(res), 5.2 * inch))
    st.append(Paragraph("Tabla de frecuencias (número de clases de Sturges)", ParagraphStyle("h", parent=N, fontName="Helvetica-Bold")))
    st.append(tabla(res["frecuencias"], dec=3))
    st.append(Spacer(1, 4))
    st.append(img(fig_histograma(res), 5.2 * inch))

    # ---- 3
    st.append(PageBreak())
    st.append(Paragraph("3. Ajuste de distribuciones de probabilidad", H2))
    st.append(Paragraph("A cada dato ordenado de mayor a menor se le asigna un periodo de retorno con la posición de "
                        "Weibull; la precipitación para un periodo de retorno se obtiene de la función inversa de cada "
                        "distribución:", N))
    st.append(formula(r"T_r = \frac{n+1}{m} \qquad\qquad F(x) = 1 - \frac{1}{T_r} \qquad\qquad x_{T} = F^{-1}\left(1 - \frac{1}{T_r}\right)"))
    st.append(formula(r"\mathrm{Gumbel:}\;\; x_T = u - \alpha\,\ln\left[-\ln\left(1-\frac{1}{T_r}\right)\right], \quad \alpha = \frac{\sqrt{6}}{\pi}\,s, \quad u = \bar{x} - 0.5772\,\alpha", 10))
    st.append(formula(r"\mathrm{Doble\;Gumbel:}\;\; F(x) = p\,e^{-e^{-(x-u_1)/\alpha_1}} + (1-p)\,e^{-e^{-(x-u_2)/\alpha_2}}", 10))
    st.append(tabla(_tabla_parametros(res), anchos=[1.4 * inch, 1.6 * inch, 4.1 * inch], envolver=("Parámetros",)))
    st.append(Paragraph("Bondad de ajuste", ParagraphStyle("h", parent=N, fontName="Helvetica-Bold", spaceBefore=8)))
    st.append(Paragraph("El error estándar de ajuste (EEA) compara los valores observados con los calculados y penaliza "
                        "el número de parámetros (mp). La prueba de Kolmogorov-Smirnov acepta la distribución si la "
                        "máxima diferencia D entre las probabilidades empírica y teórica no rebasa el valor crítico.", N))
    st.append(formula(r"EEA = \sqrt{\frac{\sum (x_{obs} - x_{calc})^2}{n - mp}} \qquad\qquad AIC = 2\,mp - 2\ln L", 10.5))
    st.append(tabla(res["bondad"].drop(columns=["clave", "Método", "RMSE (mm)", "KS D crítico"]), dec=3))
    st.append(Paragraph(
        f"Valor crítico de Kolmogorov-Smirnov (5 %): D = {res['bondad']['KS D crítico'].iloc[0]:.4f}. "
        f"Mejor ajuste: <b>{res['ajustes'][res['mejor']].nombre}</b>. Distribución usada para diseño: "
        f"<b>{aj.nombre}</b> ({aj.texto_parametros()}). Los parámetros se estiman con la misma muestra, por lo que la "
        "prueba de Kolmogorov-Smirnov es aproximada.", N))

    # ---- 4
    st.append(Paragraph("4. Precipitaciones de diseño", H2))
    st.append(Paragraph(
        f"Lluvia máxima diaria por periodo de retorno con la distribución <b>{aj.nombre}</b> y su intervalo de confianza "
        f"del {pct} % (remuestreo paramétrico). La lectura del pluviómetro se hace a hora fija, así que la lluvia máxima "
        f"en 24 horas corridas se estima multiplicando la diaria por <b>{op['factor_24h']:g}</b>.", N))
    st.append(img(fig_frecuencias(res), 6.3 * inch))
    st.append(Spacer(1, 4))
    st.append(tabla(res["diseno_sel"]))
    st.append(Paragraph("Comparación entre distribuciones (hp diaria, mm)", ParagraphStyle("h", parent=N, fontName="Helvetica-Bold", spaceBefore=8)))
    st.append(tabla(res["diseno"].drop(columns="P no excedencia"), fs=6.8, dec=1))

    # ---- 5
    st.append(PageBreak())
    st.append(Paragraph("5. Curvas intensidad - duración - periodo de retorno", H2))
    if idf["metodo"] == "bell":
        st.append(Paragraph(f"Estimadas con la fórmula de Bell a partir de la lluvia de 1 hora y 10 años, tomada como "
                            f"R = {idf['R']:g} veces la de 24 horas (válida de 5 a 120 minutos y de 2 a 100 años):", N))
        st.append(formula(r"P_d^{T} = (0.21\,\ln T + 0.52)\,(0.54\,d^{0.25} - 0.50)\;P_{60}^{10}", 10.5))
    else:
        st.append(Paragraph("La estación solo mide lluvia diaria, así que la lluvia de duraciones menores se estima con "
                            "la relación de Dick y Peschke (d en minutos):", N))
        st.append(formula(r"P_d = P_{24h}\left(\frac{d}{1440}\right)^{0.25} \qquad\qquad i = \frac{P_d}{d/60}", 10.5))
    st.append(Paragraph(f"Ecuación ajustada por regresión (i en mm/h, d en minutos, Tr en años; R² = {idf['r2']:.4f}):", N))
    st.append(formula(rf"i = \frac{{{idf['K']:.3f}\;T_r^{{\,{idf['m']:.4f}}}}}{{d^{{\,{idf['n']:.4f}}}}}", 12))
    st.append(Paragraph("Intensidad (mm/h)", ParagraphStyle("h", parent=N, fontName="Helvetica-Bold")))
    st.append(tabla(idf["intensidad"].reset_index(), fs=6.8, dec=1))
    st.append(Spacer(1, 4))
    st.append(img(fig_idf(res), 6.0 * inch))
    st.append(Paragraph("Estas curvas son una estimación a partir de lluvia diaria. Si existe un pluviógrafo o una "
                        "estación automática cercana, sus datos deben tener prioridad.", S))

    # ---- 6
    hieto = hieto if hieto is not None else hietograma_de(res, tr_hieto)
    st.append(Paragraph("6. Hietograma de diseño", H2))
    dur = float(hieto["Fin (min)"].iloc[-1])
    st.append(Paragraph(f"Tormenta de diseño de {dur / 60:g} horas para Tr = {tr_hieto:g} años por el método de bloques "
                        "alternos: la lluvia de cada intervalo se obtiene de las curvas IDF y el bloque mayor se coloca al centro.", N))
    st.append(img(fig_hietograma(hieto, f"Hietograma de diseño, Tr = {tr_hieto:g} años"), 6.0 * inch))
    st.append(Spacer(1, 4))
    st.append(tabla(hieto))
    doc.build(st, onFirstPage=pie, onLaterPages=pie)
    return archivo


# =====================================================================
# 10) EJECUCIÓN
# =====================================================================
def imprimir_resumen(res):
    e, aj = res["estadisticos"], res["ajustes"][res["elegida"]]
    print(f"\n=== {res['nombre']}  ({res['periodo'][0]}-{res['periodo'][1]}, {res['n']} años válidos) ===")
    print(f"Media {e['Media']:.1f} mm | desv. {e['Desviación estándar']:.1f} mm | asimetría "
          f"{e['Coeficiente de asimetría']:.2f} | máximo {e['Máximo']:.1f} mm")
    for p in res["pruebas"]:
        print(f"  [{'ok' if p['pasa'] else '!!'}] {p['nombre']}: {p['detalle']}")
    print(res["bondad"][["Lugar", "Distribución", "EEA (mm)", "KS", "AIC"]].round(3).to_string(index=False))
    print(f"Distribución de diseño: {aj.nombre}  ({aj.texto_parametros()})")
    print(res["diseno_sel"].round(1).to_string(index=False))
    i = res["idf"]
    print(f"IDF: i = {i['K']:.3f} * Tr^{i['m']:.4f} / d^{i['n']:.4f}   (mm/h, min)")
    for a in res["avisos"]:
        print("  Aviso:", a)


def main(archivo="Estaciones24h.xlsx", carpeta="resultados", **opciones):
    """Analiza todas las estaciones del archivo y guarda Excel, PDF y gráficas en `carpeta`."""
    estaciones = leer_archivo(archivo)
    os.makedirs(carpeta, exist_ok=True)
    resultados = {}
    for nombre, diario in estaciones.items():
        print(f"Procesando la estación: {nombre}")
        try:
            res = analizar_estacion(diario, nombre, **opciones)
        except ErrorDatos as e:
            print("  No se pudo analizar:", e)
            continue
        resultados[nombre] = res
        imprimir_resumen(res)
        base = os.path.join(carpeta, re.sub(r"[^\w\-]+", "_", nombre))
        exportar_excel(res, base + "_resultados.xlsx")
        generar_pdf(res, base + "_reporte.pdf")
        for etiqueta, f in (("serie", fig_serie), ("frecuencias", fig_frecuencias), ("idf", fig_idf)):
            with open(f"{base}_{etiqueta}.png", "wb") as fh:
                fh.write(f(res).getvalue())
    if resultados:
        resumen_estaciones(resultados).round(3).to_excel(os.path.join(carpeta, "resumen_estaciones.xlsx"), index=False)
    print(f"\nProceso finalizado. Archivos en la carpeta '{carpeta}'.")
    return resultados


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["Estaciones24h.xlsx"]))
