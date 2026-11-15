# -*- coding: utf-8 -*-
"""
App de Streamlit - Análisis hidrológico de lluvias máximas con estaciones climatológicas
Ejecutar local:   streamlit run app.py
Requiere en la misma carpeta: hidrologia.py  (y Estaciones24h.xlsx como ejemplo)
"""
import io
import os

import pandas as pd
import streamlit as st

import hidrologia as hd

st.set_page_config(page_title="Análisis hidrológico", page_icon="🌧️", layout="wide")
AQUI = os.path.dirname(os.path.abspath(__file__))
EJEMPLO = os.path.join(AQUI, "Estaciones24h.xlsx")
NOMBRES = {"": "La de mejor ajuste (automática)", "normal": "Normal", "lognormal": "Log-Normal", "gumbel": "Gumbel",
           "gumbel_mv": "Gumbel (máxima verosimilitud)", "gamma": "Gamma 2P", "pearson3": "Pearson III",
           "logpearson3": "Log-Pearson III", "gev": "GEV", "doble_gumbel": "Doble Gumbel"}


@st.cache_resource(show_spinner=False, max_entries=6)
def _leer(datos, nombre):
    return hd.leer_archivo(io.BytesIO(datos), nombre)


@st.cache_resource(show_spinner=False, max_entries=24)
def _analizar(datos, nombre, estacion, op):
    return hd.analizar_estacion(_leer(datos, nombre)[estacion], estacion, **dict(op))


@st.cache_resource(show_spinner=False, max_entries=24)
def _archivos(datos, nombre, estacion, op, tr_h, dur_h, dt_h):
    res = _analizar(datos, nombre, estacion, op)
    hieto = hd.hietograma_de(res, tr_h, dur_h, dt_h)
    xls, pdf = io.BytesIO(), io.BytesIO()
    hd.exportar_excel(res, xls, hieto)
    hd.generar_pdf(res, pdf, hieto, tr_h)
    return xls.getvalue(), pdf.getvalue()


def _lista_numeros(texto, defecto):
    try:
        nums = sorted({float(v) for v in str(texto).replace(";", ",").split(",") if v.strip()})
        nums = [v for v in nums if v > 1]
        return tuple(int(v) if v.is_integer() else v for v in nums) or tuple(defecto)
    except ValueError:
        return tuple(defecto)


# ---------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Datos")
    archivo = st.file_uploader("Archivo de lluvia diaria", type=["xlsx", "xls", "csv", "txt"],
                               help="Excel con una hoja por estación y columnas Mes, Dia, Año, Datos (o Fecha, Datos); "
                                    "CSV con esas columnas; o el archivo de texto de datos diarios del SMN.")
    if archivo is not None:
        datos, nombre_archivo = archivo.getvalue(), archivo.name
    elif os.path.exists(EJEMPLO):
        datos, nombre_archivo = open(EJEMPLO, "rb").read(), "Estaciones24h.xlsx"
        st.caption("Usando el ejemplo incluido: estaciones Jaina y Yecorato, Sinaloa.")
    else:
        datos = None

    st.header("Criterios")
    min_meses = st.slider("Meses con dato para aceptar un año", 6, 12, 9)
    cobertura = st.slider("Días mínimos con dato en temporada de lluvias (%)", 0, 100, 0, step=5,
                          help="0 = no exigir. Súbelo para descartar años a los que les falta parte de la temporada.")
    factor = st.number_input("Factor de lluvia diaria a 24 h", value=hd.FACTOR_24H, min_value=1.0, max_value=1.3, step=0.01,
                             help="El pluviómetro se lee a hora fija; 1.13 lleva la lluvia diaria a 24 horas corridas. "
                                  "Usa 1.00 para no corregir.")
    dist = st.selectbox("Distribución para diseño", list(NOMBRES), format_func=NOMBRES.get)
    trs = _lista_numeros(st.text_input("Periodos de retorno (años)", ", ".join(map(str, hd.TR_DEF))), hd.TR_DEF)
    nivel = st.select_slider("Nivel de confianza del intervalo", [0.80, 0.90, 0.95], value=0.90,
                             format_func=lambda v: f"{int(v * 100)} %")
    excluir_txt = st.text_input("Años a excluir de la serie", "", placeholder="1955, 1969",
                                help="Por ejemplo, datos dudosos que ya verificaste y decidiste descartar.")
    st.header("Curvas IDF")
    metodo = st.radio("Método", ["dick_peschke", "bell"],
                      format_func={"dick_peschke": "Dick-Peschke", "bell": "Bell (requiere R)"}.get)
    R = None
    if metodo == "bell":
        R = st.number_input("R = lluvia de 1 h / lluvia de 24 h", value=0.45, min_value=0.10, max_value=0.90, step=0.01,
                            help="Cociente propio de la zona (isoyetas o un pluviógrafo cercano). Bell es válido de 5 a "
                                 "120 minutos y de 2 a 100 años.")

st.title("Análisis hidrológico de lluvias máximas")
st.caption("Serie de máximos anuales, pruebas estadísticas, ajuste de distribuciones, lluvias de diseño, curvas IDF "
           "y hietograma, a partir de la lluvia diaria de estaciones climatológicas.")

if datos is None:
    st.info("Sube un archivo de lluvia diaria para comenzar.")
    st.stop()

try:
    estaciones = _leer(datos, nombre_archivo)
except hd.ErrorDatos as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()
except Exception as e:
    st.error(f"No se pudo abrir el archivo: {e}")
    st.stop()

nombres = list(estaciones)
estacion = st.selectbox("Estación", nombres) if len(nombres) > 1 else nombres[0]
excluir = tuple(int(v) for v in _lista_numeros(excluir_txt, ()))
op = tuple(sorted(dict(min_meses=min_meses, min_cobertura_lluvias=cobertura / 100, trs=trs, factor_24h=float(factor),
                       distribucion=dist or None, nivel_confianza=nivel, metodo_idf=metodo, R=R,
                       excluir_anios=excluir).items(), key=lambda kv: kv[0]))

try:
    with st.spinner("Analizando la estación..."):
        res = _analizar(datos, nombre_archivo, estacion, op)
except hd.ErrorDatos as e:
    st.error(str(e))
    st.stop()

e, aj = res["estadisticos"], res["ajustes"][res["elegida"]]
sel = res["diseno_sel"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registro", f"{res['periodo'][0]}-{res['periodo'][1]}", help=f"{res['dias_faltantes']:,} días sin dato")
c2.metric("Años válidos", res["n"])
c3.metric("Distribución de diseño", aj.nombre,
          help=f"Mejor ajuste: {res['ajustes'][res['mejor']].nombre}. Parámetros: {aj.texto_parametros()}")
tr_ref = 100 if 100 in list(sel["Tr (años)"]) else sel["Tr (años)"].iloc[-1]
c4.metric(f"hp 24 h, Tr {tr_ref:g} años", f"{float(sel.loc[sel['Tr (años)'] == tr_ref, 'hp 24 h (mm)'].iloc[0]):.1f} mm")
for a in res["avisos"]:
    st.warning(a)

t_reg, t_pru, t_aju, t_dis, t_idf, t_des, t_comp = st.tabs(
    ["Registro", "Pruebas", "Ajuste", "Lluvias de diseño", "IDF y hietograma", "Descargas", "Comparar estaciones"])

with t_reg:
    st.image(hd.fig_serie(res).getvalue(), width="stretch")
    a, b = st.columns(2)
    with a:
        st.markdown("**Estadísticos de la serie de máximos anuales**")
        st.dataframe(pd.DataFrame([(k, v) for k, v in e.items()], columns=["Estadístico", "Valor"]).round(4),
                     hide_index=True, width="stretch")
    with b:
        st.markdown("**Régimen mensual**")
        st.dataframe(res["mensual"].round(1), hide_index=True, width="stretch")
    st.image(hd.fig_regimen(res).getvalue(), width="stretch")
    st.markdown("**Calidad del registro por año** (los años no válidos quedan fuera de la serie)")
    st.dataframe(res["anual"].round(1), hide_index=True, width="stretch")

with t_pru:
    st.caption("El análisis de frecuencias supone máximos anuales independientes, sin tendencia y de una sola población. "
               "Nivel de significancia 5 % (Grubbs-Beck 10 %).")
    for p in res["pruebas"]:
        (st.success if p["pasa"] else st.error)(f"**{p['nombre']}** — {p['conclusion']}  \n{p['detalle']}")
    st.image(hd.fig_correlograma(res).getvalue(), width="stretch")
    st.markdown("**Tabla de frecuencias**")
    st.dataframe(res["frecuencias"].round(3), hide_index=True, width="stretch")
    st.image(hd.fig_histograma(res).getvalue(), width="stretch")

with t_aju:
    st.caption("EEA: error estándar de ajuste (menor es mejor). KS: prueba de Kolmogorov-Smirnov al 5 %. "
               "La distribución de diseño es la de menor EEA entre las que acepta KS, salvo que elijas otra en la barra lateral.")
    st.dataframe(res["bondad"].drop(columns="clave").round(4), hide_index=True, width="stretch")
    st.image(hd.fig_frecuencias(res).getvalue(), width="stretch")
    st.markdown("**Parámetros ajustados**")
    st.dataframe(hd._tabla_parametros(res), hide_index=True, width="stretch")
    with st.expander("Datos observados contra calculados (posición de Weibull)"):
        st.dataframe(res["posiciones"].round(3), hide_index=True, width="stretch")

with t_dis:
    st.markdown(f"**Lluvias de diseño con {aj.nombre}**")
    st.dataframe(sel.round(4), hide_index=True, width="stretch")
    st.caption(f"hp 24 h = hp diaria × {res['opciones']['factor_24h']:g}. El intervalo de confianza se obtiene por "
               "remuestreo paramétrico; mientras más se aleja el periodo de retorno del tamaño del registro, más se abre.")
    st.markdown("**Comparación entre distribuciones (hp diaria, mm)**")
    st.dataframe(res["diseno"].round(2), hide_index=True, width="stretch")

with t_idf:
    idf = res["idf"]
    st.latex(rf"i = \frac{{{idf['K']:.3f}\; T_r^{{{idf['m']:.4f}}}}}{{d^{{{idf['n']:.4f}}}}}"
             rf"\qquad (i\ \mathrm{{mm/h}},\ d\ \mathrm{{min}},\ R^2 = {idf['r2']:.4f})")
    st.caption("Estimación a partir de lluvia diaria ("
               + ("Bell" if idf["metodo"] == "bell" else "Dick-Peschke")
               + "). Si hay un pluviógrafo o estación automática cercana, sus datos tienen prioridad.")
    st.image(hd.fig_idf(res).getvalue(), width="stretch")
    st.markdown("**Intensidad (mm/h)**")
    st.dataframe(idf["intensidad"].round(2), width="stretch")
    with st.expander("Lluvia acumulada por duración (mm)"):
        st.dataframe(idf["lluvia"].round(2), width="stretch")
    st.subheader("Hietograma de diseño")
    h1, h2, h3 = st.columns(3)
    tr_h = h1.number_input("Periodo de retorno (años)", value=100, min_value=2, max_value=10000)
    limite = 2 if idf["metodo"] == "bell" else 24
    dur_h = h2.number_input("Duración de la tormenta (h)", value=min(6, limite), min_value=1, max_value=limite)
    dt_h = h3.selectbox("Intervalo (min)", [5, 10, 15, 30, 60], index=3 if idf["metodo"] != "bell" else 1)
    try:
        hieto = hd.hietograma_de(res, tr_h, dur_h * 60, dt_h)
        st.image(hd.fig_hietograma(hieto, f"Hietograma de diseño, Tr = {tr_h:g} años").getvalue(), width="stretch")
        st.dataframe(hieto.round(2), hide_index=True, width="stretch")
    except ValueError as err:
        hieto = None
        st.error(str(err))

with t_des:
    st.caption("El Excel trae todas las tablas del análisis; el PDF explica el procedimiento con fórmulas y gráficas. "
               "Ambos usan el hietograma configurado en la pestaña anterior.")
    if st.button("Preparar Excel y PDF de esta estación", type="primary"):
        with st.spinner("Generando archivos..."):
            st.session_state["archivos"] = (estacion, op) + _archivos(datos, nombre_archivo, estacion, op, tr_h,
                                                                       dur_h * 60, dt_h)
    arch = st.session_state.get("archivos")
    if arch and arch[0] == estacion and arch[1] == op:
        d1, d2 = st.columns(2)
        d1.download_button("Descargar Excel", arch[2], file_name=f"{estacion}_resultados.xlsx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        d2.download_button("Descargar PDF", arch[3], file_name=f"{estacion}_reporte.pdf", mime="application/pdf",
                           width="stretch")

with t_comp:
    if len(nombres) < 2:
        st.info("El archivo solo tiene una estación. Sube un Excel con una hoja por estación para compararlas.")
    else:
        todos, fallas = {}, []
        with st.spinner("Analizando todas las estaciones..."):
            for n_ in nombres:
                try:
                    todos[n_] = _analizar(datos, nombre_archivo, n_, op)
                except hd.ErrorDatos as err:
                    fallas.append(str(err))
        for f in fallas:
            st.warning(f)
        if todos:
            resumen = hd.resumen_estaciones(todos)
            st.dataframe(resumen.round(2), hide_index=True, width="stretch")
            graf = pd.DataFrame({n_: r["diseno_sel"].set_index("Tr (años)")["hp 24 h (mm)"] for n_, r in todos.items()})
            graf.index = graf.index.astype(str)
            st.markdown("**hp en 24 h por periodo de retorno (mm)**")
            st.bar_chart(graf, stack=False)
            buf = io.BytesIO()
            resumen.round(3).to_excel(buf, index=False)
            st.download_button("Descargar comparación en Excel", buf.getvalue(), file_name="resumen_estaciones.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
