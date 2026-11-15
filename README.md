# Análisis hidrológico de lluvias máximas con estaciones climatológicas

Motor en Python y app de Streamlit para hacer el análisis de frecuencias de precipitaciones máximas
a partir de la lluvia diaria de estaciones climatológicas (SMN / CICESE-CLICOM). Incluye como ejemplo
las estaciones **Jaina** y **Yecorato**, Sinaloa (`Estaciones24h.xlsx`), pero sirve para cualquier estación.

## Qué hace

1. **Serie de máximos anuales en 24 h**, con la fecha de cada máximo y una revisión de la calidad del
   registro (días y meses con dato por año, cobertura de la temporada de lluvias).
2. **Pruebas a la serie**: datos dudosos (Grubbs-Beck), independencia (Anderson), tendencia
   (Mann-Kendall y pendiente de Sen) y homogeneidad (Helmert, t de Student, Cramer, Pettitt).
3. **Ajuste de 9 distribuciones**: Normal, Log-Normal, Gumbel (momentos y máxima verosimilitud),
   Gamma, Pearson III, Log-Pearson III, GEV (momentos L) y Doble Gumbel (dos poblaciones, útil en
   zonas con ciclones).
4. **Bondad de ajuste**: error estándar de ajuste (EEA), Kolmogorov-Smirnov, AIC, BIC y R². Se elige
   la de menor EEA entre las que acepta Kolmogorov-Smirnov, o la que tú indiques.
5. **Lluvias de diseño** por periodo de retorno, con intervalo de confianza por remuestreo y el
   factor 1.13 para pasar de lluvia diaria a lluvia en 24 horas.
6. **Curvas IDF** (Dick-Peschke, o Bell si conoces el cociente lluvia 1 h / 24 h), con su ecuación
   `i = K·Tr^m / d^n`, y **hietograma de diseño** por bloques alternos.
7. **Entregables**: Excel con todas las tablas, PDF con el procedimiento y las fórmulas, gráficas y
   una tabla comparativa entre estaciones.

## Formato de los datos

Un Excel con **una hoja por estación** y estas columnas (también se acepta `Fecha | Datos`, un CSV con
las mismas columnas o el archivo de texto de datos diarios del SMN):

| Mes | Dia | Año | Datos |
|-----|-----|-----|-------|
| 1 | 1 | 1998 | 6 |

`Datos` es la lluvia del día en mm. Las celdas vacías y los valores negativos (-99, -9999) se toman
como dato faltante.

## Cómo usarlo

App (también se puede desplegar en Streamlit Community Cloud con `app.py` como archivo principal):

```bash
pip install -r requirements.txt
streamlit run app.py
```

Script, para procesar todas las estaciones de un archivo y guardar los resultados en `resultados/`:

```bash
python hidrologia.py                 # usa Estaciones24h.xlsx
python hidrologia.py MiArchivo.xlsx
```

Desde Colab u otro programa:

```python
import hidrologia as hd

estaciones = hd.leer_archivo("Estaciones24h.xlsx")
res = hd.analizar_estacion(estaciones["Jaina"], "Jaina", trs=(2, 5, 10, 25, 50, 100), min_meses=9)
print(res["bondad"])            # comparación de distribuciones
print(res["diseno_sel"])        # lluvias de diseño con intervalo de confianza
hd.exportar_excel(res, "Jaina.xlsx")
hd.generar_pdf(res, "Jaina.pdf")
```

Pruebas: `pip install pytest` y luego `pytest`.

## Cambios respecto a la versión original

- Se **quitó el modelo Random Forest**. Un bosque aleatorio no puede extrapolar más allá del mayor
  valor con el que se entrenó, así que no sirve para estimar lluvias de periodos de retorno mayores
  al registro; además se entrenaba con el periodo de retorno empírico de los mismos datos, por lo que
  su error pequeño era solo memorización. En su lugar hay más distribuciones de probabilidad, que sí
  extrapolan con una base teórica.
- La desviación estándar ahora es la muestral (n − 1) y el número de clases de la tabla de
  frecuencias sigue la regla de Sturges.
- `AnalisisHidrologico_Sinaloa.pdf` es el reporte de la versión original (diciembre de 2023): sus
  secciones de Gumbel, Log-Normal y posición de Weibull siguen vigentes; la de Random Forest ya no
  corresponde al código.

## Alcances

- Las curvas IDF y el hietograma se **estiman** a partir de lluvia diaria. Si hay un pluviógrafo o una
  estación automática cercana, sus datos tienen prioridad.
- Los periodos de retorno mucho mayores que el registro son extrapolaciones: revisa siempre el
  intervalo de confianza.
- El lector del archivo de texto del SMN se probó con una muestra del formato, no con una descarga
  completa; si alguna estación no abre, pásala al formato de Excel descrito arriba.

Datos: Servicio Meteorológico Nacional (SMN) y CICESE. Autor: Christian Ramos Castillo.
