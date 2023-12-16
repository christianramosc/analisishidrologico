# -*- coding: utf-8 -*-
"""
@author: Ramos Castillo Christian
"""

##
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
import multiprocessing


#Nombre de tu archivo

nombre = 'Estaciones24h'
hojas1 = pd.ExcelFile(nombre + str('.xlsx'))
hojas = hojas1.sheet_names


#hoja = input('Hoja : ')


#Lector de archivos en formato excel


for sheetnames in hojas:

    
    print ('Procesando la estación : ' + sheetnames)
    
    est1 = pd.read_excel(nombre + str('.xlsx'), sheet_name= sheetnames )

    # Eliminar todos los null

    est1.dropna(inplace=True)

    tabla = pd.pivot_table(est1, values = ['Datos'], index = ['Año'], aggfunc = np.max ,columns = ['Mes'])

    asd = tabla.reset_index()

    asd.columns = ['Año','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    asd1 = asd.set_index('Año')

    asd1.dropna(inplace=True, thresh = 9)

    asd2 = asd1.reset_index()

    mies  = asd2['Año'].reset_index()

    mies2 = mies.drop(columns = 'index')
        
    # Obteniendo las precipiaciones máximas anuales diarias

    hpmax0 = est1.groupby(['Año']).max()['Datos'].reset_index()

    hpmax= mies2.merge(hpmax0, how = 'inner')

        # Meses con mayor frecuencia en precipitacion

    
    mesesmax = est1.groupby(['Mes']).max('Mes')['Datos'].reset_index()

    columnas = ['Año', 'Hpmax (mm)']

        # Estableciendo una nueva lista con los nuevos valores

    hpmax.columns = columnas

        #Espacio en blanco dejado por estética

    hpmax[''] = ''

        #Uniendo la tabla de precipitación máxima anual diaria con la mensual histórica

    tablas = hpmax.join(mesesmax).fillna(' ')

    tablaord = tablas.sort_values(by ='Hpmax (mm)')

    #Eliminando lo que ya no sirve de información

    del hpmax, mesesmax, columnas

    #Creando las tablas de frecuencias

    maximo = tablas['Hpmax (mm)'].max()

    minimo = tablas['Hpmax (mm)'].min()

    numdatostotales = len(tablas['Año'])

    rango = maximo - minimo

    nclas = 12

    tablas[' '] = ' '

    anchoclase= rango/nclas

        #############################################################
    tablas['Separador 1'] = '|'

    tablas['N max'] = pd.Series(maximo)

    tablas['N min'] = pd.Series(minimo)

    tablas['Rango'] = pd.Series(rango)

    tablas[''] = ''

    tablas['Ancho de clase'] = pd.Series(anchoclase)

    tablas[''] = ''

        # EL RANGO NO CAMBIA, EL RANGO ES DE 12

    tablas['N Clase'] = pd.Series(range (1,nclas+1))

    clasesyevent = tablas['Hpmax (mm)'].value_counts(bins = 12).reset_index()

    clasesyevent.columns = ['Clases','Eventos']

    tablas = tablas.join(clasesyevent)

    tablas['Frecuencia Relativa'] = tablas['Eventos']/numdatostotales

    tablas['Frecuencia Acumulada'] = tablas['Frecuencia Relativa'].cumsum()
    
    

    ##GRAFICAR HISTOGRAMA DE FRECUENCIA RELATIVA Y ACUMULADA#####
    # df= tablas[['N Clase','Eventos','Clases','Frecuencia Relativa','Frecuencia Acumulada',]].dropna()
    

    
    # relat = tablas['Frecuencia Relativa']
    # #df= tablas[['N Clase','Eventos','Clases','Frecuencia Relativa','Frecuencia Acumulada',]].dropna()
    # # Crear un histograma
    # relat.plot(kind = 'bar')
    # plt.plot(relat,'m--')
    # # Configurar etiquetas y título
    # plt.title('Histograma de Distribución Relativa ' + sheetnames)
    # plt.xlabel('Clases')
    # plt.ylabel('Frecuencia Relativa')
    
    
    # plt.savefig("Histograma de Frecuencia Relativa " + sheetnames)
    
    # plt.figure()
    
    # acum = tablas['Frecuencia Acumulada']
    # #df= tablas[['N Clase','Eventos','Clases','Frecuencia Relativa','Frecuencia Acumulada',]].dropna()
    # # Crear un histograma
    # acum.plot(kind = 'bar')
    # # Configurar etiquetas y título
    # plt.title('Histograma de Distribución Acumulada ' + sheetnames)
    # plt.ylabel('Frecuencia Acumulada')
    
    # # Mostrar la gráfica
    
    # plt.savefig("Histograma de Frecuencia Acumulada " + sheetnames)
    
    # plt.figure()
    
    ##############################################################
    tablas['  '] ='  '

    tablas['m'] = pd.Series(range(1, numdatostotales + 1))

    tablas['Hpmax (mm) Ordenada']= tablas['Hpmax (mm)']

    hpmaxord = tablas['Hpmax (mm) Ordenada'].reset_index(drop = True).sort_values(ignore_index = True, ascending = False )

    del tablas['Hpmax (mm) Ordenada']

    media= hpmaxord.mean()

    desvest= np.std(hpmaxord)

    beta = media-0.45*desvest

    alfa = 1.2825/desvest

    kurt = stats.kurtosis(hpmaxord)

    skw = stats.skew(hpmaxord)

    tablas['Hpmax Ordenada (mm)'] = pd.Series(hpmaxord)

    tablas['Tr'] = (numdatostotales+1)/tablas['m']

    tablas['Probabilidad de excendencia'] = 1/tablas['Tr']

    tablas['Probabilidad de no excendencia'] = 1- tablas['Probabilidad de excendencia']

     ##############################################################
    
    #tablas[['m','Hpmax Ordenada (mm)','Tr','Probabilidad de excendencia','Probabilidad de no excendencia']].astype(float)

    #tablas[['m']].astype(int)

    #df = tablas[['m','Hpmax Ordenada (mm)','Tr','Probabilidad de excendencia','Probabilidad de no excendencia']].round(3)




    tablas['Separador 3'] = tablas['Separador 1']
        
    tablas['Desviación Estándar'] = pd.Series(desvest)
    
    tablas['Alfa'] = pd.Series(alfa)
    
    tablas['Beta'] = pd.Series(beta)
    
    tablas['Media'] = pd.Series(media)    
    
    tablas['Kurtosis'] = pd.Series(kurt)
    
    tablas['Coeficiente de asimetría'] = pd.Series(skw)
    
        ############################################################
    tr2  = tablas['Tr']

    tablas['Precipitaciones Calculadas'] = beta - 1/(alfa)*np.log(-np.log((tr2-1)/(tr2)))

    precip = tablas['Precipitaciones Calculadas']

    xobx = hpmaxord - precip

    tablas['Separador 6'] = tablas['Separador 1']

    tablas['(Xobs - Xcalc)^2'] = (xobx)**2

    tablas['Error Cuadrático Mínimo (ECM) Gumbell']  = pd.Series(mean_squared_error(tablas['Hpmax Ordenada (mm)'], tablas['Precipitaciones Calculadas'],squared=False ))
    
    
    tablas['Hpmax Observada']=tablas['Hpmax Ordenada (mm)']
    tablas['Hpmax Calculadas Gumbel'] =tablas['Precipitaciones Calculadas']

    #plt.scatter(tr2, hpmaxord, color = 'black')
    # plt.xlabel('Periodo de retorno')
    # plt.ylabel('Precipitaciones máximas anuales (mm)')
    # plt.title('Curva de frecuencia hp máxima anual en 24h')
    # plt.show()
    
    #############################################################
    
    tablas['Gumbel calculados'] = tablas['Separador 1']
    
    tablas['Análisis de periodo de retorno (años) '] = pd.Series([50, 75, 100,200,500,750,1000])
        
    trx = tablas['Análisis de periodo de retorno (años) ']
        
    tablas['Inferencia de precipitaciones (Precipitaciones de diseño) en los eventos propabilisticos(hpmax, Tr = x) (mm)'] = beta - 1/(alfa)*np.log(-np.log((trx-1)/(trx)))

    xcalcgumb = tablas['Inferencia de precipitaciones (Precipitaciones de diseño) en los eventos propabilisticos(hpmax, Tr = x) (mm)'] 

    tablas['Datos Observados'] = pd.Series(hpmaxord)
    
    tablas['Datos Calculados'] = precip
    
    tablas['Periodo de retorno (Tr)'] = tablas['Tr']
    
    # plt.plot(tr2,precip,label = 'Método Gumbell', color = 'orange')

    # plt.plot(trx,xcalcgumb, color = 'red')
    
    # plt.scatter([tr2], [hpmaxord], color = 'black', label = 'Datos Observados')
    
    # plt.legend()
    # plt.xlabel('Periodo de retorno')
    # plt.ylabel('Precipitaciones máximas anuales (mm)')
    # plt.title('Análisis de frecuencias con Método Gumbell')
    # plt.show()
    
    ###GRAFICAR####
    
    tablas['Gumbel calculados 1'] = tablas['Separador 1']
    #############################################################
    
    tablas['Distribución LogNormal'] = '####'
    
    
    
    tablas['Ln(x)'] = np.log(tablas['Hpmax Ordenada (mm)'])

    alfaln = np.mean(tablas['Ln(x)'])

    tablas['Alfa LogNormal'] = pd.Series(alfaln)


    tablas['(Ln(x) - alfa )^2'] = (tablas['Ln(x)'] - alfaln)**2

    betaln = (np.mean(tablas['(Ln(x) - alfa )^2']))**0.5

    tablas['Beta LogNormal'] = pd.Series(betaln)

    tablas['Análisis de periodo de retorno (años) (LogNormal)'] = pd.Series([50, 75,100,200,500,750,1000])

    trlog = tablas['Análisis de periodo de retorno (años) (LogNormal)']

    tablas['Probabilidad de NO excedencia (hp < x) LogNormal'] = (trlog-1)/trlog
    tablas['Probabilidad de Excedencia (hp > x) LogNormal'] = 1- tablas['Probabilidad de NO excedencia (hp < x) LogNormal']

    probnoexlog= tablas['Probabilidad de NO excedencia (hp < x) LogNormal']

    tablas['XTr (Precipitaciones de diseño LogNormal)'] = stats.lognorm(betaln,scale = np.exp(alfaln)).ppf(probnoexlog)

    tablas['Distribución LogNormal CALCULADOS'] = '####'


    tablas['Periodo de retorno (Tr) LogNormal'] = tablas['Tr']

    tablas['Probabilidad de excendencia LogN'] = 1/tablas['Tr']
    tablas['Probabilidad de no excendencia LogN'] = 1- tablas['Probabilidad de excendencia']
    tablas['Datos Observados LogNormal'] = pd.Series(hpmaxord)
    tablas['Precipitaciones Calculadas LogNormal'] = stats.lognorm(betaln,scale = np.exp(alfaln)).ppf(tablas['Probabilidad de no excendencia LogN'])

    calclog = tablas['Precipitaciones Calculadas LogNormal']

    xobx2 = hpmaxord - calclog

    xtrlog = tablas['XTr (Precipitaciones de diseño LogNormal)']

    tablas['(Xobs - Xcalc)^2 Log Normal'] = (xobx2)**2

    tablas['Error Cuadrático Mínimo (ECM) Lognormal ']  = pd.Series(mean_squared_error(tablas['Hpmax Observada'], tablas['Precipitaciones Calculadas LogNormal'],squared=False ))

    tablas['Calculo RandomForestRegression'] = '##'
    #MODELO REGRESION##### RANDOM FOREST ##
    
    df = tablas[['Tr','Hpmax Observada']]
    
    #df2 = tablas[['Tr','Hpmax Observada','Hpmax Calculadas Gumbel','Precipitaciones Calculadas LogNormal']].dropna()



    X = df[['Tr']]
    y = df['Hpmax Observada']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
    
    # Define el modelo
    rf_regressor = RandomForestRegressor()
    
    #max_depth= 20, min_samples_leaf = 1, min_samples_split= 2, n_estimators= 10
    
    # Define el espacio de búsqueda para los hiperparámetros
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Configura la métrica a maximizar (r2_score)
    scorer = make_scorer(r2_score)
    
    # Realiza la búsqueda exhaustiva de hiperparámetros
    grid_search = GridSearchCV(rf_regressor, param_grid=param_grid, scoring=scorer, cv=5)
    
    # Ajusta el modelo a los datos
    grid_search.fit(X_train, y_train)
    
    # Muestra los mejores hiperparámetros y el rendimiento asociado
    #print("Mejores Hiperparámetros:", grid_search.best_params_)
    # Obtiene los mejores hiperparámetros
    best_params = grid_search.best_params_
    
    # Crea un nuevo modelo RandomForestRegressor con los mejores hiperparámetros
    best_rf_regressor = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf']
    )
    
    # Ajusta el nuevo modelo con los mejores hiperparámetros a los datos de entrenamiento
    best_rf_regressor.fit(X_train, y_train)
    
    # Realiza predicciones en el conjunto de prueba
    #y_pred = best_rf_regressor.predict(X_test)
    
    # Evalúa el rendimiento del modelo final
    #final_r2_score = r2_score(y_test, y_pred)
    
    #print("Rendimiento final del modelo (r2_score):", final_r2_score)
    
    
    
    df['Hpmax Calculada RandomForestRegression'] = best_rf_regressor.predict(df[['Tr']])
    
    
    # Calcular el MSE
    #mse = mean_squared_error(df['Hpmax Observada'], df['Hpmax Calculada RandomForestRegression'],squared=False)
    
    # Imprimir el resultado
    #print(f'Mean Squared Error: {mse}')
    
    
    
    #GRAFICO FINAL DE LOS 3 MODELOS
    
    y=tablas['Hpmax Ordenada (mm)']
    x=tablas['Tr']
    
    df2=tablas[['Tr','Hpmax Observada','Hpmax Calculadas Gumbel','Precipitaciones Calculadas LogNormal']].dropna()

    df2 = pd.merge(df2, df[['Tr', 'Hpmax Calculada RandomForestRegression']])


    
    ##GRAFICAR COMPARACIÓN DE MODELOS APLICADOS
    ygumbel=df2['Hpmax Calculadas Gumbel']
    ylognormal=df2['Precipitaciones Calculadas LogNormal']
    yrandomforest=df2['Hpmax Calculada RandomForestRegression']
    
    plt.figure()
    
    plt.scatter(x, y, label='Observado', color='black')
    
    plt.plot(x, ygumbel, label='Predicción Gumbel')
    plt.plot(x, ylognormal, label='Predicción LogNormal')
    plt.plot(x, yrandomforest, label='Predicción RandomForest')
    
    plt.xlabel('Periodo de retorno')
    
    plt.ylabel('Precipitaciones máximas anuales (mm)')
    
    plt.title('Comparación de resultados en el análisis de frecuencias '+ sheetnames)
    
    plt.legend()
    
    plt.savefig("Gráfica de comparación de modelos de estación " + sheetnames)
    
    ####

    tablas['Tr '] =tablas['Tr']
    
    tablas['Precipitaciones Observadas'] = tablas['Hpmax Ordenada (mm)']
    
    tablas['Precipitaciones Calculadas RandomForestRegression'] = df['Hpmax Calculada RandomForestRegression']
    
    tablas['Error Cuadrático Mínimo (ECM) RandomForestRegression ']  = pd.Series(mean_squared_error(tablas['Hpmax Observada'], tablas['Precipitaciones Calculadas RandomForestRegression'],squared=False ))
    
        
        
        
    ##################FINAL##############
    ## ESTILIZANDO##
    
    output= tablas.style.background_gradient(cmap = 'viridis')
    
    
    ext = '.xlsx'
    
    #exc1 = input ('ARCHIVO GENERADO, PORFAVOR, INSERTA EL NOMBRE CON EL QUE QUIERES GUARDAR TU ARCHIVO: ')
    
    salida = sheetnames + ' Resultado'
    
    
    output.to_excel( salida + str(ext), sheet_name= "Hoja1")
    
    
print ('Proceso finalizado')
    
    
    # # salida = input ('TABLA DINAMICA POR MES Y AÑO, GUARDAR CON NOMBRE : ')
    
    
    
    
    
    
    # # tabla.to_excel(salida + str('.xlsx'))
