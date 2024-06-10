#!pip install mglearn
#!apt-get install graphwiz
#!pip install-U statsmodels
#!pip install shap

"""**Librerias**"""

import pandas as pd #Especializada en el manejo y análisis de estructuras de datos
import numpy as np #Especializada en el cálculo numérico y el análisis de datos para grandes volumenes de datos
import seaborn as sns #Permite generar elegantes gráficos.Basada en matplotlib y proporciona una interfaz de alto nivel
import matplotlib.pyplot as plt #Especializada en la creación de gráficos en dos dimensiones
from sklearn.model_selection import train_test_split, cross_val_score #Para el entrenamiento y Cross-Validation
                                  # Modelos de Machine Learning

from sklearn.tree import DecisionTreeClassifier #importamos el algoritmo Decision Tree
from sklearn.ensemble import RandomForestClassifier #importamos el algoritmo Random Forest
from sklearn.linear_model import LogisticRegression #importamos el algoritmo Logistic Regression
from sklearn.metrics import *

from imblearn.over_sampling import SMOTE #Aplicacion del metodo de SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import shap
import warnings
# Filtrar advertencias que contengan la cadena especificada
warnings.filterwarnings("ignore", message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names")
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")



"""#Carga y análisis de los datos"""

#dataset = pd.read_csv("/content/drive/MyDrive/data.csv", sep = ';')
dataset = pd.read_csv("data_dropout.csv")
#dataset = pd.read_excel("data_dropout.xlsx")
#Mostramos el encabezado
dataset.head()

# Crear un diccionario con los nombres de las columnas actuales como claves y los nombres en español como valores
nombres_espanol = {
    'Marital status': 'Estado_Civil',
    'Application mode': 'Modo_Aplicacion',
    'Application order': 'Id_Aplicacion',
    'Course': 'Id_Curso',
    'Daytime/evening attendance': 'Horario_Asistencia',
    'Previous qualification': 'Calificacion_Previa',
    'Previous qualification (grade)': 'Calificacion_Previa_Nota',
    'Nacionality': 'Nacionalidad',
    'Mother\'s qualification': 'Calificacion_Madre',
    'Father\'s qualification': 'Calificacion_Padre',
    'Mother\'s occupation': 'Ocupacion_Madre',
    'Father\'s occupation': 'Ocupacion_Padre',
    'Admission grade': 'Nota_Admision',
    'Displaced': 'Desplazado',
    'Educational special needs': 'Necesidad_Educativa_Especial',
    'Debtor': 'Deudor',
    'Tuition fees up to date': 'Cuotas_Matricula_al_Dia',
    'Gender': 'Genero',
    'Scholarship holder': 'Beneficiario_Beca',
    'Age at enrollment': 'Edad_al_Matricularse',
    'International': 'Internacional',
    'Curricular units 1st sem (credited)': 'Unidades_Curriculares_1erSemestre_Acreditadas',
    'Curricular units 1st sem (enrolled)': 'Unidades_Curriculares_1erSemestre_Inscritas',
    'Curricular units 1st sem (evaluations)': 'Unidades_Curriculares_1erSemestre_Evaluaciones',
    'Curricular units 1st sem (approved)': 'Unidades_Curriculares_1erSemestre_Aprobadas',
    'Curricular units 1st sem (grade)': 'Unidades_Curriculares_1er Semestre_Nota',
    'Curricular units 1st sem (without evaluations)': 'Unidades_Curriculares_1er_Semestre_SinEvaluaciones',
    'Curricular units 2nd sem (credited)': 'Unidades_Curriculares_2doSemestre_Acreditadas',
    'Curricular units 2nd sem (enrolled)': 'Unidades_Curriculares_2doSemestre_Inscritas',
    'Curricular units 2nd sem (evaluations)': 'Unidades_Curriculares_2doSemestre_Evaluaciones',
    'Curricular units 2nd sem (approved)': 'Unidades_Curriculares_2doSemestre_Aprobadas',
    'Curricular units 2nd sem (grade)': 'Unidades_Curriculares_2doSemestre_Nota',
    'Curricular units 2nd sem (without evaluations)': 'Unidades_Curriculares_2doSemestre_Sin evaluaciones',
    'Unemployment rate': 'Tasa_Desempleo',
    'Inflation rate': 'Tasa_Inflacion',
    'GDP': 'PBI',
    'Target': 'Objetivo'
}

# Renombrar las columnas del DataFrame
dataset.rename(columns=nombres_espanol, inplace=True)

# Mostrar la información actualizada del DataFrame
#Definicion de las tipologias de variables que tiene la dataset
dataset.info()

#from google.colab import drive
#drive.mount('/content/drive')

#Cantidad de filas y columnas en la dataset
dataset.shape

#Analisis estadisticos de la dataset
dataset.describe()

"""#Limpieza de datos"""

# Eliminar espacios en blanco alrededor de los nombres de las columnas
#dataset.columns = dataset.columns.str.strip()

# Cuenta de valores nulos por columna
null_counts = dataset.isnull().sum()

# Gráfico de barras para mostrar valores faltantes por columna
plt.figure(figsize=(10, 6))
#sns.barplot(x=null_counts.values, y=null_counts.index, palette='viridis')
sns.barplot(x=null_counts.values, y=null_counts.index, hue=null_counts.index, palette='viridis', legend=False)
plt.title('Valores Faltantes por Columna')
plt.xlabel('Cantidad de Valores Faltantes')
plt.ylabel('Factores')
plt.show()

# Suponiendo que 'dataset' es tu DataFrame
columna = 'Id_Curso'

# Encuentra los valores nulos en la columna 'Marital Status'
valores_nulos = dataset[columna].isnull()

# Muestra los índices de las filas con valores nulos en esa columna
indices_nulos = dataset[valores_nulos].index

# Muestra las filas completas con valores nulos en esa columna
filas_nulas = dataset.loc[valores_nulos]

print(filas_nulas)

#Reemplazamos la media a los valores vacios  en la columna x
#dataset[a] = dataset[a].replace(np.nan,round(dataset[a].mean(),3))
#Verificamos que no contenga valores nulo
#np.where(np.isnan(dataset[a]))

#Observamos el comportamiento de las columnas con valores NaN
dataset[["Estado_Civil","Modo_Aplicacion","Id_Aplicacion","Id_Curso"]].astype('float').hist(figsize=(10,10),color='b')
plt.show()
plt.close()

#Reemplazamos por media
dataset["Estado_Civil"].replace(np.nan,round(dataset["Estado_Civil"].astype("float").mean(),3),inplace=True)
dataset["Modo_Aplicacion"].replace(np.nan,round(dataset["Modo_Aplicacion"].astype("float").mean(),3),inplace=True)
dataset["Id_Aplicacion"].replace(np.nan,round(dataset["Id_Aplicacion"].astype("float").mean(),3),inplace=True)
dataset["Id_Curso"].replace(np.nan,round(dataset["Id_Curso"].astype("float").mean(),3),inplace=True)

"""
##Encoding de variables"""

#Observamos que nuestra variable objetivo, osea la columna Target cuenta con 3 valores, para ello solo necesitamos que tenga 2 valores
#dato = dataset['Target'].value_counts()
#dato

#Realizamos encoding de nuestras variables que contienen Cadenas de texto
#a = 'Smoking'
#dataset['Smoking'] = dataset[a].map({'Yes':1,'No':0}).astype(int)

#Definiremos de esta manera que si el estudiante DESERTADO sera identificado por la variable '1', de lo contrario, sera '0'
p = 'Objetivo'
dataset['Outcome'] = dataset[p].map({'Dropout':1,'Graduate':0,'Enrolled':0}).astype(int)
#dataset['Outcome']

#Eliminamos las columnas que ya no necesitaremos para las modelos predictivos
dataset = dataset.drop([p], axis=1)
dataset.head()

# Extraer los nombres de los encabezados
feature_names = dataset.columns.tolist()
feature_names

"""# Análisis exploratorio de los datos


"""

#Implementamos diagrama de HEATMAP si queremos observar todas las correlaciones
plt.figure(figsize=(25,20)) #configurar el tamaño de la figura
#sns.heatmap(dataset.corr(),vmax=.8,linewidths=0.01,square=True,annot=True)
#ANNOT = para ver las etiquetas en los cuadros interiores
sns.heatmap(dataset.corr(),linewidths=0.1,square=True,annot=True, cmap='RdYlGn')

"""#**Implementación de los modelos Machine Learning**"""

# Especificar las columnas que deseas eliminar
columnas_a_eliminar = ['Horario_Asistencia','Modo_Aplicacion','Id_Aplicacion', 'Id_Curso','Nacionalidad', 'Calificacion_Madre','Calificacion_Padre','Cuotas_Matricula_al_Dia',
                       'Deudor','Internacional','Tasa_Desempleo','Tasa_Inflacion','PBI']

# Eliminar las columnas especificadas
dataset = dataset.drop(columns=columnas_a_eliminar)

# Mostrar el encabezado del dataset después de eliminar columnas
print("Encabezado después de eliminar columnas:")
dataset.info()

dataset.head()

"""##Funciones para las métricas"""

#Funcion para determinar la metrica ACCURACY
def accuracy(ConfusionMatriz):  # Method for metric
  valor=(ConfusionMatriz[0][0]+ConfusionMatriz[1][1])/np.sum(ConfusionMatriz)
  return valor

#Funcion para determinar la metrica PRECISION
def precision(ConfusionMatriz):  # Method for metric
  valor=(ConfusionMatriz[0][0])/(ConfusionMatriz[0][0]+ConfusionMatriz[0][1])
  return valor

"""La precisión es la proporción de instancias positivas predichas correctamente respecto a todas las instancias que el modelo predijo como positivas. En este contexto, indica la proporción de estudiantes predichos como no desertores que realmente no desertaron (clase 0) y la proporción de estudiantes predichos como desertores que realmente desertaron (clase 1)."""

#Funcion para determinar la metrica RECALL
def sensitivity(ConfusionMatriz):  # Method for metric
  valor=(ConfusionMatriz[0][0]+ConfusionMatriz[1][0])
  if valor != 0:
    return (ConfusionMatriz[0][0])/(ConfusionMatriz[0][0]+ConfusionMatriz[1][0])
  else:
    return 0.0

"""El recall es la proporción de instancias positivas que el modelo predijo correctamente en relación con todas las instancias positivas reales. En este contexto, indica la proporción de estudiantes reales no desertores que el modelo predijo correctamente como no desertores y la proporción de estudiantes reales desertores que el modelo predijo correctamente como desertores."""

#Funcion para determinar la metrica score-F1
def F1(Precision, Recall):  # Method for metric
  valor= (2*Recall*Precision)/(Recall+Precision)
  return valor

"""El F1-score es la media armónica entre precision y recall. Es útil cuando hay un desequilibrio entre las clases. Un F1-score alto indica un buen equilibrio entre precision y recall."""

#Función para determinar la especificidad
def specificity(ConfusionMatrix):
    true_negatives = ConfusionMatrix[1][1]  # True Negatives (TN)
    false_positives = ConfusionMatrix[1][0]  # False Positives (FP)
    return true_negatives / (true_negatives + false_positives)

"""# Modelo Predictivo A: Decision Tree

"""

target = len(dataset.iloc[0])

#Clasificamos las variables dependientes "x" e independiente "y"
x1 = dataset.iloc[:,[i for i in range(len(dataset.iloc[0])-1)]]
y1= dataset.iloc[:,target-1]

#Definimos los conjuntos de prueba(20%) y entrenamiento (80%)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,test_size=0.20,random_state=1)

#Aplicacion del metodo SMOTE
smote = SMOTE(random_state=1)
x1_train_smote, y1_train_smote = smote.fit_resample(x1_train, y1_train) #node SMOTE

#Seleccionamos y entrenamos el modelo de clasificacion
dt = DecisionTreeClassifier(max_depth=4) # Machine Learning Algorithm
dt.fit(x1_train_smote, y1_train_smote) #Nodo Learn

# Crear un objeto explainer de SHAP con los datos de entrenamiento
explainer = shap.TreeExplainer(dt)

# Obtener los valores SHAP (asegúrate de que `1` sea la clase que te interesa)
shap_values = explainer.shap_values(x1_train_smote[:min(1000, len(x1_train_smote))])

# Visualizar el resumen de SHAP values para la clase 1
#shap.summary_plot(shap_values[1], x1_train_smote[:min(1000, len(x1_train_smote))])
shap.summary_plot(shap_values[:,:,1],x1_train_smote[:min(1000, len(x1_train_smote))])

#print("Entrenamiento", x1_train_smote.shape)
#print("Forma de los valores SHAP:", shap_values.shape)
print("Clases:", np.unique(y1_train_smote))
#print("Metodo Shap", np.unique(shap_values))
#print("Nombres de las características:", feature_names)

#Predecimos con conjunto de prueba
y1_prediction = dt.predict(x1_test)
#y1_prediction

#Evaluamos el modelo despues de aplicar SMOTE
print("Metricas para el modelo Decision Tree")
matriz_confusion1 = confusion_matrix(y1_test,y1_prediction)
print(matriz_confusion1)
print("Accuracy: ", accuracy(matriz_confusion1))
print("Precision: ", precision(matriz_confusion1))
print("F1: ", F1(precision(matriz_confusion1),sensitivity(matriz_confusion1)))
print("Sensitivity: ", sensitivity(matriz_confusion1))
print("Specifity: ", specificity(matriz_confusion1))

#disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion1, display_labels=dt.classes_)
#disp.plot()
#plt.show

# Realizar validación cruzada en el conjunto de entrenamiento con SMOTE
accuracy_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='accuracy')
precisions_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='precision')
recall_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='recall')
f1_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='f1')

# Obtener la precisión promedio de la validación cruzada
average_accuracy_1 = accuracy_cv.mean()
average_precision_1 = precisions_cv.mean()
average_recall_1 = recall_cv.mean()
average_f1_1 = f1_cv.mean()

print("Accuracy promedio de la validación cruzada:", average_accuracy_1)
print("Precision promedio de la validación cruzada:", average_precision_1)
print("Recall promedio de la validación cruzada:", average_recall_1)
print("F1 promedio de la validación cruzada:", average_f1_1)

# Imprimir el informe de clasificación
#print("Informe de clasificación en el conjunto de prueba:\n", classification_report(y1_test, y1_prediction))

"""# Modelo Predictivo B: Random Forest"""

#Clasificamos las variables dependientes "x" e independiente "y"
x4 = dataset.iloc[:,[i for i in range(len(dataset.iloc[0])-1)]]
y4 = dataset.iloc[:,target-1]

#Definimos los conjuntos de prueba(20%) y entrenamiento (80%)
x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4,test_size=0.20,random_state=1)

#Aplicación del método SMOTE
smote = SMOTE(random_state=1)
x4_train_smote, y4_train_smote = smote.fit_resample(x4_train, y4_train) #node SMOTE

#Seleccionamos y entrenamos el modelo de clasificación
rf = RandomForestClassifier()
rf.fit(x4_train_smote,y4_train_smote)

# Definir el espacio de búsqueda de parámetros
param_grid = {
    'n_estimators': [10,20],
    'max_depth': [5,10],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

# Aplicar la búsqueda de cuadrícula para optimizar los parámetros
grid_search = GridSearchCV(rf, param_grid, cv=5, refit=True) # cv = numero de iteraciones
# Realizar el ajuste del modelo con GridSearchCV en los datos de entrenamiento
grid_search.fit(x1_train_smote, y1_train_smote)

# Obtener los mejores parámetros encontrados
best_params = grid_search.best_params_
print("Mejores parámetros para Random  Forest:", best_params)

# Obtener el mejor modelo entrenado con los mejores hiperparámetros
best_model = grid_search.best_estimator_

#Algoritmo SHAP
# Crear un objeto explainer de SHAP con los datos de entrenamiento
explainer = shap.TreeExplainer(rf)

# Obtener los valores SHAP (asegúrate de que `1` sea la clase que te interesa)
shap_values = explainer.shap_values(x4_train_smote[:min(1000, len(x4_train_smote))])

# Visualizar el resumen de SHAP values
shap.summary_plot(shap_values[:,:,1], x4_train_smote[:min(1000, len(x4_train_smote))])

#Predecimos con conjunto de prueba
#y4_prediction = rf.predict(x4_test)
y4_prediction = grid_search.predict(x4_test)
#y4_prediction

print("Metricas para el modelo Random Forest")
matriz_confusion4 = confusion_matrix(y4_test,y4_prediction)
print(matriz_confusion4)
print("Accuracy: ", accuracy(matriz_confusion4))
print("Precision: ", precision(matriz_confusion4))
print("F1: ", F1(precision(matriz_confusion4),sensitivity(matriz_confusion4)))
print("Sensitivity: ", sensitivity(matriz_confusion4))
print("Specifity: ", specificity(matriz_confusion4))
#disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion4, display_labels=svm.classes_)
#disp.plot()
#plt.show

#disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion4, display_labels=dt.classes_)
#disp.plot()
#plt.show

# Realizar validación cruzada en el conjunto de entrenamiento con SMOTE
accuracy_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='accuracy')
precisions_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='precision_macro')
recall_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='recall_macro')
f1_cv = cross_val_score(dt, x1_train_smote, y1_train_smote, cv=5, scoring='f1_macro')

# Obtener la precisión promedio de la validación cruzada
average_accuracy_4 = accuracy_cv.mean()
average_precision_4 = precisions_cv.mean()
average_recall_4 = recall_cv.mean()
average_f1_4 = f1_cv.mean()

print("Accuracy promedio de la validación cruzada:", average_accuracy_4)
print("Precision promedio de la validación cruzada:", average_precision_4)
print("Sensitivity promedio de la validación cruzada:", average_recall_4)
print("F1-score promedio de la validación cruzada:", average_f1_4)

"""# Modelo Predictivo C: Logistic Regression"""

#Clasificamos las variables dependientes "x" e independiente "y"
x5 = dataset.iloc[:,[i for i in range(len(dataset.iloc[0])-1)]]
y5 = dataset.iloc[:,target-1]

#Definimos los conjuntos de prueba(20%) y entrenamiento (80%)
x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.20, random_state=1)

#Aplicacion del metodo SMOTE
smote = SMOTE(random_state=1)
x5_train_smote, y5_train_smote = smote.fit_resample(x5_train, y5_train) #node SMOTE

# Estándar: Crear y estandarizar el conjunto de entrenamiento
scaler = StandardScaler()
x5_train_smote_standardized = scaler.fit_transform(x5_train_smote)
x5_test_standardized = scaler.transform(x5_test)

#Seleccionamos y entrenamos el modelo de clasificacion
lregression = LogisticRegression()
lregression.fit(x5_train_smote_standardized,y5_train_smote)

#Algoritmo SHAP
# Crear un objeto explainer de SHAP con los datos de entrenamiento
explainer = shap.LinearExplainer(lregression, x5_train_smote_standardized)

# Obtener los valores SHAP
shap_values = explainer.shap_values(x5_train_smote[:min(1000, len(x5_train_smote))])

# Visualizar el resumen de SHAP values
shap.summary_plot(shap_values, x5_train_smote[:min(1000, len(x5_train_smote))], feature_names=x5_train_smote.columns)

#Predecimos con conjunto de prueba
y5_prediction = lregression.predict(x5_test_standardized)
#y5_prediction

#Evaluamos el algoritmo despues del SMOTE
print("Metricas para el modelo Logistic Regression")
matriz_confusion5 = confusion_matrix(y5_test,y5_prediction)
print(matriz_confusion5)
print("Accuracy: ", accuracy(matriz_confusion5))
print("Precision: ", precision(matriz_confusion5))
print("F1: ", F1(precision(matriz_confusion5),sensitivity(matriz_confusion5)))
print("Sensitivity: ", sensitivity(matriz_confusion5))
print("Specifity: ", specificity(matriz_confusion5))

#disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion5, display_labels=dt.classes_)
#disp.plot()
#plt.show

# Realizar validación cruzada en el conjunto de entrenamiento con SMOTE
accuracy_cv = cross_val_score(dt, x5_train_smote, y5_train_smote, cv=5, scoring='accuracy')
precisions_cv = cross_val_score(dt, x5_train_smote, y5_train_smote, cv=5, scoring='precision')
recall_cv = cross_val_score(dt, x5_train_smote, y5_train_smote, cv=5, scoring='recall')
f1_cv = cross_val_score(dt, x5_train_smote, y5_train_smote, cv=5, scoring='f1')

# Obtener la precisión promedio de la validación cruzada
average_accuracy_5 = accuracy_cv.mean()
average_precision_5 = precisions_cv.mean()
average_recall_5 = recall_cv.mean()
average_f1_5 = f1_cv.mean()

print("Accuracy promedio de la validación cruzada:", average_accuracy_5)
print("Precision promedio de la validación cruzada:", average_precision_5)
print("Recall promedio de la validación cruzada:", average_recall_5)
print("F1 promedio de la validación cruzada:", average_f1_5)

#disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion4, display_labels=svm.classes_)
#disp.plot()
#plt.show

"""#**Resultados**

##Resultados de la métrica Accuracy

EVALUACION DE LOS MODELOS DESPUES DE USAR SMOTE
"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT', 'RF', 'LR']
valores = [accuracy(matriz_confusion1), accuracy(matriz_confusion4), accuracy(matriz_confusion5)]

# Crear la gráfica de barras
plt.bar(nombres, valores, color='#8B0000')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de Accuracy (%)')

# Establecer el rango del eje de ordenadas
plt.ylim(0.3, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores)):
    etiqueta = round(valores[i]*100, 3)
    plt.text(i, valores[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""Resultados después Cross-Validation"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT','RF', 'LR']
valores1 = [average_accuracy_1, average_accuracy_4, average_accuracy_5]

# Crear la gráfica de barras
plt.bar(nombres, valores1, color='#8B0000')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de Accuracy después de Cross-Validation')

# Establecer el rango del eje de ordenadas
plt.ylim(0.2, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores1)):
    etiqueta = round(valores1[i]*100, 3)
    plt.text(i, valores1[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""##Resultados de la métrica Precision

Resultados despues de usar SMOTE
"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT', 'RF', 'LR']
valores = [precision(matriz_confusion1), precision(matriz_confusion4), precision(matriz_confusion5)]

# Crear la gráfica de barras
plt.bar(nombres, valores, color='#005c8b')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de Precision (%)')

# Establecer el rango del eje de ordenadas
plt.ylim(0.3, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores)):
    etiqueta = round(valores[i]*100, 3)
    plt.text(i, valores[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""Resultados después Cross-Validation"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT','RF', 'LR']
valores1 = [average_precision_1, average_precision_4, average_precision_5]

# Crear la gráfica de barras
plt.bar(nombres, valores1, color='#005c8b')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de Precision después de Cross-Validation')

# Establecer el rango del eje de ordenadas
plt.ylim(0.2, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores1)):
    etiqueta = round(valores1[i]*100, 3)
    plt.text(i, valores1[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""##Resultados de la métrica F1

Resultados despues de usar SMOTE
"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT', 'RF', 'LR']
valores = [F1(precision(matriz_confusion1),sensitivity(matriz_confusion1)), F1(precision(matriz_confusion4),sensitivity(matriz_confusion4)), F1(precision(matriz_confusion5),sensitivity(matriz_confusion5))]

# Crear la gráfica de barras
plt.bar(nombres, valores, color='#1c542d')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de Score F1 (%)')

# Establecer el rango del eje de ordenadas
plt.ylim(0.3, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores)):
    etiqueta = round(valores[i]*100, 3)
    plt.text(i, valores[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""Resultados después Cross-Validation"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT','RF', 'LR']
valores1 = [average_f1_1, average_f1_4, average_f1_5]

# Crear la gráfica de barras
plt.bar(nombres, valores1, color='#1c542d')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de F1 después de Cross-Validation')

# Establecer el rango del eje de ordenadas
plt.ylim(0.2, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores1)):
    etiqueta = round(valores1[i]*100, 3)
    plt.text(i, valores1[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""##Resultados de la Sensitivity"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT', 'RF', 'LR']
valores = [sensitivity(matriz_confusion1), sensitivity(matriz_confusion4), sensitivity(matriz_confusion5)]

# Crear la gráfica de barras
plt.bar(nombres, valores, color='#d6ae01')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de Sensitivity (%)')

# Establecer el rango del eje de ordenadas
plt.ylim(0.3, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores)):
    etiqueta = round(valores[i]*100, 3)
    plt.text(i, valores[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""##Resultados de la Specifity"""

#Definimos las nombre de las etiquetas y los valores del eje de abscisas
nombres = ['DT', 'RF', 'LR']
valores = [specificity(matriz_confusion1), specificity(matriz_confusion4), specificity(matriz_confusion5)]

# Crear la gráfica de barras
plt.bar(nombres, valores, color='#8673a1')

# Personalizar el gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resultados de Specifity (%)')

# Establecer el rango del eje de ordenadas
plt.ylim(0.3, 1.0)

# Agregar etiquetas de datos
for i in range(len(valores)):
    etiqueta = round(valores[i]*100, 3)
    plt.text(i, valores[i], str(etiqueta), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""#**Predicciones**"""

import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LogisticRegression

# Crear la ventana principal
root = tk.Tk()
root.title("Predicción de Deserción Estudiantil")
root.configure(background="#1EAE98")  # Color de fondo

# Crear un frame para el formulario
form_frame = ttk.Frame(root, padding=(20, 10))
form_frame.pack()

# Crear los widgets del formulario dinámicamente
widgets_data = [
    {'widget': ttk.Label, 'description': 'Estado Civil:'},
    {'widget': ttk.Label, 'description': 'Calificación Prevía:'},
    {'widget': ttk.Label, 'description': 'Nota de Calificación Prevía:'},
    {'widget': ttk.Label, 'description': 'Ocupación de la Madre:'},
    {'widget': ttk.Label, 'description': 'Ocupación del Padre:'},
    {'widget': ttk.Label, 'description': 'Nota de Admisión:'},
    {'widget': ttk.Label, 'description': '¿Desplazado?:'},
    {'widget': ttk.Label, 'description': '¿Necesidad de Educación Especial?:'},
    {'widget': ttk.Label, 'description': 'Género:'},
    {'widget': ttk.Label, 'description': '¿Beneficiario de Beca?:'},
    {'widget': ttk.Label, 'description': 'Edad al Matricularse:'},
    {'widget': ttk.Label, 'description': 'UC 1º Semestre Acreditadas:'},
    {'widget': ttk.Label, 'description': 'UC 1º Semestre Inscritas:'},
    {'widget': ttk.Label, 'description': 'UC 1º Semestre Evaluaciones:'},
    {'widget': ttk.Label, 'description': 'UC 1º Semestre Aprobadas:'},
    {'widget': ttk.Label, 'description': 'UC 1º Semestre Nota:'},
    {'widget': ttk.Label, 'description': 'UC 1º Semestre Sin Evaluaciones:'},
    {'widget': ttk.Label, 'description': 'UC 2º Semestre Acreditadas:'},
    {'widget': ttk.Label, 'description': 'UC 2º Semestre Inscritas:'},
    {'widget': ttk.Label, 'description': 'UC 2º Semestre Evaluaciones:'},
    {'widget': ttk.Label, 'description': 'UC 2º Semestre Aprobadas:'},
    {'widget': ttk.Label, 'description': 'UC 2º Semestre Nota:'},
    {'widget': ttk.Label, 'description': 'UC 2º Semestre Sin Evaluaciones:'},
]

widgets_list = []
for i, widget_data in enumerate(widgets_data):
    label = widget_data['widget'](form_frame, text=widget_data['description'], font=('Helvetica', 10, 'bold'), background="#1EAE98", foreground="white")  # Colores de texto y fondo
    label.grid(row=i, column=0, sticky="W", padx=5, pady=5)
    widget = ttk.Entry(form_frame)
    widget.grid(row=i, column=1, sticky="EW", padx=5, pady=5)
    widgets_list.append(widget)

def hacer_prediccion():
    global lregression  # Acceder a la variable global lregression
    try:
        # Obtener los valores de los widgets
        cadena = [widget.get() for widget in widgets_list]
        # Hacer la predicción
        prediction = lregression.predict([cadena])
        # Actualizar la etiqueta de resultado con la predicción
        if prediction == 1:
            resultado_label.config(text="Predicción: Dropout", foreground="#FF5733")
        else:
            resultado_label.config(text="Predicción: Graduate o Enrolled", foreground="#33FF5C")
    except AttributeError:
        resultado_label.config(text="Error: Modelo no ajustado", foreground="#FF5733")

# Botón para realizar la predicción
predict_button = ttk.Button(root, text="Realizar Predicción", command=hacer_prediccion)
predict_button.pack(pady=10)

# Etiqueta para mostrar el resultado de la predicción
resultado_label = ttk.Label(root, text="", font=('Helvetica', 12, 'bold'), background="#1EAE98")  # Color de fondo
resultado_label.pack(pady=10)

# Modelo de regresión logística (sustituye con tu modelo entrenado)
lregression = LogisticRegression()

# Ejecutar la aplicación
root.mainloop()

