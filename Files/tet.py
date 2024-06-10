import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LogisticRegression

def hacer_prediccion():
    cadena = [widget.get() for widget in widgets_list]
    prediction = lregression.predict([cadena])
    if prediction == 1:
        resultado_label.config(text="Predicción: Dropout", foreground="#FF5733")
    else:
        resultado_label.config(text="Predicción: Graduate o Enrolled", foreground="#33FF5C")

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
