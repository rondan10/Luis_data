import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Crea la ventana principal
root = tk.Tk()
root.title("Predicción de Egresos")
root.configure(background="#1EAE98")

# Define los campos del formulario
fields = [
    "Estado Civil:",
    "Calificación Prevía:",
    "Nota de Calificación Prevía:",
    "Ocupación de la Madre:",
    "Ocupación del Padre:",
    "Nota de Admisión:",
    "¿Desplazado?:",
    "¿Necesidad de Educación Especial?:",
    "Género:",
    "¿Beneficiario de Beca?:",
    "Edad al Matricularse:",
    "UC 1º Semestre Acreditadas:",
    "UC 1º Semestre Inscritas:",
    "UC 1º Semestre Evaluaciones:",
    "UC 1º Semestre Aprobadas:",
    "UC 1º Semestre Nota:",
    "UC 1º Semestre Sin Evaluaciones:",
    "UC 2º Semestre Acreditadas:",
    "UC 2º Semestre Inscritas:",
    "UC 2º Semestre Evaluaciones:",
    "UC 2º Semestre Aprobadas:",
    "UC 2º Semestre Nota:",
    "UC 2º Semestre Sin Evaluaciones:"
]

# Función para hacer la predicción
def hacer_prediccion():
    try:
        cadena = [float(entry.get()) for entry in entry_fields]
        # Aquí va tu lógica de predicción
        prediction = lregression.predict([cadena])
        if prediction == 1:
            messagebox.showinfo("Predicción", "Predicción: Dropout")
        else:
            messagebox.showinfo("Predicción", "Predicción: Graduate o Enrolled")
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores numéricos en todos los campos.")

# Crea los campos de entrada del formulario
entry_fields = []
max_label_width = max(len(field) for field in fields)  # Calcula el ancho máximo de las etiquetas
for field in fields:
    row = ttk.Frame(root)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label = ttk.Label(row, width=max_label_width + 1, text=field, anchor='w', background="#1EAE98", foreground="white")  # Ajusta el ancho de la etiqueta
    entry = ttk.Entry(row)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    label.pack(side=tk.LEFT)
    entry_fields.append(entry)

# Crea el botón para realizar la predicción
submit_button = ttk.Button(root, text="Predecir", command=hacer_prediccion)
submit_button.pack(side=tk.TOP, padx=5, pady=5)

# Inicia el bucle principal de la ventana
root.mainloop()
