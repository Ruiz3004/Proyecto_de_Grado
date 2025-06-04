# INFLUENCIA DE LOS MUROS DE MAMPOSTERÍA EN EL PERIODO FUNDAMENTAL DE VIBRACIÓN DE PÓRTICOS DE CONCRETO REFORZADO CON SECCIONES GRUESAS Y FISURADAS

**Autores:**  
Juan David Ruiz Ospina  
Víctor Alfonso Reino Beltrán  

**Trabajo de grado presentado para optar al título de Ingeniero Civil**  

**Asesor:**  
John Esteban Ardila González, Magíster (MSc) en Ingeniería Civil  

**Universidad Pontificia Bolivariana**  
Escuela de Ingenierías y Arquitectura  
Programa de Ingeniería Civil  
Montería, 2025  

---

## 📘 Descripción

Este repositorio contiene el código fuente en Python utilizado para realizar el análisis modal espectral de pórticos de concreto reforzado, considerando el efecto de los muros de mampostería y la fisuración de las secciones.

El objetivo principal es evaluar cómo varía el **periodo fundamental de vibración** en función del estado de rigidez de los elementos estructurales, comparando modelos con secciones **gruesas** (sin fisura) y **fisuradas**, tanto con como sin la influencia de muros de mampostería.

---

## 🧠 Contenido

- 📎 `Analisis_Modal_Espectral.py`: Script principal que realiza el análisis estructural utilizando OpenSeesPy.
- 🏗️ Modelos estructurales que consideran diferentes configuraciones:
  - Pórticos sin muros de mampostería.
  - Pórticos con muros de mampostería.
  - Secciones estructurales fisuradas y no fisuradas.
- 📈 Resultados comparativos de periodos de vibración para cada configuración analizada.

---

## ▶️ Ejecución

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Ruiz3004/Proyecto_de_Grado.git
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd Proyecto_de_Grado
   ```
3. Asegúrate de tener un entorno Python compatible (recomendado: Anaconda) y las dependencias instaladas.
4. Ejecuta el script principal:
   ```bash
   python Analisis_Modal_Espectral.py
   ```

---

## 🛠️ Requisitos

- Python 3.8 o superior
- [OpenSeesPy](https://openseespydoc.readthedocs.io/)
- Numpy
- Matplotlib

Puedes instalar las dependencias necesarias utilizando pip:
```bash
pip install openseespy numpy matplotlib
```

---

## 📬 Contacto

Para dudas o sugerencias, puedes escribirnos:

- 📧 juan.ruiz@upb.edu.co  
- 📧 victor.reino@upb.edu.co  

¡Gracias por tu interés en nuestro trabajo!

---
