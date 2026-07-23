# 🧠 Mi Primer RAG y QA 

Este es un proyecto educativo tipo challenger de la plataforma ALURA LATAM diseñado para comprender el funcionamiento de un sistema de **Generación Aumentada por Recuperación (RAG)**. La aplicación permite hacer consultas a archivos de ejemplo o subir archivos propios .PDF/.txt, procesar su contenido mediante modelos de lenguaje, indexarlo en una base de datos vectorial y realizar consultas semánticas.

---

## 🚀 Arquitectura del Sistema

El flujo del proyecto sigue la arquitectura estándar de un sistema RAG:

1. **Ingesta de Documentos:** Se extrae el texto plano de los archivos de ejemplo (carpeta: documentos_ejemplos) o de un PDF/txt subido usando `pypdf`.
2. **Fragmentación (Chunking):** El texto se divide en fragmentos de 1000 caracteres con `langchain-text-splitters` para no perder el contexto de las oraciones.
3. **Embeddings:** Cada fragmento se convierte en un vector matemático utilizando el modelo `gemini-embedding-001` de Google AI Studio.
4. **Almacenamiento Vectorial:** Los vectores se guardan localmente en disco usando `FAISS`.
5. **Recuperación (Retrieval):** Cuando haces una pregunta, el sistema busca los 3 fragmentos vectoriales más cercanos matemáticamente a tu consulta.
6. **Generación (LLM):** Se inyectan los fragmentos recuperados en un prompt estructurado y se envían a `Gemini 2.5 Flash` para obtener una respuesta exacta basada únicamente en el documento.

---

## 🛠️ Tecnologías Utilizadas

* **Frontend:** Streamlit
* **Orquestación y Chunking:** LangChain (Text Splitters)
* **Base de Datos Vectorial:** FAISS
* **LLM & Embeddings:** Gemini 2.5 Flash && gemini-embedding-001 (obtenemos la API desde Google AI Studio)
* **Procesamiento de Archivos:** PyPDF

---

## 💻 Instalación y Uso Local

Sigue estos pasos para ejecutar el proyecto en tu entorno local:

### 1. Clonar el repositorio
```bash o powershell
git clone https://github.com/ssymfreelancer/Challenge-Alura-Agente.git
cd Challenge-Alura-Agente
```

### 2. Crear un entorno virtual
```bash o powershell
python -m venv venv
# Windows:
venv\Scripts\activate   
```

### 3. Instalar las dependencias
```bash o powershell
pip install -r requirements.txt
```
### 4. Ejecutar la aplicación 
```bash o powershell
streamlit run app.py
```

### 5. Desplegar en la nube ( en mi caso utilice una free tier de aws)
El proceso consiste en instalar y desplegar el contenido de este repositorio en una instancia de EC2 , configurar las politicas de seguridad para que se pueda acceder a travez de los puertos que corresponden a 
los utilizados por streamlit.

Link temporal donde esta hosteado en aws el proyecto:   http://44.202.242.18:8501/

---



> 💡 **Nota:** Debido a los limites al usar hacer peticiones con el llm de embeddings se dejo la posibilidad en la barra lateral de agregar la API-KEY de Google , necesitarás una clave de API que puedes adquirir en : [Google AI Studio](https://aistudio.google.com/api-keys) para poder utilizar las funciones de Embedding y el Chatbot. Introduce tu clave directamente en la interfaz de usuario al iniciar la app para desbloquear el chat.

---

### Se ha agregado el archivo que configura el Tema del front de Streamlit con una tematica mas moderna.
### Puedes ver ejemplos de posibles preguntas a realizar para probar el chatbot ademas del diseño del tema aplicado:


---
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/6030a158-74a0-4458-bb05-c85d5fcda440" />
<img width="1600" height="896" alt="image" src="https://github.com/user-attachments/assets/0e4ae2a9-dee9-41cd-896b-9647fe8ca396" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/bcf3b04d-0172-442a-b111-f99acee1712c" />


