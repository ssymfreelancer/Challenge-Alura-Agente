import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS Y DATOS BASE
# ---------------------------------------------------------------------
RUTA_DB = "faiss_index"
CARPETA_EJEMPLOS = "documentos_ejemplos"

if not os.path.exists(CARPETA_EJEMPLOS):
    os.makedirs(CARPETA_EJEMPLOS)

CONTACTOS_AREAS = """
- Recursos Humanos (RH): rrhh@empresa.com | Ext. 401
- Soporte TI / Financiero: finanzas@empresa.com | Ext. 303
"""

def obtener_motor_embeddings(api_key):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        max_retries=6,
        timeout=60
    )

# ---------------------------------------------------------------------
# INTERFAZ GRÁFICA (STREAMLIT)
# ---------------------------------------------------------------------
st.sidebar.image("https://robohash.org/jorge.png?bgset=bg2", width=400)
st.sidebar.title("⚙️ Configuración del RAG")
API_KEY = st.sidebar.text_input("Introduce tu Google API Key:", type="password")
archivos_subidos = st.sidebar.file_uploader(
    "Opcional: Sube más documentos (.pdf / .txt):",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

st.title("🧠 Mi Primer RAG y QA")

if "historial" not in st.session_state:
    st.session_state.historial = []

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# ---------------------------------------------------------------------
#  BASE DE DATOS VECTORIAL
# ---------------------------------------------------------------------
if API_KEY:
    nombres_subidos = "-".join([f.name for f in archivos_subidos]) if archivos_subidos else ""
    
    #  bloquea la primera ejecución
    if "db_inicializada" not in st.session_state or st.session_state.ultimo_estado_archivos != nombres_subidos:
        with st.spinner("⏳ Extrayendo información ..."):
            
            cortador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            documentos_langchain = []

            # --- PARTE A: Carpeta de ejemplos ---
            archivos_locales = [f for f in os.listdir(CARPETA_EJEMPLOS) if f.endswith(('.pdf', '.txt'))]
            for nombre in archivos_locales:
                ruta_completa = os.path.join(CARPETA_EJEMPLOS, nombre)
                if nombre.endswith(".pdf"):
                    lector = PdfReader(ruta_completa)
                    for num_pag, pagina in enumerate(lector.pages, start=1):
                        texto = pagina.extract_text()
                        if not texto: continue
                        for chunk in cortador.split_text(texto):
                            documentos_langchain.append(Document(page_content=chunk, metadata={"fuente": f"[Ejemplo] {nombre}", "pagina": str(num_pag)}))
                
                elif nombre.endswith(".txt"):
                    with open(ruta_completa, "r", encoding="utf-8") as f:
                        texto = f.read()
                    for chunk in cortador.split_text(texto):
                        documentos_langchain.append(Document(page_content=chunk, metadata={"fuente": f"[Ejemplo] {nombre}", "pagina": "Texto Plano"}))

            # --- PARTE B: Archivos subidos por interfaz ---
            if archivos_subidos:
                for archivo in archivos_subidos:
                    if archivo.name.endswith(".pdf"):
                        lector = PdfReader(archivo)
                        for num_pag, pagina in enumerate(lector.pages, start=1):
                            texto = pagina.extract_text()
                            if not texto: continue
                            for chunk in cortador.split_text(texto):
                                documentos_langchain.append(Document(page_content=chunk, metadata={"fuente": archivo.name, "pagina": str(num_pag)}))
                    
                    elif archivo.name.endswith(".txt"):
                        texto = archivo.read().decode("utf-8")
                        for chunk in cortador.split_text(texto):
                            documentos_langchain.append(Document(page_content=chunk, metadata={"fuente": archivo.name, "pagina": "Texto Plano"}))

            # --- PARTE C: Se agregan los vectores a la base vectorial ---
            if documentos_langchain:
                embeddings = obtener_motor_embeddings(API_KEY)
                TAMAÑO_LOTE = 15
                print(f"\n[TERMINAL] Iniciando indexación de {len(documentos_langchain)} fragmentos...")
                
                db_vectorial = FAISS.from_documents(documentos_langchain[:TAMAÑO_LOTE], embeddings)
                
                for i in range(TAMAÑO_LOTE, len(documentos_langchain), TAMAÑO_LOTE):
                    lote = documentos_langchain[i : i + TAMAÑO_LOTE]
                    print(f"[TERMINAL] Indexando fragmentos del {i} al {min(i+TAMAÑO_LOTE, len(documentos_langchain))}")
                    db_vectorial.add_documents(lote)
                    time.sleep(1.5)
                
                db_vectorial.save_local(RUTA_DB)
                st.session_state.ultimo_estado_archivos = nombres_subidos
                st.session_state.db_inicializada = True  # Marca de éxito en memoria
                st.rerun()  # Forzamos recarga limpia para activar la barra de chat de inmediato

# ---------------------------------------------------------------------
# PASO 3: (CHAT DE CONVERSACIÓN)
# ---------------------------------------------------------------------
# Se debe ingresar el api_key para desbloquear el chat de conversacion
if not API_KEY:
    st.info("👋 ¡Estimado! Para activar el chat introduce tu Google API Key en el panel lateral.<------")
    st.chat_input("El chat se activará al ingresar tu API Key...", disabled=True)
else:
    # Verificamos si realmente se creó la base de datos física antes de dejar preguntar
    if not os.path.exists(RUTA_DB):
        st.warning("⚠️ No se encontraron documentos en 'documentos_ejemplo' ni archivos cargados para indexar.")
    else:
        if pregunta_usuario := st.chat_input("Hazme una pregunta sobre tus documentos locales o subidos:"):
            
            with st.chat_message("user"):
                st.markdown(pregunta_usuario)
            st.session_state.historial.append({"rol": "user", "contenido": pregunta_usuario})

            embeddings = obtener_motor_embeddings(API_KEY)
            db_vectorial = FAISS.load_local(RUTA_DB, embeddings, allow_dangerous_deserialization=True)

            busqueda_con_score = db_vectorial.similarity_search_with_score(pregunta_usuario, k=3)

            st.info("📊 **Consola :**")
            contexto_estructurado = ""
            
            for i, (doc, score) in enumerate(busqueda_con_score, start=1):
                meta = doc.metadata
                fuente = meta.get("fuente")
                pagina = meta.get("pagina")
                
                st.write(f"🔹 **Fragmento #{i}** | Origen: `{fuente}` (Pág. {pagina}) | **Distancia L₂:** `{score:.4f}`")
                contexto_estructurado += f"\n[Fuente: {fuente}, Página: {pagina}]\nTexto: {doc.page_content}\n"

            prompt_estudio = f"""
            Utiliza exclusivamente la información provista en el 'Contexto de los Documentos' para responder la pregunta.
            No utilices suposiciones ni conocimiento externo del mundo.

            Si la respuesta exacta no puede ser deducida de la información suministrada, responde textualmente: 
            'No encontré esta información en los documentos disponibles.'
            
            Al final de tu respuesta, añade una sección clara llamada 'Fuentes' indicando el archivo y página.

            --- INICIO DEL CONTEXTO ---
            {contexto_estructurado}
            --- FIN DEL CONTEXTO ---

            Pregunta: {pregunta_usuario}
            Respuesta analítica:
            """

            genai.configure(api_key=API_KEY)
            modelo_llm = genai.GenerativeModel('gemini-2.5-flash')

            with st.chat_message("assistant"):
                try:
                    respuesta_llm = modelo_llm.generate_content(prompt_estudio)
                    texto_final = respuesta_llm.text

                    if "no encontré" in texto_final.lower() or "no disponible" in texto_final.lower():
                        texto_final = f"⚠️ **Información fuera de alcance:** No encontré esta información en los documentos disponibles.\n\nPor favor, ponte en contacto con el área responsable:\n{CONTACTOS_AREAS}"

                    st.markdown(texto_final)
                    st.session_state.historial.append({"rol": "assistant", "contenido": texto_final})
                    
                except Exception as e:
                    st.error(f"Error en el motor de lenguaje: {e}")
