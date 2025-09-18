from pathlib import Path
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#Panel 5 Y 6
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
#Panel 7
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Pestaña 8
from shiny import render, reactive
import joblib
import tempfile


sns.set_style("darkgrid")

datos_dir = Path(__file__).parent / "datos"

# =========================
# UI
# =========================
app_ui = ui.page_fillable(

    ui.tags.style(
    """
    /* Fondo general */
    body {
        background-image: url('Fondo-2.webp');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        margin: 0;
        height: 100vh;
        color: #f5f5f5;
        font-family: Arial, sans-serif;
    }

    /* Paneles semi-transparentes oscuros */
    .navbar, .tab-content, .tab-pane {
        background-color: rgba(20,20,20,0.35);
        color: #f5f5f5;
        height: 100%;
    }

    /* Contenedor full height SOLO para gráficos */
    .full-height {
        height: calc(100vh - 120px);
        min-height: calc(100vh - 120px);
        width: 100%;
        overflow: auto;
    }

    /* Forzar tamaño solo en gráficos */
    #grafico_missing, #grafico_histogramas {
        height: 100% !important;
        width: 100% !important;
    }

    /* Sidebar layout */
    .bslib-sidebar-layout {
        display: flex;
        flex-direction: row;
        height: 100%;
    }

    /* Sidebar oscuro sólido y legible */
    .bslib-sidebar-layout .sidebar {
        flex: 0 0 auto;
        max-width: 420px;
        background-color: rgba(255,165,0,0.95) !important; 
        color: #f5f5f5 !important;
        padding: 35px;
        position: relative;
    }

    /* Main content */
    .bslib-sidebar-layout .main {
        flex: 1 1 auto;
        min-width: 0;
        width: 100%;
        color: #f5f5f5;
    }

    /* Sidebar colapsado */
    .bslib-sidebar-layout.sidebar-collapsed .sidebar,
    .bslib-sidebar-layout[data-collapsed="true"] .sidebar {
        display: none !important;
    }
    .bslib-sidebar-layout.sidebar-collapsed .main,
    .bslib-sidebar-layout[data-collapsed="true"] .main {
        flex: 1 1 100% !important;
        width: 100% !important;
    }

    /* TODAS las pestañas activas en naranja */
    .nav-link.active[data-bs-toggle="tab"] {
        background-color: rgb(255,165,0) !important; /* naranja */
        color: #fff !important;  /* texto blanco */
    }

    /* Tablas (primeros registros + resumen) */
    #primeros_registros, 
    #primeros_registros table,
    #primeros_registros .dataframe,
    #resumen_estadistico, 
    #resumen_estadistico table,
    #resumen_estadistico .dataframe {
        width: 100% !important;
        color: #f5f5f5;
        background-color: rgba(30,30,30,0.8);
        border-color: #555555;
    }

    #resumen_estadistico, #primeros_registros {
        overflow-x: auto;
    }

    /* Encabezados */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }

    /* Sliders y marcadores */
    .slider, .shiny-output-error {
        color: #f5f5f5;
    }

    /* Pestañas inactivas en naranja */
    .nav-link[data-bs-toggle="tab"] {
        color: orange !important;   /* texto naranja */
    }

    /* Al pasar el mouse sobre una pestaña inactiva */
    .nav-link[data-bs-toggle="tab"]:hover {
        color: #ffcc80 !important;  /* naranja más claro */
    }

    /* Alinear texto de todas las tablas de datos a la izquierda */
    #primeros_registros table th,
    #primeros_registros table td,
    #resumen_estadistico table th,
    #resumen_estadistico table td {
    text-align: left !important;  /* todo a la izquierda */
    vertical-align: middle !important; /* centrado vertical */
    padding: 6px 10px; /* un poco de espacio */
    }
    /* Título del sidebar */
    .titulo-sidebar {
        font-size: 20px;
        font-weight: bold;
        color: #f0f0f0;
        margin-bottom: 15px;
    }

    /* Botones redondeados */
    .btn-redondeado {
        border-radius: 25px !important;
        font-size: 14px;
        font-weight: bold;
        padding: 8px 16px;
        margin-bottom: 10px;
        width: 100%;
    }
    /* Aplica estilo al botón de subida de fichero */
    input[type="file"] + .btn {
        border-radius: 12px;       /* bordes redondeados */
        background-color: #4CAF50; /* verde */
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border: none;
        cursor: pointer;
    }
    input[type="file"] + .btn:hover {
        background-color: #45a049;
    }

    /* Colores personalizados */
    .btn-verde {
        background-color: #28a745 !important;
        border: none;
        color: white !important;
    }
    .btn-verde:hover {
        background-color: #218838 !important;
    }

    .btn-azul {
        background-color: #007bff !important;
        border: none;
        color: white !important;
    }
    .btn-azul:hover {
        background-color: #0069d9 !important;
    }


    """
    ),
#=================================================================================
# UI ===========================================================================

    # Título general
    ui.div(
        ui.h2(
            "Programa Experto en Data Science - Datamecum/ Ejercicio fin de Curso/ Guille Renart",
            style=(
                "color: orange; "
                "text-align: center; "
                "margin-top: 10px; "
                "margin-bottom: 10px; "
                "font-weight: bold;"
            )
        ),
        style="width: 100%;"
    ),

# UI 1 ==========================================================
# ---------------- Pestaña 1 ------------------------------------
    ui.navset_tab(
        # Panel 1: solo resumen estadístico
        ui.nav_panel(
            "1→ TRAINING: Resumen estadístico",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Carga Fichero de Entreno"),
                    ui.input_file(
                        "Fichero_ID",
                        "Selecciona fichero",
                        accept=[".csv", ".txt", ".xlsx"],
                        multiple=False,
                        button_label="Fichero"
                    ),

                    ui.div(
                        ui.output_text("fichero_status"),
                        style="color: orange; text-align: center; margin-top: 5px;"
                    )
                ),
                ui.div(
                    #ui.h4("Resumen estadístico", style="color: orange; text-align: center; margin: 10px;"),
                    ui.output_table("resumen_estadistico"),
                    class_="resumen-container"
                ),
                fill=True
            )
        ),

# UI 2 ==========================================================
# ---------------- Pestaña 2 ------------------------------------
        # Gráfico de datos faltantes + primeros registros
        ui.nav_panel(
             "2→ TRAINING: Porcentaje datos Faltantes",
            ui.div(
                ui.h3("Primeros datos del fichero", 
                      style="color: orange; text-align: center; margin: 10px;"),
                ui.output_table("primeros_registros"),
            ui.div(
                ui.output_plot("grafico_valores_faltantes"),
                style="height: 60%;"  # ajusta al 60% de la altura disponible
                ),
            class_="full-height"
            ),
        ),


# UI 3 ==========================================================
# ---------------- Pestaña 3 ------------------------------------
        # Histograma /Curvas de densidad
        ui.nav_panel(
            "3→ TRAINING: Curvas de densidad",
            ui.div(
                ui.output_plot("curvas_densidad", height="100%", width="100%"),
                class_="full-height",
                style="display: flex; flex-direction: column; height: calc(100vh - 120px);"
            )
        ),

# UI 4 ==========================================================
# ---------------- Pestaña 4 ------------------------------------
        # Imputación MICE
        ui.nav_panel(
            "4→ TRAINING: Imputación MICE",
            ui.layout_columns(
                # Columna izquierda: descripción MICE
                ui.card(
                    ui.h4("Método MICE (Multiple Imputation by Chained Equations)",
                        style="color: orange; text-align: center; margin: 10px;"),
                    ui.p(
                        ui.HTML(
                            "→ <b>MICE</b>  es un método iterativo que 👉 <b>imputa valores faltantes</b> "
                            "modelando cada variable incompleta como función del resto. <br><br>"
                            "→ En cada iteración, los valores ausentes se rellenan con <b>predicciones</b> "
                            "obtenidas a partir de modelos de <b>regresión</b> adecuados al tipo de variable. <br><br>"
                            "→ El proceso se <b> repite en cadena</b> hasta alcanzar la <b>convergencia</b>, "
                            "generando imputaciones estadísticamente consistentes. <br><br>"
                            "👉 Este enfoque preserva la <b>estructura multivariante</b> de los datos, "
                            "reduciendo sesgos frente a imputaciones simples."
                        ),
                        style="color: #f5f5f5; text-align: justify; margin: 10px; font-size: 20px;"
                    ),
                    style="padding: 15px; background-color: rgba(20,20,20,0.7); border-radius: 8px; height: 100%;"
                ),
                
                # Columna derecha: tabla resultados
                ui.card(
                    ui.h4("📊 Resumen imputación MICE",
                        style="color: orange; text-align: center; margin: 10px;"),
                    ui.output_table("mice_summary"),
                    style="padding: 15px; background-color: rgba(0,0,0,0.7); border-radius: 8px; height: 100%;"
                ),
                col_widths=(6, 6)  # 50% izquierda, 50% derecha
            )
        ),


# UI 5 ==========================================================
# ---------------- Pestaña 5 ------------------------------------
        # Auditoría Dataset imputado
        ui.nav_panel(
            "5→ TRAINING: Auditoría del dataset",
            
           ui.div(
                ui.layout_columns(
                    ui.output_ui("auditoria_dataset"),  # tabla a la izquierda
                    ui.output_plot("auditoria_heatmap"),  # heatmap a la derecha
                    col_widths=[6, 6],  # 50%-50%, puedes ajustar [4,8] por ejemplo
                ),
                ui.div(
                    ui.output_plot("audit_densidad", height="100%", width="100%"),
                    style="margin-top: 20px; height: calc(100vh - 200px);"
                ),
                
            )
        ),

# UI 6 ==========================================================
# ---------------- Pestaña 6 ------------------------------------
#Transformación y Winsorización

            ui.nav_panel(
                "6→ TRAINING: Tratamiento de Datos",
                ui.layout_column_wrap(
                    # Parte superior: descripción
                    ui.card(
                        ui.h4("🔄 Transformación y Winsorización ",
                        style="color: orange; text-align: center; margin: 10px;"),
                        ui.markdown(
                            """ 
                            - Valores faltantes iniciales: → Se ha aplicado **MICE** (ver Panel 3).  
                            - Estudio de **Correlación entre variables** → No hay multicolinealidad grave.  
                            - **Sesgo alto /Asimetria en distribución /Cola pesada"** 
                                        → Variable **x8** presenta un sesgo extremo (+3.42).  
                                        → Variable **x1** también sobre el umbral (+1.06).  

                            - **Outliers** → Especialmente concentrados en **x8 (40)** y **x1 (23)**,  
                            lo que valida el uso de la *winsorización* para reducir su impacto.  

                            👉 Transformacion del Datset:  
                            - Correccion de sesgo usando Yeo-Johnson (acepta ceros/negativos).  
                            - Winsorización al 1% y 99% (recorte de valores extremos).  

                            **Resultados esperados:**  
                            - Reducción del sesgo.  
                            - Menor número de outliers en x1 y x8.  
                            """
                        ),
                        style=(
                            "margin-bottom: 10px; "
                            "background-color: rgba(0,0,0,0.7); "
                            "color: white; "
                            "border-radius: 10px; "
                            "padding: 15px;"
                        )
                    ),
                    # Gráfico antes vs después
                    ui.card(
                        ui.output_plot("plot_trans_winsor", height="600px"),
                        style=(
                            "margin-bottom: 10px; "
                            "background-color: rgba(0,0,0,0.7); "
                            "border-radius: 10px; "
                            "padding: 10px;"
                        )
                    ),
                    # Tabla comparativa
                    ui.card(
                        ui.output_data_frame("tabla_trans_winsor"),
                        style=(
                            "margin-bottom: 10px; "
                            "background-color: rgba(0,0,0,0.7); "
                            "border-radius: 10px; "
                            "padding: 10px; "
                            "color: orange;"
                        )
                    ),
                    column_size="12"
                )

            
                
            ),
      
# UI 7 ==========================================================
# ---------------- Pestaña 7 ------------------------------------
      #Resultados Modelo XGBoost ----------
       
            ui.nav_panel(
                "7→ MODELO: Resultados Modelo XGBRegressor",
                ui.layout_columns(
                    # ================== Columna Izquierda (3/12) ==================
                    ui.card(
                        # ---------------- Parte superior: resumen ----------------
                        ui.h4("📊 Resultados principales del modelo", 
                            style="color: orange; text-align:center; margin:10px;"),
                        ui.output_ui("resumen_modelo", style="font-size:14px; color:white; line-height:1.3;"),

                        # ---------------- Parte inferior: gráfico R² ----------------
                        ui.output_plot("grafico_r2", height="300px"),

                        # ---------------- Estilo de la card ----------------
                        style=(
                            "margin-bottom: 10px; "
                            "background-color: rgba(0,0,0,0.7); "
                            "color: white; "
                            "border-radius: 10px; "
                            "padding: 15px;"
                        )
                    ),

                    # ================== Columna Central (6/12) ==================
                    ui.card(
                        ui.h4("📊 Curva de predicciones vs valores reales", 
                            style="color:orange; text-align:center;"),
                        ui.output_plot("curva_real_predicho", height="600px"),
                        style="background-color:rgba(0,0,0,0.6); margin-top:15px; padding:10px; border-radius:10px;"
                    ),

                    # ================== Columna Derecha (3/12) ==================
                    ui.card(
                        ui.output_plot("grafico_importancia", height="600px"),
                        style=(
                            "margin-bottom: 10px; "
                            "background-color: rgba(0,0,0,0.7); "
                            "border-radius: 10px; "
                            "padding: 10px;"
                        )
                    ),
                    col_widths=[3, 6, 3]   # <-- define proporción de cada bloque
                )
            ),


#= UI 7.b - Comparativa 4 Modelos ==================================================

        ui.nav_panel(
            "7.b→ Comparador de Modelos",
            
            ui.layout_columns(

                # ==========Columna LinearRegression============
                ui.card(
                    ui.output_ui("resumen_comparativa_lr", style="font-size:16px; color:orange; text-align:center; margin:10px;"),
                    ui.output_plot("grafico_r2_lr", height="440px"),
                    style=(
                        "background-color: rgba(0,0,0,0.7); "
                        "border-radius: 10px; padding: 10px; margin-bottom: 10px;"
                    )
                ),

                # ================== Columna XGBoost==================
                ui.card(
                    ui.output_ui("resumen_comparativa_xgb", style="font-size:16px; color:orange; text-align:center; margin:10px;"),
                    ui.output_plot("grafico_r2_xgb", height="440px"),
                    #ui.output_plot("curva_real_predicho_xgb", height="300px"),
                    style=(
                        "background-color: rgba(0,0,0,0.7); "
                        "border-radius: 10px; "
                        "padding: 10px; "
                        "margin-bottom: 10px;"
                    )
                ),

                # ================== Columna RandomForest ==================
                ui.card(
                    ui.output_ui("resumen_comparativa_rf", style="font-size:16px; color:orange; text-align:center; margin:10px;"),
                    ui.output_plot("grafico_r2_rf", height="440px"),
                    #ui.output_plot("curva_real_predicho_rf", height="300px"),
                    style=(
                        "background-color: rgba(0,0,0,0.7); "
                        "border-radius: 10px; "
                        "padding: 10px; "
                        "margin-bottom: 10px;"
                    )
                ),

                # ================== Columna HistGB ==================
                ui.card(
                    ui.output_ui("resumen_comparativa_hgb", style="font-size:16px; color:orange; text-align:center; margin:10px;"),
                    ui.output_plot("grafico_r2_hgb", height="440px"),
                    #ui.output_plot("curva_real_predicho_hgb", height="300px"),
                    style=(
                        "background-color: rgba(0,0,0,0.7); "
                        "border-radius: 10px; "
                        "padding: 10px; "
                        "margin-bottom: 10px;"
                    )
                ),

                col_widths=[3,3,3,3]  # cada columna igual
            )
        ),


# UI 8 ==========================================================
# ---------------- Pestaña 8 ------------------------------------
        # ------Resultados de PREDICCION ------------------------
        ui.nav_panel(
            "8→ PREDICCIONES XGB Entrenado",
            ui.layout_sidebar(
                    ui.sidebar(
                        # --- Título mejorado ---
                        ui.h5("📂 Cargar Datos para Predicción", class_="titulo-sidebar"),

                        # --- Botón de carga ---
                        ui.input_file( "input_file_pred",
                            "",
                            accept=[".csv", ".txt", ".xlsx"],
                            multiple=False,
                            button_label="📁 Subir Fichero"                
                        ),

                        # --- Botones de descarga estilizados ---
                        ui.download_button(
                            "download_predicciones",
                            "⬇ Predicciones",
                            class_="btn-redondeado btn-verde"
                        ),
                        ui.download_button(
                            "download_dataset_predicciones",
                            "⬇ Datos + Predicciones",
                            class_="btn-redondeado btn-azul"
                        ),

                        # --- Información ---
                        #ui.h5("🔧 Columnas transformadas:"),
                        ui.output_ui("info_transformadas")
                    ),
                

                ui.layout_column_wrap(
                    ui.card(
                        ui.h4("🔮 Predicciones", style="color:orange;text-align:center;"),
                        ui.output_data_frame("tabla_predicciones"),
                        style="background-color:rgba(0,0,0,0.5);color:orange;"
                    ),
    # ----------------- Importancia de variables ------------------
                    ui.card(
                        ui.output_plot("grafico_importancia_8", height="500px"),
                        style=("margin-bottom: 10px;"
                            "background-color: rgba(0,0,0,0.7); "
                            "border-radius: 10px; "
                            "padding: 10px;"
                        )
                    ),

                    column_size="6"
                )
            )
        )


    ),
    fill=True
)

# ====================================================================================================
# SERVIDOR
# ====================================================================================================

#=== SERVER =============== INPUT FICHERO =============================

def server(input, output, session):
    datos = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.Fichero_ID)
    def _():
        file_info = input.Fichero_ID()
        if not file_info:
            datos.set(None)
            return
        file_path = file_info[0]["datapath"]
        if file_path.endswith(".csv") or file_path.endswith(".txt"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            datos.set(None)
            return
        datos.set(df)

    @output
    @render.text
    def fichero_status():
        file_info = input.Fichero_ID()
        if not file_info:
            return "No hay fichero"
        return f"Cargado: {file_info[0]['name']}"
    
#=== SERVER =============== 1: TRAINiING RESUMEN ESTADISTICO ====================
 
    @output
    @render.ui
    def resumen_estadistico():
        df = datos.get()
        if df is None:
            # Texto de presentación en HTML estilizado tipo card
            presentacion_html = """
            <div style="max-width: 900px; 
                        margin: 0 auto; 
                        background-color: rgba(0,0,0,0.75); 
                        color: white; 
                        padding: 25px; 
                        border-radius: 15px; 
                        box-shadow: 0 0 20px rgba(255,140,0,0.3); 
                        font-family: Arial, sans-serif; 
                        line-height: 1.6;">

                <h2 style="color: orange; text-align: center; margin-bottom: 15px;">
                    📊 Bienvenidos al ejercicio final de curso
                </h2>

                <h3 style="color: orange; text-align: center; margin-bottom: 20px;">
                    Programa Experto en Data Science / Datamecum → Datathon
                </h3>
                <p>
                    <strong>Objetivo:</strong>
                </p>

                <p> Se pretende generar un 
                    <span style="color: orange;">Modelo predictivo</span> 
                    con cierta fiabilidad, aplicando las técnicas aprendidas 
                    durante el curso.
                </p>

                <p>
                    La dinámica es la siguiente: cargaremos un conjunto de datos inicial 
                    llamado: Train.xlsx. 
                    Este Dataset se compone de un conjunto variables + una objetivo:  
                    <span style="color: orange;">"deseada"</span>.  
                </p>
                <p>
                    Realizaremos un análisis exploratorio y posterior procesado, 
                    para después construir un modelo capaz de predecir esta variable objetivo.
                </p>

                <p>
                    Para ello, iremos navegando por las pestañas de 
                    <span style="color: orange;">Training (1 al 7)</span>, 
                    donde se irán generando informaciones y gráficos de nuestro conjunto 
                    de entrenamiento.  
                    En la última parte de esta secuencia (pestaña 7), se ofrece el 
                    <strong>análisis del Modelo generado</strong>.
                </p>

                <p>
                    Finalmente, en la pestaña <span style="color: orange;">Predicción (8)</span>, 
                    pondremos a prueba nuestro trabajo sobre datos que el modelo nunca ha visto, 
                    obteniendo la predicción final y la opción de descargar los resultados 
                    en formato Excel.
                </p>

                <p style="text-align: center; font-weight: bold; margin-top: 20px; color: orange;">
                    🎓 Agradecido por este viaje, a profesores y compañeros.
                </p>
            </div>
            """
            return ui.HTML(presentacion_html)

        else:
            # ===============================
            # Resumen estadístico del dataset
            # ===============================
            df_desc = df.describe(include="all").reset_index()
            alpha = 0.7
            bg_color = f"rgba(0,0,0,{alpha})"

            styled = (
                df_desc.style
                .set_table_attributes('class="dataframe table table-striped"')
                .set_table_styles([
                    {'selector': 'th', 'props': f'color: orange; background-color: {bg_color}; text-align: center;'},
                    {'selector': 'td', 'props': f'color: white; background-color: {bg_color};'},
                ])
            )

            titulo_html = """
            <h4 style="color: orange; text-align: center; margin-bottom: 10px;">
                📊 Resumen estadístico de los datos cargados
            </h4>
            """

            return ui.HTML(titulo_html + styled.to_html())
            
#=== SERVER =============== 2: TRAINiING PORCENTAAJE DATOS FALTANTES =============================

    @output
    @render.ui
    def primeros_registros():
        df = datos.get()
        if df is None:
            return ui.HTML("<div>No hay fichero cargado</div>")

        # Primeros registros
        df_head = df.head(6).reset_index().rename(columns={"index": "Fila"})

        # Estilo: celdas NaN en naranja, texto negro
        def highlight_nan(val):
            return 'background-color: orange; color: #000;' if pd.isna(val) else ''

        styled = df_head.style.map(highlight_nan)

        # Convertir a HTML
        return ui.HTML(styled.to_html())

    #-------Grafica de Barras datos Faltantes
    @output
    @render.plot
    def grafico_valores_faltantes():
        df = datos.get()
        if df is None:
            return

        missing = df.isna().mean() * 100
        fig, ax = plt.subplots(figsize=(10, 6))

        # Fondo negro con transparencia
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(0.6)  # 60% opaco
        ax.set_facecolor("black")
        ax.patch.set_alpha(0.6)   # 60% opaco

        # Gráfico de barras
        sns.barplot(x=missing.index, y=missing.values, hue=missing.index, palette="viridis", legend=False, ax=ax)

        # Títulos y etiquetas
        ax.set_title("Porcentaje de valores faltantes", color="orange", fontsize=20, pad=30)
        ax.set_ylabel("%", fontsize=16, color="white")
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=14, color="white")
        ax.set_xticks(range(len(missing.index)))
        ax.set_xticklabels(missing.index, rotation=45, fontsize=14, color="white")

        ax.tick_params(colors="white")  # ticks en blanco
        ax.yaxis.label.set_color("white")

        # Añadir valores encima de cada barra
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2.,   # posición X centrada en la barra
                height + 1,                       # un poco encima del final de la barra
                f"{height:.1f}%",                 # formato: un decimal + %
                ha="center", va="bottom",
                color="orange", fontsize=12, fontweight="bold"
            )

        return fig
    

#=== SERVER =============== 3: TRAINiING CURVAS DE DENSIDAD  =============================
    
    #------------ Curvas densidad de Variables Originales
    @output
    @render.plot
    def curvas_densidad():
        df = datos.get()
        if df is None:
            return
        numeric_cols = df.select_dtypes(include=np.number)
        if numeric_cols.empty:
            return

        n_vars = len(numeric_cols.columns)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols

    # Ajustar tamaño de figura según número de filas/columnas
        fig_width = 6 * n_cols
        fig_height = max(6, 4 * n_rows)  # asegurar alto mínimo
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = np.array(axes).reshape(-1)

    # Fondo negro con transparencia
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(0.7)
        for ax in axes:
            ax.set_facecolor("black")
            ax.patch.set_alpha(0.7)

    # Dibujar gráficos de densidad
        for i, col in enumerate(numeric_cols.columns):
            sns.kdeplot(df[col].dropna(), fill=True, ax=axes[i], color="skyblue", alpha=0.7)

        # Todos los títulos y ticks en blanco
            axes[i].set_title(f"Densidad: {col}", fontsize=12, color="white")
            axes[i].tick_params(axis='x', colors='white')
            axes[i].tick_params(axis='y', colors='white')
            axes[i].xaxis.label.set_color("white")
            axes[i].yaxis.label.set_color("white")

    # Apagar ejes sobrantes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

    # Título general
        fig.suptitle(
            "Curvas de densidad: Conjunto de Datos de Entrenamiento",
            fontsize=18,
            y=0.99,
            fontweight="bold",
            color="orange"
        )

    # Ajuste de espacios
        fig.subplots_adjust(
            top=0.90,
            bottom=0.08,
            left=0.06,
            right=0.98,
            hspace=0.5,
            wspace=0.3
        )

        return fig

#=== SERVER =============== 4: TRAINiING IMPUTACION MICE  =============================

# ---------- Funcion auxiliar: ---Relleno Iterativo de valores faltantes
#  Funcion IMPUTACION_MICE-------------------

    def imputacion_mice(df, random_state=777, max_iter=20):
        """
        Aplica imputación múltiple MICE (IterativeImputer) a un DataFrame.
        
        Parámetros:
            df (pd.DataFrame): DataFrame con valores faltantes.
            random_state (int): Semilla para reproducibilidad.
            max_iter (int): Número máximo de iteraciones.

        Devuelve:
            pd.DataFrame: DataFrame imputado (sin NaN).
        """
        if df.empty:
            return df
        
        imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)
        df_imputado = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        return df_imputado

#-----------------------------------------------------------------------------------

    @output
    @render.ui
    def mice_summary():
        df = datos.get()
        if df is None:
            return ui.HTML('<p style="color:white;">No hay fichero cargado</p>')

        # Calcular % NaN antes
        nan_antes = df.isna().mean() * 100

        #---Rellenar Datos faltantes con  MICE
        df_imputado = imputacion_mice(df)

        # Calcular % NaN después
        nan_despues = df_imputado.isna().mean() * 100

        # Preparar tabla resumen
        resumen = pd.DataFrame({
            "%_NaN_originalmente": nan_antes,
            "%_NaN_despues": nan_despues
        }).sort_values(by="%_NaN_originalmente", ascending=False)

        # Añadir tamaño de datos
        #resumen.loc["numero de datos"] = [df.shape[0], df_imputado.shape[0]]

        # Reiniciamos índice y creamos columna "Variable"
        resumen = resumen.reset_index()
        resumen.rename(columns={"index": "Variable"}, inplace=True)

        # Seleccionamos solo las columnas necesarias para evitar columna ordinal extra
        resumen = resumen[['Variable', '%_NaN_originalmente', '%_NaN_despues']]

        # Fondo negro con 70% de opacidad
        alpha = 0.7
        bg_color = f"rgba(0,0,0,{alpha})"

        # Aplicar estilo
        styled = resumen.style.set_table_attributes(
            'class="dataframe table table-striped" style="width:50%;"'
        ).set_table_styles([
            {'selector': 'th', 'props': f'color: white; background-color: {bg_color};'},
            {'selector': 'td', 'props': f'color: white; background-color: {bg_color};'},
        ])

        # Devolver HTML
        return ui.HTML(styled.to_html())
    

#=== SERVER =============== 5: TRAINiING AUDITORIA DEL DATASET 'SOLO IMPUTADO'  =============================
        
    @output
    @render.ui
    def auditoria_dataset():
        df = datos.get()
        if df is None:
            return ui.h5("No hay fichero cargado", style="color: orange; text-align:center;")
        

        # Rellenamos datos con MICE
        df_audit = imputacion_mice(df)

        # --- 1️⃣ Número de datos
        n_datos = f"{df_audit.shape[0]} filas, {df_audit.shape[1]} columnas"

        # --- 2️⃣ Valores faltantes
        missing = df_audit.isnull().sum()
        missing_vars = ", ".join(missing[missing > 0].index) if missing.sum() > 0 else "✅ Ninguno"

        # --- 3️⃣ Correlaciones > 0.95
        corr_matrix = df_audit.corr(numeric_only=True)
        high_corr = [
            f"{c1}↔{c2} ({corr_matrix.loc[c1, c2]:.2f})"
            for c1 in corr_matrix.columns
            for c2 in corr_matrix.columns
            if c1 < c2 and abs(corr_matrix.loc[c1, c2]) > 0.95
        ]
        corr_info = ", ".join(high_corr) if high_corr else "✅ Ninguna"

        # --- 4️⃣ Sesgo alto (>1 o <-1)
        skew_vals = df_audit.apply(
            lambda x: skew(x.dropna()) if np.issubdtype(x.dtype, np.number) else None
        )
        high_skew = skew_vals[(skew_vals > 1) | (skew_vals < -1)]
        skew_info = ", ".join([f"{var} ({val:.2f})" for var, val in high_skew.items()]) if not high_skew.empty else "✅ Ninguno"
        #"Se evalúa la asimetría (skewness) de cada variable numérica. Un sesgo alto (>|1|) indica 
        # que la distribución se desvía significativamente de la simetría normal, pudiendo generar problemas en el modelado y 
        # requerir transformaciones (p.ej. log o Yeo-Johnson)."

        # --- 5️⃣ Outliers (1.5 * IQR)
        outlier_info = []
        for col in df_audit.select_dtypes(include=np.number).columns:
            Q1, Q3 = df_audit[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((df_audit[col] < (Q1 - 1.5 * IQR)) |
                        (df_audit[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_info.append(f"{col}: {outliers}")
        outlier_info = ", ".join(outlier_info) if outlier_info else "✅ Ninguno"
        #"Se identifican outliers mediante el rango intercuartílico (IQR). Todo valor inferior a Q1 − 1.5·IQR o superior a Q3 + 1.5·IQR 
        # se considera un dato atípico. Los outliers pueden distorsionar estadísticas descriptivas y afectar el rendimiento de los modelos predictivos."


        # --- Crear tabla resumen
        resumen = pd.DataFrame({
            "Informacion del Dataset": [
                "Disposicion de los Datos",
                "Valores faltantes de las variables",
                "Correlaciones entre variables > al 95%",
                "Sesgo alto /Desvio de simetria (>|1|) ",
                "Outliers / valor inferior a Q1 − 1.5·IQR o superior a Q3 + 1.5·IQR "
            ],
            "Resultado": [
                n_datos,
                missing_vars,
                corr_info,
                skew_info,
                outlier_info
            ]
        })

        # --- Fondo negro con transparencia configurable
        alpha = 0.7
        bg_color = f"rgba(0,0,0,{alpha})"

        # --- Estilizar tabla
        styled_html = resumen.style.set_table_attributes('class="dataframe table table-striped"') \
            .set_table_styles([
                {'selector': 'th', 'props': f'color: white; background-color: {bg_color};'},
                {'selector': 'td', 'props': f'color: white; background-color: {bg_color};'}
            ]).hide(axis="index").to_html()

        return ui.HTML(styled_html)  
  
    #---------------------- HeatMap------------------------
    @output
    @render.plot
    def auditoria_heatmap():
        df = datos.get()
        if df is None:
            return

        # Aplicar MICE
        df_heat = imputacion_mice(df)

        corr_matrix = df_heat.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Fondo negro ajustable con alpha
        alpha = 0.3  # 🔧 ajusta entre 0 (transparente) y 1 (sólido)
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(alpha)
        ax.set_facecolor("black")
        ax.patch.set_alpha(alpha)

        # Heatmap
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            ax=ax,
            annot=False,
            cbar=True
        )

        # Título más grande y separado
        ax.set_title(
            "Mapa de correlaciones",
            color="orange",
            fontsize=18,   # más grande
            fontweight="bold",
            pad=5         # más separación
        )
        fig.subplots_adjust(
            top=0.9,   # espacio arriba (1.0 = borde total)
            bottom=0.15,  # espacio abajo (0 = borde total)
            left=0.15,
            right=0.95
        )

        # Etiquetas en naranja y negrita
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, color="orange", fontweight="bold", rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, color="orange", fontweight="bold", rotation=0)

        # Ajustar layout
        plt.tight_layout()

        return fig

    #-------------------  DENSIDAD VARIABLES ANTES Y DESPUES DE IMPUTACION
    @output
    @render.plot
    def audit_densidad():
        df = datos.get()
        if df is None:
            return

        # Si es conjunto de Training o de Test
        target_col = "deseada"
        if target_col in df.columns:
            df_no_imputado = df.drop(columns=[target_col])
        else:
            df_no_imputado = df.copy()

        # 🔹 Aplicar MICE al dataset
        
        df_imputado = imputacion_mice(df_no_imputado)

        numeric_cols = df_imputado.select_dtypes(include="number")
        if numeric_cols.empty:
            return

        n_vars = len(numeric_cols.columns)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols

        # Ajustar tamaño de figura
        fig_width = 6 * n_cols
        fig_height = max(6, 4 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = np.array(axes).reshape(-1)

        # Fondo negro
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(0.7)
        for ax in axes:
            ax.set_facecolor("black")
            ax.patch.set_alpha(0.7)

        # Dibujar curvas de densidad
        for i, col in enumerate(numeric_cols.columns):
            # Original (ignora NaN)
            sns.kdeplot(
                df_no_imputado[col].dropna(),
                ax=axes[i],
                color="skyblue",
                lw=2,
                label="Original"
            )
            # Imputado
            sns.kdeplot(
                df_imputado[col],
                ax=axes[i],
                color="orange",
                lw=2,
                linestyle="--",
                label="Imputado"
            )

            axes[i].set_title(f"Densidad: {col}", fontsize=12, color="white")
            axes[i].tick_params(axis='x', colors='white')
            axes[i].tick_params(axis='y', colors='white')
            axes[i].xaxis.label.set_color("white")
            axes[i].yaxis.label.set_color("white")
            axes[i].legend(facecolor="black", edgecolor="white", labelcolor="white")

        # Apagar ejes sobrantes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Título general
        fig.suptitle(
            "Curvas de densidad: Original vs Imputado (MICE)",
            fontsize=18,
            y=0.99,
            fontweight="bold",
            color="orange"
        )

        # Ajustar espacios
        fig.subplots_adjust(
            top=0.90,
            bottom=0.08,
            left=0.06,
            right=0.98,
            hspace=0.5,
            wspace=0.3
        )

        return fig

#=== SERVER =============== 6: TRAINING TRATAMIENTO DE DATOS  IMPUTADO + YEO + WINSOR ====================

    
    # ---------- Funciones auxiliares ------------------
    #------------Logaritmica ---------------------------

    def fix_skew(series):
        """Corrige sesgo con log1p, manejando ceros y negativos"""
        if (series <= 0).any():
            shift = abs(series.min()) + 1
            return np.log1p(series + shift)
        else:
            return np.log1p(series)
        
    # ---------- Funciones auxiliares ------------------
    #------------Yeo-Johnson----------------------------
    def fix_skew2(series):
        """
        Corrige sesgo usando Yeo-Johnson (acepta ceros y negativos).
        Devuelve un pandas.Series transformado.
        """
        series_reshaped = series.values.reshape(-1, 1)

        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        transformed = pt.fit_transform(series_reshaped)

        return pd.Series(transformed.flatten(), index=series.index)
        
    # ---------- Funciones auxiliares ------------------
    #------------Winsorizado----------------------------
    def winsorize_series(s, lower=0.01, upper=0.99):
        """Recorta valores extremos usando percentiles, quitamos bigotes del Boxplot"""
        return s.clip(lower=s.quantile(lower), upper=s.quantile(upper)) 
    #----------------------------------------------------

    @output
    @render.plot
    def plot_trans_winsor():
        df = datos.get()
        if df is None:
            return

    #-----Quitamos Deseada--------------------------------
        target_col = "deseada"
        if target_col in df.columns:
            df= df.drop(columns=[target_col])
        else:
            df= df.copy()
    
     #----Rellenamos Valores faltantes en Variables -----
        df=imputacion_mice(df)
    
    #--------Transformando: Yeo-Johnson + Windsorize ------

        # Copia original y transformada
        df_Panel6 = df.copy()
        for col in ["x1", "x8"]:
            df_Panel6[col] = fix_skew2(df_Panel6[col])
            df_Panel6[col] = winsorize_series(df_Panel6[col])

        # Gráfico 2x2 con densidades
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        # Fondo negro con transparencia
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(0.7)
        for ax in axes:
            ax.set_facecolor("black")        
            ax.patch.set_alpha(0.7)
            colors = ["skyblue", "green"]

        for i, col in enumerate(["x1", "x8"]):
            # Antes
            sns.kdeplot(df[col].dropna(), fill=True, alpha=0.7, color=colors[0], ax=axes[i])
            axes[i].set_title(f"{col} antes", fontsize=14, color="orange", fontweight="bold")
            axes[i].tick_params(axis='x', colors='orange')
            axes[i].tick_params(axis='y', colors='orange')
            axes[i].xaxis.label.set_color("orange")
            axes[i].yaxis.label.set_color("orange")

            # Después
            sns.kdeplot(df_Panel6[col].dropna(), fill=True, alpha=0.7, color=colors[1], ax=axes[i+2])
            axes[i+2].set_title(f"{col} después", fontsize=14, color="orange", fontweight="bold")
            axes[i+2].tick_params(axis='x', colors='orange')
            axes[i+2].tick_params(axis='y', colors='orange')
            axes[i+2].xaxis.label.set_color("orange")
            axes[i+2].yaxis.label.set_color("orange")

        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.98, 
                            hspace=0.35, wspace=0.3)

        return fig

    # --------------------------------------------------
    
    @output
    @render.data_frame
    def tabla_trans_winsor():
        df = datos.get()
        if df is None:
            return
        
        #-----Quitamos Deseada--------------------------------
        target_col = "deseada"
        if target_col in df.columns:
            df= df.drop(columns=[target_col])
        else:
            df= df.copy()
    
        #----Rellenamos Valores faltantes en Variables -----
        df=imputacion_mice(df)
    
        #--------Transformando: Yeo-Johnson + Windsorize ------

        df_Panel6 = df.copy()
        for col in ["x1", "x8"]:
            df_Panel6[col] = fix_skew(df_Panel6[col])
            df_Panel6[col] = winsorize_series(df_Panel6[col])

        # Sesgo
        skew_before = df[["x1", "x8"]].apply(lambda x: skew(x.dropna()))
        skew_after = df_Panel6[["x1", "x8"]].apply(lambda x: skew(x.dropna()))

        # Outliers antes y después
        outliers_before, outliers_after = {}, {}
        for col in ["x1", "x8"]:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers_before[col] = ((df[col] < (Q1 - 1.5 * IQR)) |
                                    (df[col] > (Q3 + 1.5 * IQR))).sum()

            Q1_a, Q3_a = df_Panel6[col].quantile(0.25), df_Panel6[col].quantile(0.75)
            IQR_a = Q3_a - Q1_a
            outliers_after[col] = ((df_Panel6[col] < (Q1_a - 1.5 * IQR_a)) |
                                (df_Panel6[col] > (Q3_a + 1.5 * IQR_a))).sum()

            comparison_df = pd.DataFrame({
                "Sesgo antes": skew_before,
                "Sesgo después": skew_after,
                "Outliers antes": pd.Series(outliers_before),
                "Outliers después": pd.Series(outliers_after)
            })
                # Redondear decimales a 3
            comparison_df = comparison_df.round(3)

        # Hacemos que el índice sea columna explícita
        comparison_df = comparison_df.reset_index().rename(columns={"index": "Variable"})
        
        # Estilo tabla acorde al tema oscuro
        return render.DataGrid(comparison_df, height="250px")
    

#=== SERVER =============== 7: TRAINiING RESULTADOS MODELO XGBOOST REGRESSOR =============================
    
    # ---------- XGBOOST REGRESSOR ------------------
    @reactive.Calc
    def modelo_xgb():

        df = datos.get()
        if df is None:
            return
        
        #-------------------Pre-Tratamiento del Dataset Panel 7-----------------
        #----Rellenar datos faltantes
        df = imputacion_mice(df)
        df_Panel7 = df.copy()

        # Aplicar Yeo-Johnson + Winsorizado solo a variables sesgadas
        for col in ["x1","x8"]:
            df_Panel7[col] = fix_skew2(df_Panel7[col])
            df_Panel7[col] = winsorize_series(df_Panel7[col])

        X = df_Panel7.drop(columns='deseada')
        y = df_Panel7['deseada']

        # Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.26, random_state=777
        )

        # Entrenamiento XGB

        model = XGBRegressor(       #99/84  97/83   99/85
            n_estimators=2000,      #900    2000            
            learning_rate=0.05,     #0.5    0.05
            max_depth=4,            #3              4
            subsample=0.7,          #0.9    0.7
            colsample_bytree=0.7,   #0.9    0.7
            reg_alpha=1,            #1
            reg_lambda=125,         #200    125
            min_child_weight=5,    #15              5
            random_state=777,       #777
            #early_stopping_rounds=100,
            #eval_metric="rmse",
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        joblib.dump(model, "modelo_xgb.pkl")

        # Predicciones Train/Test
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Importancia variables
        importances = pd.DataFrame({
            'Variable': X.columns,
            'Importancia': model.feature_importances_
        }).sort_values(by='Importancia', ascending=False)

        # -----------------------------
        # Cross-validation 5 folds
        # -----------------------------
        r2_cv = cross_val_score(model, X, y, cv=5, scoring='r2')
        r2_cv_mean = r2_cv.mean()

        return {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "importances": importances,
            "r2_cv_folds": r2_cv,
            "r2_cv_mean": r2_cv_mean
        }
    #-----------------------------------------------
    @output
    @render.ui
    def resumen_modelo():
        resultado = modelo_xgb()
        if not resultado:
            return ui.HTML("⚠️ No hay modelo entrenado todavía.")
        
        r2_train = resultado.get("r2_train", 0)
        r2_test = resultado.get("r2_test", 0)
        r2_cv_mean = resultado.get("r2_cv_mean", 0)
        r2_cv_folds = resultado.get("r2_cv_folds", [])

        texto = (
            f"📌 Número de datos XTrain = {resultado['X_train'].shape}<br>"
            f"y XTest = {resultado['X_test'].shape}<br>"
            f"📌 Número de datos yTrain = {resultado['y_train'].shape}<br>"
            f"y yTest = {resultado['y_test'].shape}<br><br>"
            f"R² Training: {r2_train:.3f}<br>"
            f"R² Test: {r2_test:.3f}<br>"
            f"R² CV (5 folds): {r2_cv_mean:.3f}  [ {', '.join(f'{x:.3f}' for x in r2_cv_folds)} ]"
        )

        return ui.HTML(texto)



    # ---------------------------------------------------------------------------
    @output
    @render.plot
    def grafico_r2():
        resultado = modelo_xgb()
        if not resultado:
            return

        r2_train = resultado.get("r2_train", 0)
        r2_test = resultado.get("r2_test", 0)
        r2_cv_mean = resultado.get("r2_cv_mean", 0)

        categorias = ["Training", "Test", "CV mean"]
        valores = [r2_train, r2_test, r2_cv_mean]

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(categorias, valores, color=["orange", "teal", "purple"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("R²", color="white", fontsize=12)
        ax.set_title("Comparación R²", color="orange", fontsize=14)
        ax.tick_params(colors="white")
        
        # Añadir valor encima de cada barra
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                    ha='center', color='white', fontsize=10)

        # Fondo negro
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(0.7)

        for spine in ax.spines.values():
            spine.set_color("white")

        return fig


    # ---------------------------------------------------------------------------

    # =======================
    # Curva Real vs Predicho (Train y Test)
    # =======================
    @output
    @render.plot
    def curva_real_predicho():
        resultado = modelo_xgb()
        if resultado is None:
            return
        
        y_train, y_test = resultado["y_train"], resultado["y_test"]
        y_train_pred = resultado["y_train_pred"]
        y_test_pred = resultado["y_test_pred"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # --- Training ---
        axes[0].scatter(y_train, y_train_pred, color="skyblue", alpha=0.7, edgecolor="k")
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    "r--", lw=2, label="Ideal")
        axes[0].set_title(f"Training (R² = {resultado['r2_train']:.3f})", color="orange")
        axes[0].set_xlabel("Valores reales")
        axes[0].set_ylabel("Predicciones")
        axes[0].grid(alpha=0.3, linestyle="--")
        axes[0].legend()

        # --- Test ---
        axes[1].scatter(y_test, y_test_pred, color="limegreen", alpha=0.7, edgecolor="k")
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    "r--", lw=2, label="Ideal")
        axes[1].set_title(f"Test (R² = {resultado['r2_test']:.3f})", color="orange")
        axes[1].set_xlabel("Valores reales")
        axes[1].grid(alpha=0.3, linestyle="--")
        axes[1].legend()

        # Estilo general
        #fig.suptitle("Predicciones vs Reales", fontsize=16, color="white", fontweight="bold")
        fig.patch.set_facecolor("black")
        for ax in axes:
            ax.set_facecolor("black")
            ax.title.set_color("orange")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.tick_params(colors="white")

        plt.tight_layout()
        return fig


    #-----------------------------------------------------------------------------
    @output
    @render.plot
    def grafico_importancia():
        resultado = modelo_xgb()
        if resultado is None:
            return

        importances = resultado['importances']
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(importances['Variable'], importances['Importancia'], color='teal')
        ax.set_xlabel('Importancia', color='orange', fontsize=14)
        ax.set_title('Importancia de Variables (XGBoost)', color='orange', fontsize=16)
        ax.tick_params(colors='white', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        fig.patch.set_alpha(0.7)
        ax.invert_yaxis()
        return fig
    
# =============================================================================
# ===================== SERVER 7.b: Comparativa 3 modelos =====================

    # --- Función principal para calcular todos los modelos ---
    @reactive.Calc
    def comparativa_modelos():
        df = datos.get()  # tu dataframe reactive
        if df is None:
            return

        # Pre-tratamiento
        df = imputacion_mice(df)
        df_Panel7 = df.copy()
        for col in ["x1", "x8"]:
            df_Panel7[col] = fix_skew2(df_Panel7[col])
            df_Panel7[col] = winsorize_series(df_Panel7[col])

        X = df_Panel7.drop(columns='deseada')
        y = df_Panel7['deseada']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.26, random_state=777
        )

        modelos = {
            "LinearRegression": LinearRegression(),
            "XGBoost": XGBRegressor(
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1,
                reg_lambda=125,
                min_child_weight=5,
                random_state=777
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                random_state=777
            ),
            "HistGB": HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=5,
                random_state=777
            )
        }

        resultados = []

        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            y_train_pred = modelo.predict(X_train)
            y_test_pred = modelo.predict(X_test)
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            r2_cv_mean = cross_val_score(modelo, X, y, cv=5, scoring='r2').mean()

            joblib.dump(modelo, f"modelo_{nombre}.pkl")

            if hasattr(modelo, 'feature_importances_'):
                importances = pd.DataFrame({
                    'Variable': X.columns,
                    'Importancia': modelo.feature_importances_
                }).sort_values(by='Importancia', ascending=False)
            else:
                importances = None

            resultados.append({
                "modelo": nombre,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "r2_cv_mean": r2_cv_mean,
                "importances": importances,
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred
            })

        return {
            "resultados": resultados,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    # --- Helper para obtener resultados de un modelo específico ---
    def get_resultado_modelo(res, nombre_modelo):
        for r in res["resultados"]:
            if r["modelo"] == nombre_modelo:
                return r
        return None

    # ----------- OUTPUTS RESUMENES (uno por modelo) -------------

    # ----------- OUTPUT RESUMEN LINEARREG ---
    @output
    @render.ui
    def resumen_comparativa_lr():
        res = comparativa_modelos()
        if res is None:
            return ui.HTML("⚠️ No disponible todavía.")

        r = get_resultado_modelo(res, "LinearRegression")
        if r is None:
            return ui.HTML("⚠️ No hay resultados LR.")
        
        return ui.HTML(
            f"<b>LinearRegression</b><br>Simple y explicativo, limitado cuando no hay relaciones linales.<br>"
            f"R² Train = {r['r2_train']:.3f}<br>"
            f"R² Test = {r['r2_test']:.3f}<br>"
            f"R² CV = {r['r2_cv_mean']:.3f}"
        )


   # ------------ OUTPUT RESUMEN XGBReg ---
    @output
    @render.ui
    def resumen_comparativa_xgb():
        res = comparativa_modelos()
        if res is None:
            return ui.HTML("⚠️ No disponible todavía.")

        r = get_resultado_modelo(res, "XGBoost")
        if r is None:
            return ui.HTML("⚠️ No hay resultados XGB.")
        
        return ui.HTML(
            f"<b>XGBReg</b><br>Preciso, con regularización y control fino, ideal para datos tabulares medianos.<br>"
            f"R² Train = {r['r2_train']:.3f}<br>"
            f"R² Test = {r['r2_test']:.3f}<br>"
            f"R² CV = {r['r2_cv_mean']:.3f}"
        )


    @output
    @render.ui
    def resumen_comparativa_rf():
        res = comparativa_modelos()
        if res is None:
            return ui.HTML("⚠️ No disponible todavía.")

        r = get_resultado_modelo(res, "RandomForest")
        if r is None:
            return ui.HTML("⚠️ No hay resultados RF.")
        
        return ui.HTML(
            f"<b>RandomForest</b><br>Simple y robusto, buena base sin tuning.<br>"
            f"R² Train = {r['r2_train']:.3f}<br>"
            f"R² Test = {r['r2_test']:.3f}<br>"
            f"R² CV = {r['r2_cv_mean']:.3f}"
        )


    @output
    @render.ui
    def resumen_comparativa_hgb():
        res = comparativa_modelos()
        if res is None:
            return ui.HTML("⚠️ No disponible todavía.")

        r = get_resultado_modelo(res, "HistGB")
        if r is None:
            return ui.HTML("⚠️ No hay resultados HistGB.")
        
        return ui.HTML(
            f"<b>HistGradientBoosting</b><br>Versión más ligera/eficiente, ideal para grandes datasets.<br>"
            f"R² Train = {r['r2_train']:.3f}<br>"
            f"R² Test = {r['r2_test']:.3f}<br>"
            f"R² CV = {r['r2_cv_mean']:.3f}"
        )

    # ===================== OUTPUT: gráfico R² XGB =====================
    @output
    @render.plot
    def grafico_r2_xgb():
        res = comparativa_modelos()
        if res is None:
            return

        r = next((rr for rr in res["resultados"] if rr["modelo"] == "XGBoost"), None)
        if r is None:
            return

        labels = ["Train", "Test", "CV"]
        valores = [r["r2_train"], r["r2_test"], r["r2_cv_mean"]]

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(labels, valores, width=0.6, color=["orange", "teal", "purple"])

        ax.set_ylim(0, 1)
        ax.set_ylabel("R²")
        ax.set_title("R² - XGBoost", color="orange")

        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)

        plt.tight_layout()
        return fig




    # ===================== OUTPUT: gráfico R² RandomForest =====================

    # --- OUTPUT GRÁFICO LINEARREG ---
    @output
    @render.plot
    def grafico_r2_lr():
        res = comparativa_modelos()
        if res is None:
            return

        r = get_resultado_modelo(res, "LinearRegression")
        if r is None:
            return

        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(["Train", "Test", "CV"],
            [r['r2_train'], r['r2_test'], r['r2_cv_mean']],
            color=["orange", "teal", "purple"])
        
        ax.set_ylim(0, 1)
        ax.set_ylabel("R²", color="white")
        ax.set_title("R² - LinearRegression", color="orange")

        # Estilo oscuro
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.tick_params(colors="white")

        plt.tight_layout()
        return fig

    @output
    @render.plot
    def grafico_r2_rf():
        res = comparativa_modelos()
        if res is None:
            return

        r = next((rr for rr in res["resultados"] if rr["modelo"] == "RandomForest"), None)
        if r is None:
            return

        labels = ["Train", "Test", "CV"]
        valores = [r["r2_train"], r["r2_test"], r["r2_cv_mean"]]

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(labels, valores, width=0.6, color=["orange", "teal", "purple"])

        ax.set_ylim(0, 1)
        ax.set_ylabel("R²")
        ax.set_title("R² - RandomForest", color="orange")

        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)

        plt.tight_layout()
        return fig


    # ===================== OUTPUT: gráfico R² HistGradientBoosting =====================
    @output
    @render.plot
    def grafico_r2_hgb():
        res = comparativa_modelos()
        if res is None:
            return

        r = next((rr for rr in res["resultados"] if rr["modelo"] == "HistGB"), None)
        if r is None:
            return

        labels = ["Train", "Test", "CV"]
        valores = [r["r2_train"], r["r2_test"], r["r2_cv_mean"]]

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(labels, valores, width=0.6, color=["orange", "teal", "purple"])

        ax.set_ylim(0, 1)
        ax.set_ylabel("R²")
        ax.set_title("R² - HistGB", color="orange")

        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)

        plt.tight_layout()
        return fig


# =================================================================================
# ---------- Server Panel 8: Carga y visualización inicial de datos XLSX ----------
# =================================================================================

    @reactive.Calc
    def datos_a_predecir():
        """Carga únicamente archivos XLSX y devuelve DataFrame"""
        uploaded_file = input.input_file_pred()
        
        if not uploaded_file:
            return pd.DataFrame()  # <- Devuelve siempre un DataFrame
        
        # Tomar el primer archivo si es lista
        uploaded_file = uploaded_file[0] if isinstance(uploaded_file, list) else uploaded_file

        filename = uploaded_file.get("name", "")
        ext = filename.split('.')[-1].lower()

        if ext not in ["xlsx", "xls"]:
            print(f"Formato no soportado: {ext}")
            return pd.DataFrame()  # <- Devuelve siempre un DataFrame

        try:
            df = pd.read_excel(uploaded_file["datapath"])
        except Exception as e:
            print("Error al leer el archivo:", e)
            return pd.DataFrame()  # <- Devuelve siempre un DataFrame

        return df
    
    # =======================
    # Variable reactiva global para columnas transformadas
    # Variable reactiva global para importancias
    # =======================
    cols_transformadas = reactive.Value([])
    importancias_modelo = reactive.Value(pd.DataFrame())

    def predicciones():
        df_nuevo_original = datos_a_predecir()
        if df_nuevo_original.empty:
            cols_transformadas.set([])  
            importancias_modelo.set(pd.DataFrame())  
            return pd.DataFrame()

        try:
            # ------------------- Imputación de NaN -----------------
            imputer = IterativeImputer(random_state=777, max_iter=20)
            df_prediccion = pd.DataFrame(
                imputer.fit_transform(df_nuevo_original),
                columns=df_nuevo_original.columns
            )

            # ------------------- Cargar modelo -----------------
            modelo_data = joblib.load("modelo_xgb.pkl")
            if isinstance(modelo_data, dict) and "model" in modelo_data:
                modelo = modelo_data["model"]
                importances = modelo_data.get("importances")
            else:
                modelo = modelo_data
                importances = None

            # ------------------- Calcular importancias (solo informativo) -----------------
            if importances is None and hasattr(modelo, "feature_importances_"):
                importances = pd.DataFrame({
                    "Variable": df_prediccion.columns,
                    "Importancia": modelo.feature_importances_
                }).sort_values(by="Importancia", ascending=False)

                # Guardar modelo actualizado
                joblib.dump({"model": modelo, "importances": importances}, "modelo_xgb.pkl")

            if importances is not None:
                importancias_modelo.set(importances)

            # ------------------- Transformaciones fijas (coherentes con entrenamiento) -----------------
            cols_fijas = ["x1", "x8"]
            for col in cols_fijas:
                if col in df_prediccion.columns:
                    df_prediccion[col] = fix_skew2(df_prediccion[col])
                    df_prediccion[col] = winsorize_series(df_prediccion[col])

            # Guardar columnas transformadas (solo para UI)
            cols_transformadas.set(cols_fijas)

            # ------------------- Generar predicciones -----------------
            df_prediccion["deseada"] = modelo.predict(df_prediccion)

        except Exception as e:
            print("Error al generar predicciones:", e)
            cols_transformadas.set([])  
            importancias_modelo.set(pd.DataFrame())
            df_prediccion["deseada"] = None

        return df_prediccion



    # =======================
    # Mostrar columnas transformadas en la UI
    # =======================

    @output
    @render.ui
    def info_transformadas():
        cols = cols_transformadas.get()
        if not cols:
            return #ui.HTML("ℹ️ <b>No se aplicaron transformaciones a ninguna columna.</b>")
        texto = (f"<span style='font-weight:bold; color:black;'> Variables transformadas (Yao+Winsor): " + ", ".join(cols) +"</span>")
    
        return ui.HTML(texto)



    # =======================
    # =======================
    # Tabla de predicciones (solo UI) con 3 decimales en 'prediccion_deseada'
    # =======================
    @output
    @render.data_frame
    def tabla_predicciones():
        df_pred = predicciones()
        if df_pred is None or df_pred.empty:
            return pd.DataFrame({"Info": ["⚠️ No hay datos para mostrar"]})

        # Copiar y renombrar columna
        df_vista = df_pred.copy()
        if "deseada" in df_vista.columns:
            df_vista.rename(columns={"deseada": "prediccion_deseada"}, inplace=True)
            df_vista["prediccion_deseada"] = pd.to_numeric(df_vista["prediccion_deseada"], errors="coerce")\
                                            .map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

        return df_vista[["prediccion_deseada"]]  # Solo mostrar esa columna

    
    # ---------------------------------------------------------------------------
    @output
    @render.plot
    def grafico_importancia_8():
        importances = importancias_modelo.get()  # Recuperamos la variable reactiva
        if importances is None or importances.empty:
            return  # No dibujamos nada si no hay datos

        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(importances['Variable'], importances['Importancia'], color='teal')
        ax.set_xlabel('Importancia', color='orange', fontsize=14)
        ax.set_title('Importancia de Variables (XGBoost)', color='orange', fontsize=16)
        ax.tick_params(colors='white', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        fig.patch.set_alpha(0.7)
        ax.invert_yaxis()
        return fig

        
    # =======================
    # Descargar solo la columna "prediccion_deseada"
    # =======================
    @render.download(filename="predicciones.xlsx")
    def download_predicciones():
        df_pred = predicciones()
        if df_pred.empty or "deseada" not in df_pred.columns:
            return None  # No genera archivo si no hay datos

        # Crear DataFrame solo con la columna renombrada
        df_solo = pd.DataFrame({
            "prediccion_deseada": pd.to_numeric(df_pred["deseada"], errors="coerce")
        })

        # Crear archivo temporal con pathlib
        tmpdir = Path(tempfile.mkdtemp())
        filepath = tmpdir / "predicciones.xlsx"

        # Guardar el DataFrame en Excel
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df_solo.to_excel(writer, index=False, sheet_name="Predicciones")

        return str(filepath)


    # =======================
    # Descargar dataset original + columna "prediccion_deseada"
    # =======================
    @render.download(filename="datos_con_predicciones.xlsx")
    def download_dataset_predicciones():
        df_pred = predicciones()
        df_original = datos_a_predecir()

        if df_pred.empty or "deseada" not in df_pred.columns or df_original.empty:
            return None  # Nada que descargar

        # Asegurar consistencia en tamaño (por si algún error)
        if len(df_original) != len(df_pred):
            print("⚠️ Tamaño de df_original y df_pred no coincide")
            return None

        # Copia del original + añadir la columna predicción
        df_final = df_original.copy()
        df_final["deseada"] = pd.to_numeric(df_pred["deseada"], errors="coerce")

        # Crear archivo temporal
        tmpdir = Path(tempfile.mkdtemp())
        filepath = tmpdir / "datos_con_predicciones.xlsx"

        # Guardar en Excel
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df_final.to_excel(writer, index=False, sheet_name="Datos+Predicciones")

        return str(filepath)





# ====================================================================================================
# APP
# ====================================================================================================
app = App(app_ui, server, static_assets=datos_dir)
