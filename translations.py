"""Language options and translations for the INTERGROWTH-21st Preterm Growth Tracker."""

from typing import Dict, Any

# Available language options
LANGUAGE_OPTIONS = {
    "English": "en",
    "Español": "es",
    "Português (Brasil)": "pt_BR"
}

# Translation dictionaries
TRANSLATIONS = {
    "en": {
        # Page title and headers
        "app_title": "INTERGROWTH-21st Preterm Growth Tracker",
        "instructions_title": "Instructions",
        "patient_info_title": "Patient Information",
        "add_measurement_title": "Add New Measurement",
        "charts_title": "Growth Charts",
        "history_title": "Patient Measurement History",
        
        # Instructions
        "instructions": "Instructions",
        "instructions_content": """
        ### How to use this app
        
        1. **Enter patient information**
           - Enter gestational age at birth (weeks and days)
           - Enter birth date
           - Select sex
           - Optionally enter patient name for reports
        
        2. **Add measurements**
           - Enter postmenstrual age (PMA) or measurement date
           - Enter weight (g), length (cm), and/or head circumference (cm)
           - Click 'Add Measurement'
        
        3. **Import/Export data**
           - Export data as CSV with the 'Export Data' button
           - Import previously exported data with the 'Import Data' button
           - CSV columns: PMA, Date, Weight, Length, HC
           - Export PDF report with all charts
        
        4. **View charts**
           - Switch between tabs to view different measurements
           - Each point shows the measurement and PMA
           - Toggle between percentiles and z-scores
        
        5. **Manage data**
           - Sort table by clicking column headers
           - Use delete buttons to remove individual measurements
           - Use 'Clear All Data' to start over
        """,
        
        # Patient information section
        "birth_ga_label": "Gestational Age at Birth",
        "weeks_label": "weeks",
        "days_label": "days",
        "birth_date_label": "Birth Date",
        "sex_label": "Sex",
        "male_option": "Male",
        "female_option": "Female",
        "patient_name_label": "Patient Name (optional)",
        
        # Add measurement section
        "measurement_method_label": "Measurement Method",
        "by_pma_option": "By PMA",
        "by_date_option": "By Date",
        "pma_label": "Postmenstrual Age (PMA)",
        "measurement_date_label": "Measurement Date",
        "weight_label": "Weight (g)",
        "length_label": "Length (cm)",
        "hc_label": "Head Circumference (cm)",
        "add_measurement_button": "Add Measurement",
        
        # Chart section
        "display_mode_label": "Display Mode",
        "percentiles_option": "Percentiles",
        "z_scores_option": "Z-Scores",
        "weight_tab": "Weight",
        "length_tab": "Length",
        "hc_tab": "Head Circumference",
        
        # Data management
        "export_data_button": "Export Data",
        "import_data_button": "Import Data",
        "export_pdf_button": "Export PDF Report",
        "clear_data_button": "Clear All Data",
        "delete_button": "Delete",
        "no_measurements_message": "No measurements added yet. Use the sidebar to add data points.",
        
        # Table headers
        "pma_column": "PMA",
        "date_column": "Date",
        "chronological_age_column": "Chronological Age",
        "corrected_age_column": "Corrected Age",
        "weight_column": "Weight (g)",
        "length_column": "Length (cm)",
        "hc_column": "HC (cm)",
        "z_score_column": "Z-Score",
        "percentile_column": "Percentile",
        
        # Alerts and messages
        "data_cleared_message": "All patient data has been cleared.",
        "invalid_input_message": "Please enter valid values.",
        "measurement_added_message": "Measurement added successfully.",
        "import_success_message": "Data imported successfully.",
        "import_error_message": "Error importing data. Please check the file format.",
        
        # Cookie consent
        "cookie_consent_title": "This website uses cookies",
        "cookie_consent_message": "We use cookies to remember your language preference and improve your experience.",
        "cookie_accept_button": "Accept",
        "cookie_decline_button": "Decline",
        
        # Footer
        "footer_data_source": "Based on data from",
        "footer_intergrowth_link": "INTERGROWTH-21st Preterm Growth Standards",
        "footer_developed_by": "Developed by Maycon Queiros",
        "footer_source_code": "Source Code on GitHub"
    },
    
    # Spanish translations
    "es": {
        # Page title and headers
        "app_title": "Seguimiento del Crecimiento de Prematuros INTERGROWTH-21st",
        "instructions_title": "Instrucciones",
        "patient_info_title": "Información del Paciente",
        "add_measurement_title": "Añadir Nueva Medición",
        "charts_title": "Gráficos de Crecimiento",
        "history_title": "Historial de Mediciones del Paciente",
        
        # Instructions
        "instructions": "Instrucciones",
        "instructions_content": """
        ### Cómo usar esta aplicación
        
        1. **Ingrese la información del paciente**
           - Ingrese la edad gestacional al nacer (semanas y días)
           - Ingrese la fecha de nacimiento
           - Seleccione el sexo
           - Opcionalmente ingrese el nombre del paciente para los informes
        
        2. **Añadir mediciones**
           - Ingrese la edad postmenstrual (EPM) o la fecha de medición
           - Ingrese peso (g), longitud (cm) y/o circunferencia cefálica (cm)
           - Haga clic en 'Añadir Medición'
        
        3. **Importar/Exportar datos**
           - Exporte datos como CSV con el botón 'Exportar Datos'
           - Importe datos previamente exportados con el botón 'Importar Datos'
           - Columnas CSV: EPM, Fecha, Peso, Longitud, CC
           - Exporte informe PDF con todos los gráficos
        
        4. **Ver gráficos**
           - Cambie entre pestañas para ver diferentes mediciones
           - Cada punto muestra la medición y la EPM
           - Alterne entre percentiles y puntuaciones z
        
        5. **Gestionar datos**
           - Ordene la tabla haciendo clic en los encabezados de columna
           - Use los botones de eliminar para quitar mediciones individuales
           - Use 'Borrar Todos los Datos' para comenzar de nuevo
        """,
        
        # Patient information section
        "birth_ga_label": "Edad Gestacional al Nacer",
        "weeks_label": "semanas",
        "days_label": "días",
        "birth_date_label": "Fecha de Nacimiento",
        "sex_label": "Sexo",
        "male_option": "Masculino",
        "female_option": "Femenino",
        "patient_name_label": "Nombre del Paciente (opcional)",
        
        # Add measurement section
        "measurement_method_label": "Método de Medición",
        "by_pma_option": "Por EPM",
        "by_date_option": "Por Fecha",
        "pma_label": "Edad Postmenstrual (EPM)",
        "measurement_date_label": "Fecha de Medición",
        "weight_label": "Peso (g)",
        "length_label": "Longitud (cm)",
        "hc_label": "Circunferencia Cefálica (cm)",
        "add_measurement_button": "Añadir Medición",
        
        # Chart section
        "display_mode_label": "Modo de Visualización",
        "percentiles_option": "Percentiles",
        "z_scores_option": "Puntuaciones Z",
        "weight_tab": "Peso",
        "length_tab": "Longitud",
        "hc_tab": "Circunferencia Cefálica",
        
        # Data management
        "export_data_button": "Exportar Datos",
        "import_data_button": "Importar Datos",
        "export_pdf_button": "Exportar Informe PDF",
        "clear_data_button": "Borrar Todos los Datos",
        "delete_button": "Eliminar",
        "no_measurements_message": "Aún no se han añadido mediciones. Use la barra lateral para añadir puntos de datos.",
        
        # Table headers
        "pma_column": "EPM",
        "date_column": "Fecha",
        "chronological_age_column": "Edad Cronológica",
        "corrected_age_column": "Edad Corregida",
        "weight_column": "Peso (g)",
        "length_column": "Longitud (cm)",
        "hc_column": "CC (cm)",
        "z_score_column": "Puntuación Z",
        "percentile_column": "Percentil",
        
        # Alerts and messages
        "data_cleared_message": "Todos los datos del paciente han sido borrados.",
        "invalid_input_message": "Por favor ingrese valores válidos.",
        "measurement_added_message": "Medición añadida con éxito.",
        "import_success_message": "Datos importados con éxito.",
        "import_error_message": "Error al importar datos. Por favor verifique el formato del archivo.",
        
        # Cookie consent
        "cookie_consent_title": "Este sitio web utiliza cookies",
        "cookie_consent_message": "Utilizamos cookies para recordar su preferencia de idioma y mejorar su experiencia.",
        "cookie_accept_button": "Aceptar",
        "cookie_decline_button": "Rechazar",
        
        # Footer
        "footer_data_source": "Basado en datos de",
        "footer_intergrowth_link": "Estándares de Crecimiento de Prematuros INTERGROWTH-21st",
        "footer_developed_by": "Desarrollado por Maycon Queiros",
        "footer_source_code": "Código Fuente en GitHub"
    },
    
    # Brazilian Portuguese translations
    "pt_BR": {
        # Page title and headers
        "app_title": "Rastreador de Crescimento de Prematuros INTERGROWTH-21st",
        "instructions_title": "Instruções",
        "patient_info_title": "Informações do Paciente",
        "add_measurement_title": "Adicionar Nova Medição",
        "charts_title": "Gráficos de Crescimento",
        "history_title": "Histórico de Medições do Paciente",
        
        # Instructions
        "instructions": "Instruções",
        "instructions_content": """
        ### Como usar este aplicativo
        
        1. **Insira as informações do paciente**
           - Insira a idade gestacional ao nascer (semanas e dias)
           - Insira a data de nascimento
           - Selecione o sexo
           - Opcionalmente insira o nome do paciente para relatórios
        
        2. **Adicionar medições**
           - Insira a idade pós-menstrual (IPM) ou data da medição
           - Insira peso (g), comprimento (cm) e/ou perímetro cefálico (cm)
           - Clique em 'Adicionar Medição'
        
        3. **Importar/Exportar dados**
           - Exporte dados como CSV com o botão 'Exportar Dados'
           - Importe dados previamente exportados com o botão 'Importar Dados'
           - Colunas CSV: IPM, Data, Peso, Comprimento, PC
           - Exporte relatório PDF com todos os gráficos
        
        4. **Visualizar gráficos**
           - Alterne entre abas para ver diferentes medições
           - Cada ponto mostra a medição e a IPM
           - Alterne entre percentis e escores-z
        
        5. **Gerenciar dados**
           - Ordene a tabela clicando nos cabeçalhos das colunas
           - Use os botões de excluir para remover medições individuais
           - Use 'Limpar Todos os Dados' para recomeçar
        """,
        
        # Patient information section
        "birth_ga_label": "Idade Gestacional ao Nascer",
        "weeks_label": "semanas",
        "days_label": "dias",
        "birth_date_label": "Data de Nascimento",
        "sex_label": "Sexo",
        "male_option": "Masculino",
        "female_option": "Feminino",
        "patient_name_label": "Nome do Paciente (opcional)",
        
        # Add measurement section
        "measurement_method_label": "Método de Medição",
        "by_pma_option": "Por IPM",
        "by_date_option": "Por Data",
        "pma_label": "Idade Pós-Menstrual (IPM)",
        "measurement_date_label": "Data da Medição",
        "weight_label": "Peso (g)",
        "length_label": "Comprimento (cm)",
        "hc_label": "Perímetro Cefálico (cm)",
        "add_measurement_button": "Adicionar Medição",
        
        # Chart section
        "display_mode_label": "Modo de Exibição",
        "percentiles_option": "Percentis",
        "z_scores_option": "Escores-Z",
        "weight_tab": "Peso",
        "length_tab": "Comprimento",
        "hc_tab": "Perímetro Cefálico",
        
        # Data management
        "export_data_button": "Exportar Dados",
        "import_data_button": "Importar Dados",
        "export_pdf_button": "Exportar Relatório PDF",
        "clear_data_button": "Limpar Todos os Dados",
        "delete_button": "Excluir",
        "no_measurements_message": "Nenhuma medição adicionada ainda. Use a barra lateral para adicionar pontos de dados.",
        
        # Table headers
        "pma_column": "IPM",
        "date_column": "Data",
        "chronological_age_column": "Idade Cronológica",
        "corrected_age_column": "Idade Corrigida",
        "weight_column": "Peso (g)",
        "length_column": "Comprimento (cm)",
        "hc_column": "PC (cm)",
        "z_score_column": "Escore-Z",
        "percentile_column": "Percentil",
        
        # Alerts and messages
        "data_cleared_message": "Todos os dados do paciente foram limpos.",
        "invalid_input_message": "Por favor, insira valores válidos.",
        "measurement_added_message": "Medição adicionada com sucesso.",
        "import_success_message": "Dados importados com sucesso.",
        "import_error_message": "Erro ao importar dados. Por favor, verifique o formato do arquivo.",
        
        # Cookie consent
        "cookie_consent_title": "Este site usa cookies",
        "cookie_consent_message": "Usamos cookies para lembrar sua preferência de idioma e melhorar sua experiência.",
        "cookie_accept_button": "Aceitar",
        "cookie_decline_button": "Recusar",
        
        # Footer
        "footer_data_source": "Baseado em dados de",
        "footer_intergrowth_link": "Padrões de Crescimento de Prematuros INTERGROWTH-21st",
        "footer_developed_by": "Desenvolvido por Maycon Queiros",
        "footer_source_code": "Código Fonte no GitHub"
    }
}


def get_translation(key: str, language_code: str) -> str:
    """Get a translation for a given key and language code.
    
    Args:
        key: The translation key
        language_code: The language code (e.g., 'en', 'es', 'pt_BR')
    
    Returns:
        str: The translated string or the key itself if translation is not found
    """
    if language_code not in TRANSLATIONS:
        language_code = "en"  # Default to English if language not found
        
    return TRANSLATIONS[language_code].get(key, key)