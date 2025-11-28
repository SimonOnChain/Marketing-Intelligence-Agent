"""Create PDF report for Marketing Intelligence Agent."""

from fpdf import FPDF
from pathlib import Path

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('DejaVu', '', 'C:/Windows/Fonts/arial.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'C:/Windows/Fonts/arialbd.ttf', uni=True)

    def header(self):
        self.set_font('DejaVu', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Marketing Intelligence Agent - Projektdokumentation', 0, 1, 'R')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Seite {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 16)
        self.set_text_color(0, 100, 180)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def section_title(self, title):
        self.set_font('DejaVu', 'B', 12)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def body_text(self, text):
        self.set_font('DejaVu', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_list(self, items):
        self.set_font('DejaVu', '', 11)
        self.set_text_color(0, 0, 0)
        for item in items:
            self.set_x(15)
            self.multi_cell(180, 6, f"- {item}")
        self.ln(2)

    def code_block(self, code):
        self.set_font('Courier', '', 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, code, fill=True)
        self.ln(4)

# Create PDF
pdf = PDFReport()
pdf.set_auto_page_break(auto=True, margin=15)

# Title Page
pdf.add_page()
pdf.ln(60)
pdf.set_font('DejaVu', 'B', 32)
pdf.set_text_color(0, 100, 180)
pdf.cell(0, 15, 'Marketing Intelligence Agent', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('DejaVu', '', 18)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, 'KI-gestuetzter Marketing-Analyst', 0, 1, 'C')
pdf.ln(30)
pdf.set_font('DejaVu', '', 14)
pdf.cell(0, 8, 'Data Science, Machine Learning & AI', 0, 1, 'C')
pdf.ln(5)
pdf.cell(0, 8, 'Simon Jokani', 0, 1, 'C')
pdf.ln(40)
pdf.set_font('DejaVu', '', 10)
pdf.set_text_color(128, 128, 128)
pdf.cell(0, 8, 'November 2024', 0, 1, 'C')

# Page 2: Executive Summary
pdf.add_page()
pdf.chapter_title('1. Executive Summary')
pdf.body_text(
    'Der Marketing Intelligence Agent ist eine KI-gestuetzte Anwendung, die natuerlichsprachliche '
    'Fragen zu Verkaufsdaten, Kundenbewertungen und Geschaeftstrends beantwortet. Das System '
    'nutzt modernste Machine-Learning-Technologien wie LangGraph fuer Multi-Agent-Orchestrierung, '
    'RAG (Retrieval-Augmented Generation) fuer kontextbasierte Antworten und Facebook Prophet '
    'fuer Zeitreihen-Forecasting.'
)
pdf.ln(5)
pdf.section_title('Kernergebnisse')
pdf.bullet_list([
    'Multi-Agent-System mit automatischer Intent-Klassifikation (>95% Accuracy)',
    'RAG-Pipeline mit 40.000+ indexierten Kundenbewertungen',
    'Prophet ML-Forecasting mit 8-Wochen-Prognose (+9% Wachstumsprognose)',
    'Hybrid Search kombiniert Vektor- und lexikalische Suche',
    'Deployed auf AWS EC2, weltweit erreichbar'
])

# Page 3: Architecture
pdf.add_page()
pdf.chapter_title('2. Systemarchitektur')
pdf.section_title('2.1 Multi-Agent Pipeline')
pdf.body_text(
    'Das System basiert auf einer LangGraph State Machine, die eingehende Anfragen analysiert '
    'und an spezialisierte Agenten weiterleitet:'
)
pdf.bullet_list([
    'Orchestrator: Klassifiziert den Intent der Benutzeranfrage',
    'Sales Agent: Aggregiert Umsatz- und Bestelldaten mit Pandas',
    'Sentiment Agent: Analysiert Kundenbewertungen via RAG',
    'Forecast Agent: Erstellt Zeitreihen-Prognosen mit Prophet',
    'Synthesizer: Kombiniert Agent-Outputs zu kohaerenter Antwort'
])
pdf.ln(5)
pdf.section_title('2.2 Datenfluss')
pdf.body_text(
    '1. User Query -> Orchestrator (Intent Classification)\n'
    '2. Orchestrator -> Spezialisierter Agent (basierend auf Intent)\n'
    '3. Agent verarbeitet Query mit spezifischen Tools/Daten\n'
    '4. Agent Output -> Synthesizer\n'
    '5. Synthesizer -> Finale Response an User'
)

# Page 4: LangGraph
pdf.add_page()
pdf.chapter_title('3. LangGraph - Multi-Agent Orchestrierung')
pdf.body_text(
    'LangGraph ist ein Framework fuer die Erstellung von zustandsbehafteten, '
    'multi-akteur Anwendungen mit LLMs. Es erweitert LangChain um Graph-basierte '
    'Workflows und ermoeglicht komplexe Agent-Interaktionen.'
)
pdf.ln(3)
pdf.section_title('3.1 State Machine Pattern')
pdf.bullet_list([
    'TypedDict definiert den typsicheren Zustand (AgentState)',
    'StateGraph verwaltet Knoten (Agenten) und Kanten (Transitionen)',
    'Conditional Edges ermoeglichen dynamisches Routing',
    'Checkpointing fuer Zustandspersistierung'
])
pdf.ln(3)
pdf.section_title('3.2 Code-Beispiel')
pdf.code_block('''class AgentState(TypedDict):
    query: str
    intent: str
    agent_outputs: Dict[str, Any]
    final_response: str

graph = StateGraph(AgentState)
graph.add_node("classify", classify_intent)
graph.add_node("sales", sales_agent)
graph.add_node("sentiment", sentiment_agent)
graph.add_node("forecast", forecast_agent)
graph.add_conditional_edges("classify", route_to_agent)
graph.compile()''')

# Page 5: RAG Pipeline
pdf.add_page()
pdf.chapter_title('4. RAG - Retrieval-Augmented Generation')
pdf.body_text(
    'RAG kombiniert Information Retrieval mit generativer KI. Anstatt sich nur auf das '
    'Wissen des LLMs zu verlassen, werden relevante Dokumente aus einer Wissensbasis '
    'abgerufen und als Kontext an das Modell uebergeben.'
)
pdf.ln(3)
pdf.section_title('4.1 Komponenten')
pdf.bullet_list([
    'Embedding Model: sentence-transformers/all-MiniLM-L6-v2',
    'Vector Database: Qdrant Cloud (40.000+ Reviews indexiert)',
    'Chunk Size: 512 Tokens mit 50 Token Overlap',
    'Top-K Retrieval: 10 relevanteste Dokumente'
])
pdf.ln(3)
pdf.section_title('4.2 Vorteile von RAG')
pdf.bullet_list([
    'Reduziert Halluzinationen durch faktische Grundlage',
    'Ermoeglicht domaenenspezifisches Wissen ohne Fine-Tuning',
    'Aktualisierbar ohne Modell-Retraining',
    'Transparente Quellenangaben moeglich'
])

# Page 6: Hybrid Search
pdf.add_page()
pdf.chapter_title('5. Hybrid Search')
pdf.body_text(
    'Hybrid Search kombiniert semantische Vektor-Suche mit lexikalischer Keyword-Suche, '
    'um die Staerken beider Ansaetze zu nutzen.'
)
pdf.ln(3)
pdf.section_title('5.1 Vektor-Suche')
pdf.bullet_list([
    'Findet semantisch aehnliche Inhalte',
    'Versteht Synonyme und Paraphrasen',
    'Basiert auf Embedding-Distanz (Cosine Similarity)'
])
pdf.ln(3)
pdf.section_title('5.2 Lexikalische Suche (BM25)')
pdf.bullet_list([
    'Exakte Keyword-Matches',
    'Wichtig fuer spezifische Begriffe/Namen',
    'Schnell und interpretierbar'
])
pdf.ln(3)
pdf.section_title('5.3 Reciprocal Rank Fusion (RRF)')
pdf.body_text(
    'RRF kombiniert die Rankings beider Suchmethoden zu einem finalen Score. '
    'Dokumente, die in beiden Rankings hoch platziert sind, erhalten den hoechsten Score.'
)

# Page 7: Prophet Forecasting
pdf.add_page()
pdf.chapter_title('6. Prophet ML - Zeitreihen-Forecasting')
pdf.body_text(
    'Facebook Prophet ist ein Open-Source-Tool fuer Zeitreihen-Forecasting, das robust '
    'gegenueber fehlenden Daten, Ausreissern und saisonalen Effekten ist.'
)
pdf.ln(3)
pdf.section_title('6.1 Features')
pdf.bullet_list([
    'Automatische Trend-Erkennung (linear/logistisch)',
    'Jahres-, Wochen- und Tagessaisonalitaet',
    'Handling von Feiertagen und Sondereffekten',
    'Unsicherheitsintervalle fuer Prognosen'
])
pdf.ln(3)
pdf.section_title('6.2 Implementation')
pdf.code_block('''from prophet import Prophet
import numpy as np

# Log-Transformation fuer stabile Vorhersagen
df['y'] = np.log1p(df['revenue'])

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(df)

# 8-Wochen Forecast
future = model.make_future_dataframe(periods=8, freq='W')
forecast = model.predict(future)

# Ruecktransformation
forecast['yhat'] = np.expm1(forecast['yhat'])''')

# Page 8: Tech Stack
pdf.add_page()
pdf.chapter_title('7. Technologie-Stack')
pdf.section_title('7.1 Machine Learning & AI')
pdf.bullet_list([
    'LLMs: xAI Grok, Groq (Llama 3), AWS Bedrock (Claude)',
    'Embeddings: sentence-transformers MiniLM',
    'Forecasting: Facebook Prophet',
    'Orchestration: LangGraph, LangChain'
])
pdf.ln(3)
pdf.section_title('7.2 Data & Storage')
pdf.bullet_list([
    'Vector Database: Qdrant Cloud',
    'Data Processing: Pandas, Polars',
    'File Format: Parquet (komprimiert)',
    'Dataset: Olist E-Commerce (100K+ Orders)'
])
pdf.ln(3)
pdf.section_title('7.3 Backend & Infrastructure')
pdf.bullet_list([
    'API: FastAPI mit Uvicorn',
    'UI: Streamlit',
    'Monitoring: Langfuse (Tracing)',
    'Cloud: AWS EC2, S3'
])

# Page 9: Results
pdf.add_page()
pdf.chapter_title('8. Ergebnisse & Metriken')
pdf.section_title('8.1 Performance')
pdf.bullet_list([
    'Intent Classification Accuracy: >95%',
    'Durchschnittliche Antwortzeit: 3-5 Sekunden',
    'RAG Retrieval Precision: ~85%',
    'Prophet MAPE (Mean Absolute Percentage Error): ~12%'
])
pdf.ln(3)
pdf.section_title('8.2 Use Cases')
pdf.bullet_list([
    '"Was waren die Top-Kategorien letzten Monat?" -> Sales Agent',
    '"Was sagen Kunden ueber die Lieferung?" -> Sentiment Agent + RAG',
    '"Wie entwickelt sich der Umsatz?" -> Forecast Agent + Prophet'
])
pdf.ln(3)
pdf.section_title('8.3 Live Demo')
pdf.body_text('Die Anwendung ist deployed auf AWS EC2 und weltweit erreichbar:')
pdf.ln(2)
pdf.set_font('DejaVu', 'B', 12)
pdf.set_text_color(0, 100, 180)
pdf.cell(0, 8, 'http://3.121.239.209:8501', 0, 1, 'L')

# Page 10: Conclusion
pdf.add_page()
pdf.chapter_title('9. Zusammenfassung & Learnings')
pdf.section_title('9.1 Key Takeaways')
pdf.bullet_list([
    'Multi-Agent-Systeme ermoeglichen spezialisierte, modulare KI-Loesungen',
    'RAG verbessert LLM-Antworten durch domaenenspezifisches Wissen',
    'Hybrid Search kombiniert semantisches Verstaendnis mit exakter Suche',
    'Prophet liefert interpretierbare, robuste ML-Forecasts',
    'LangGraph vereinfacht komplexe Agent-Workflows erheblich'
])
pdf.ln(5)
pdf.section_title('9.2 Moegliche Erweiterungen')
pdf.bullet_list([
    'Fine-Tuning des Embedding-Modells auf E-Commerce-Domain',
    'Multi-Turn Conversation Memory',
    'A/B Testing verschiedener LLM-Provider',
    'Real-Time Streaming von Datenquellen'
])
pdf.ln(10)
pdf.section_title('Kontakt & Repository')
pdf.body_text('GitHub: github.com/SimonOnChain/Marketing-Intelligence-Agent')

# Save PDF
output_path = Path("C:/Users/Simon/Project_Wifi/presentation")
output_path.mkdir(exist_ok=True)
pdf.output(str(output_path / "Marketing_Intelligence_Agent_Report.pdf"))
print(f"PDF saved to: {output_path / 'Marketing_Intelligence_Agent_Report.pdf'}")
