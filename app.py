import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import os

# Desativa os logs chatos de warning do C++ do TensorFlow no terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# 1. CONFIGURAÇÃO DO BANCO DE DADOS (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('avaliacoes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            analista TEXT,
            avaliacao_texto TEXT,
            sentimento TEXT,
            confianca REAL,
            correcao_sugerida TEXT
        )
    ''')
    conn.commit()
    return conn

# ==========================================
# 2. MOTOR DE IA (TENSORFLOW)
# ==========================================
@st.cache_resource
def load_or_train_model():
    """
    Treina um modelo simples de NLP do TensorFlow.
    Dataset expandido para lidar melhor com variações de palavras no MVP.
    """
    
    # Base de conhecimento expandida
    X_train = [
        # Exemplos Positivos (1)
        "atendimento excelente e rápido",
        "analista muito prestativo e educado",
        "resolveu meu problema com facilidade",
        "muito eficiente, super rápido e bem educado",
        "ótimo suporte, direto ao ponto",
        "perfeito, sem reclamações",
        "bom atendimento",
        
        # Exemplos Negativos (0)
        "demora excessiva no suporte",
        "péssimo atendimento",
        "analista rude e lento",
        "demorou muito para iniciar atendimento e resolver",
        "não resolveu o problema",
        "muito ruim e ineficiente",
        "falta de respeito e lentidão",
        "suporte péssimo, demorou demais"
    ]
    
    y_train = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    # Aumentamos os tokens para abranger mais palavras
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=150, output_sequence_length=8)
    X_train_tensor = tf.constant(X_train, dtype=tf.string)
    vectorizer.adapt(X_train_tensor)

    # Construção do modelo sequencial
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(input_dim=150, output_dim=16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Aumentamos as épocas (epochs) para ele aprender com mais firmeza
    model.fit(X_train_tensor, y_train, epochs=40, verbose=0)
    
    return model

def analisar_feedback(texto, model):
    """Analisa o texto e retorna o sentimento e a sugestão de correção."""
    texto_tensor = tf.constant([texto], dtype=tf.string)
    predicao = model.predict(texto_tensor, verbose=0)[0][0]
    
    confianca = float(predicao if predicao > 0.5 else 1 - predicao)
    texto_lower = texto.lower()
    
    if predicao >= 0.5:
        sentimento = "Positivo"
        correcao = "Manter o padrão de excelência. Elogiar o analista."
    else:
        sentimento = "Negativo"
        # Melhoria nas regras para capturar conjugações diferentes
        if any(palavra in texto_lower for palavra in ["demora", "demorou", "lento", "lentidão"]):
            correcao = "Melhorar o SLA. Treinamento de gestão de tempo e priorização."
        elif any(palavra in texto_lower for palavra in ["rude", "péssimo", "respeito"]):
            correcao = "Feedback urgente. Treinamento de Soft Skills e empatia."
        else:
            correcao = "Analisar ticket detalhadamente para identificar o gargalo técnico."
            
    return sentimento, confianca, correcao

# ==========================================
# 3. INTERFACE DE USUÁRIO (STREAMLIT)
# ==========================================
st.set_page_config(page_title="T.I Support Tracker - AI", layout="wide")

# Inicializa DB e IA
conn = init_db()
model = load_or_train_model()

st.title("🛠️ IT Support Satisfaction & AI Tracker")
st.markdown("Plataforma de avaliação de chamados com análise de sentimento via TensorFlow.")

tab1, tab2 = st.tabs(["📝 Nova Avaliação", "📊 Dashboard de Qualidade"])

# --- TAB 1: Formulário de Avaliação ---
with tab1:
    st.subheader("Registrar Feedback do Usuário")
    with st.form("avaliacao_form"):
        analista = st.selectbox("Analista de Suporte", ["João Silva", "Maria Souza", "Carlos Costa", "Ana Oliveira"])
        feedback_texto = st.text_area("Descreva como foi o atendimento:")
        
        submit = st.form_submit_button("Analisar e Salvar")
        
        if submit and feedback_texto:
            with st.spinner("A rede neural está analisando o feedback..."):
                sentimento, confianca, correcao = analisar_feedback(feedback_texto, model)
                data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Salvar no Banco
                c = conn.cursor()
                c.execute(
                    "INSERT INTO feedbacks (data_hora, analista, avaliacao_texto, sentimento, confianca, correcao_sugerida) VALUES (?, ?, ?, ?, ?, ?)",
                    (data_atual, analista, feedback_texto, sentimento, confianca, correcao)
                )
                conn.commit()
                
            st.success("Avaliação registrada com sucesso!")
            
            # Formatação de cores no Streamlit baseada no sentimento
            if sentimento == "Positivo":
                st.info(f"**Análise da IA:** Sentimento {sentimento} ({confianca:.1%} de confiança).")
                st.success(f"**Ação Recomendada:** {correcao}")
            else:
                st.error(f"**Análise da IA:** Sentimento {sentimento} ({confianca:.1%} de confiança).")
                st.warning(f"**Ação Recomendada:** {correcao}")

# --- TAB 2: Dashboard ---
with tab2:
    st.subheader("Histórico e Métricas")
    df = pd.read_sql_query("SELECT * FROM feedbacks ORDER BY id DESC", conn)
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        total_avaliacoes = len(df)
        positivas = len(df[df['sentimento'] == 'Positivo'])
        taxa_satisfacao = (positivas / total_avaliacoes) * 100
        
        col1.metric("Total de Avaliações", total_avaliacoes)
        col2.metric("Taxa de Satisfação", f"{taxa_satisfacao:.1f}%")
        col3.metric("Feedbacks para Correção", total_avaliacoes - positivas)
        
        st.markdown("### Últimos Registros")
        st.dataframe(df[['data_hora', 'analista', 'sentimento', 'correcao_sugerida']], use_container_width=True)
    else:
        st.write("Nenhuma avaliação registrada ainda.")

conn.close()
