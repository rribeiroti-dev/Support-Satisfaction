import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import os

# Desativa logs de C++ do TensorFlow no terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# POP-UP NATIVO (Requer Streamlit >= 1.36.0)
# ==========================================
@st.dialog("📊 Resultado da Análise IA")
def exibir_popup(sentimento, confianca, correcao):
    st.write("Sua avaliação foi salva no banco e processada com sucesso!")
    
    if sentimento == "Positivo":
        st.info(f"**Sentimento:** {sentimento} ({confianca:.1%} de confiança)")
    else:
        st.error(f"**Sentimento:** {sentimento} ({confianca:.1%} de confiança)")
        
    st.warning(f"**Ação Recomendada:** {correcao}")
    
    if st.button("Fechar Aba"):
        # Ao clicar, recarrega a página fechando o popup
        st.rerun()

# ==========================================
# 1. CONFIGURAÇÃO DO BANCO DE DADOS
# ==========================================
def init_db():
    conn = sqlite3.connect('avaliacoes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            analista TEXT,
            empresa TEXT,
            nome_contato TEXT,
            email TEXT,
            telefone TEXT,
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
    X_train = [
        # Positivos
        "atendimento excelente e rápido",
        "analista muito prestativo e educado",
        "resolveu meu problema com facilidade",
        "muito eficiente, super rápido e bem educado",
        "ótimo suporte, direto ao ponto",
        "perfeito, sem reclamações",
        "bom atendimento",
        
        # Negativos
        "demora excessiva no suporte",
        "péssimo atendimento",
        "analista rude e lento",
        "demorou muito para iniciar atendimento e resolver",
        "não resolveu o problema",
        "muito ruim e ineficiente",
        "falta de respeito e lentidão",
        "suporte péssimo, demorou demais",
        "muito mal educada",  # <--- Adicionado após o seu teste!
        "mal educado",
        "horrível"
    ]
    y_train = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    vectorizer = tf.keras.layers.TextVectorization(max_tokens=150, output_sequence_length=8)
    X_train_tensor = tf.constant(X_train, dtype=tf.string)
    vectorizer.adapt(X_train_tensor)

    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(input_dim=150, output_dim=16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_tensor, y_train, epochs=40, verbose=0)
    return model

def analisar_feedback(texto, model):
    texto_tensor = tf.constant([texto], dtype=tf.string)
    predicao = model.predict(texto_tensor, verbose=0)[0][0]
    
    confianca = float(predicao if predicao > 0.5 else 1 - predicao)
    texto_lower = texto.lower()
    
    if predicao >= 0.5:
        sentimento = "Positivo"
        correcao = "Manter o padrão de excelência. Elogiar o analista."
    else:
        sentimento = "Negativo"
        if any(palavra in texto_lower for palavra in ["demora", "demorou", "lento", "lentidão"]):
            correcao = "Melhorar o SLA. Treinamento de gestão de tempo e priorização."
        elif any(palavra in texto_lower for palavra in ["rude", "péssimo", "respeito", "mal educada", "mal educado", "horrível"]):
            correcao = "Feedback urgente. Treinamento de Soft Skills e empatia."
        else:
            correcao = "Analisar ticket detalhadamente para identificar o gargalo técnico."
            
    return sentimento, confianca, correcao

# ==========================================
# 3. INTERFACE DE USUÁRIO
# ==========================================
st.set_page_config(page_title="T.I Support Tracker - AI", layout="wide")

conn = init_db()
model = load_or_train_model()

st.title("🛠️ IT Support Satisfaction & AI Tracker")
st.markdown("Plataforma de avaliação de chamados com análise de sentimento via TensorFlow.")

tab1, tab2 = st.tabs(["📝 Nova Avaliação", "📊 Dashboard de Qualidade"])

# --- TAB 1: Formulário ---
with tab1:
    st.subheader("Registrar Feedback do Usuário")
    
    # clear_on_submit=True garante que o fundo será limpo instantaneamente
    with st.form("avaliacao_form", clear_on_submit=True):
        st.markdown("*Campos marcados com ** são obrigatórios.*")
        
        analista = st.selectbox("Analista de Suporte *", ["João Silva", "Maria Souza", "Carlos Costa", "Ana Oliveira"])
        
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            empresa = st.text_input("Empresa *")
            email = st.text_input("E-mail (Opcional)")
        with col_form2:
            nome_contato = st.text_input("Nome do Contato *")
            telefone = st.text_input("Telefone (Opcional)")
            
        feedback_texto = st.text_area("Descreva como foi o atendimento *")
        
        submit = st.form_submit_button("Analisar e Salvar")
        
        if submit:
            if not empresa.strip() or not nome_contato.strip() or not feedback_texto.strip():
                st.error("Por favor, preencha todos os campos obrigatórios (Empresa, Contato, Descrição).")
            else:
                with st.spinner("A rede neural está analisando o feedback..."):
                    sentimento, confianca, correcao = analisar_feedback(feedback_texto, model)
                    data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO feedbacks_v2 
                        (data_hora, analista, empresa, nome_contato, email, telefone, avaliacao_texto, sentimento, confianca, correcao_sugerida) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data_atual, analista, empresa, nome_contato, email, telefone, feedback_texto, sentimento, confianca, correcao))
                    
                    # CORREÇÃO APLICADA AQUI: O commit salva definitivamente os dados!
                    conn.commit()
                    
                    # Chama o nosso novo Pop-Up
                    exibir_popup(sentimento, confianca, correcao)

# --- TAB 2: Dashboard ---
with tab2:
    st.subheader("Histórico e Métricas Avançadas")
    
    df = pd.read_sql_query("SELECT * FROM feedbacks_v2 ORDER BY id DESC", conn)
    
    if not df.empty:
        total_avaliacoes = len(df)
        positivas = len(df[df['sentimento'] == 'Positivo'])
        negativas = total_avaliacoes - positivas
        
        taxa_satisfacao = (positivas / total_avaliacoes) * 100
        taxa_reprovacao = (negativas / total_avaliacoes) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Avaliações", total_avaliacoes)
        col2.metric("Total Positivas (Elogios)", positivas)
        col3.metric("Total Negativas (Correções)", negativas)
        
        st.markdown("### Distribuição Percentual")
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Índice de Satisfação (Positivos %)", f"{taxa_satisfacao:.1f}%")
        col_p2.metric("Índice de Melhoria (Negativos %)", f"{taxa_reprovacao:.1f}%")
        
        st.markdown("### Registro Detalhado de Avaliações")
        df_exibicao = df[[
            'data_hora', 'analista', 'empresa', 'nome_contato', 
            'email', 'telefone', 'avaliacao_texto', 'sentimento', 'correcao_sugerida'
        ]].rename(columns={
            'data_hora': 'Data/Hora',
            'analista': 'Analista',
            'empresa': 'Empresa Cliente',
            'nome_contato': 'Quem Avaliou',
            'email': 'E-mail',
            'telefone': 'Telefone',
            'avaliacao_texto': 'Depoimento/Avaliação Escrita',
            'sentimento': 'Sentimento IA',
            'correcao_sugerida': 'Ação Corretiva Sugerida'
        })
        
        st.dataframe(df_exibicao, use_container_width=True)
    else:
        st.write("Nenhuma avaliação registrada ainda nesta versão.")

conn.close()
