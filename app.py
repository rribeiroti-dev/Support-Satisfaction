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

# Inicialização do estado para controle de mensagens após o reset do formulário
if 'msg_sucesso' not in st.session_state:
    st.session_state.msg_sucesso = None
if 'msg_ia' not in st.session_state:
    st.session_state.msg_ia = None
if 'msg_acao' not in st.session_state:
    st.session_state.msg_acao = None
if 'tipo_sentimento' not in st.session_state:
    st.session_state.tipo_sentimento = None

# ==========================================
# 1. CONFIGURAÇÃO DO BANCO DE DADOS (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('avaliacoes.db')
    c = conn.cursor()
    # Atualizado para a versão v2 para conter os novos campos corporativos
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
    """
    Treina um modelo simples de NLP do TensorFlow.
    Dataset expandido para lidar melhor com variações de palavras no MVP.
    """
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

conn = init_db()
model = load_or_train_model()

st.title("🛠️ IT Support Satisfaction & AI Tracker")
st.markdown("Plataforma de avaliação de chamados com análise de sentimento via TensorFlow.")

tab1, tab2 = st.tabs(["📝 Nova Avaliação", "📊 Dashboard de Qualidade"])

# --- TAB 1: Formulário de Avaliação ---
with tab1:
    st.subheader("Registrar Feedback do Usuário")
    
    # Exibe notificações persistidas em sessão caso existam
    if st.session_state.msg_sucesso:
        st.success(st.session_state.msg_sucesso)
        if st.session_state.tipo_sentimento == "Positivo":
            st.info(st.session_state.msg_ia)
        else:
            st.error(st.session_state.msg_ia)
        st.warning(st.session_state.msg_acao)
        
        # Limpa as mensagens após a exibição para que não repitam sozinhas
        st.session_state.msg_sucesso = None
        st.session_state.msg_ia = None
        st.session_state.msg_acao = None
        st.session_state.tipo_sentimento = None

    with st.form("avaliacao_form", clear_on_submit=False):
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
                st.error("Por favor, preencha todos os campos obrigatórios: Empresa, Nome do Contato e a Descrição do Atendimento.")
            else:
                with st.spinner("A rede neural está analisando o feedback..."):
                    sentimento, confianca, correcao = analisar_feedback(feedback_texto, model)
                    data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Salvar no Banco com os novos campos inclusos
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO feedbacks_v2 
                        (data_hora, analista, empresa, nome_contato, email, telefone, avaliacao_texto, sentimento, confianca, correcao_sugerida) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data_atual, analista, empresa, nome_contato, email, telefone, feedback_texto, sentimento, confianca, correcao))
                    conn.close()
                    
                    # Guarda os resultados no session_state para exibi-los após o reload do formulário
                    st.session_state.msg_sucesso = "Avaliação registrada com sucesso e campos limpos para o próximo uso!"
                    st.session_state.msg_ia = f"**Análise da IA:** Sentimento {sentimento} ({confianca:.1%} de confiança)."
                    st.session_state.msg_acao = f"**Ação Recomendada:** {correcao}"
                    st.session_state.tipo_sentimento = sentimento
                    
                    # Força a limpeza visual recarregando a página limpa
                    st.rerun()

# --- TAB 2: Dashboard ---
with tab2:
    st.subheader("Histórico e Métricas Avançadas")
    
    # Busca dados atualizados da nova tabela
    df = pd.read_sql_query("SELECT * FROM feedbacks_v2 ORDER BY id DESC", conn)
    
    if not df.empty:
        total_avaliacoes = len(df)
        positivas = len(df[df['sentimento'] == 'Positivo'])
        negativas = total_avaliacoes - positivas
        
        taxa_satisfacao = (positivas / total_avaliacoes) * 100
        taxa_reprovacao = (negativas / total_avaliacoes) * 100
        
        # Primeira Linha de Métricas (Volumes Gerais)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Avaliações", total_avaliacoes)
        col2.metric("Total Positivas (Elogios)", positivas)
        col3.metric("Total Negativas (Correções)", negativas)
        
        # Segunda Linha de Métricas (Porcentagens Claras da Operação)
        st.markdown("### Distribuição Percentual")
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Índice de Satisfação (Positivos %)", f"{taxa_satisfacao:.1f}%")
        col_p2.metric("Índice de Melhoria (Negativos %)", f"{taxa_reprovacao:.1f}%")
        
        # Exibição Completa das Avaliações solicitadas
        st.markdown("### Registro Detalhado de Avaliações")
        
        # Organização das colunas na tabela final para visualização completa
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
