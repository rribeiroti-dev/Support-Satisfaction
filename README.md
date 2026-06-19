# 🛠️ IT Support Satisfaction & AI Tracker

Uma aplicação Fullstack desenvolvida em Python para avaliar o atendimento de analistas de suporte de T.I. O sistema utiliza **Streamlit** para o Frontend/Dashboard e um modelo de Machine Learning nativo em **TensorFlow** para analisar o sentimento do feedback em tempo real, gerando insights e correções para atendimentos futuros.

## 🚀 Tecnologias Utilizadas
* **Frontend/Backend:** Streamlit
* **Inteligência Artificial:** TensorFlow (Keras)
* **Banco de Dados:** SQLite (nativo do Python)
* **Manipulação de Dados:** Pandas & NumPy

## ⚙️ Instalação Local

1. Clone este repositório:
   ```bash
   git clone [https://github.com/rribeiroti-dev/Support-Satisfaction.git](https://github.com/rribeiroti-dev/Support-Satisfaction.git)
   cd Support-Satisfaction

   Crie um ambiente virtual (recomendado):

Bash
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
Instale as dependências:

Bash
pip install -r requirements.txt
Execute a aplicação:

Bash
streamlit run app.py
☁️ Deploy no Render
Este projeto está configurado para deploy imediato no Render.com.

Suba o código para o seu GitHub.

Acesse o Render e crie um novo Web Service.

Conecte o repositório do GitHub criado.

Preencha as configurações do Render da seguinte forma:

Language: Python 3

Build Command: pip install -r requirements.txt

Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Clique em Create Web Service.

Nota: O Render Free tier pode demorar cerca de 1 a 2 minutos para "acordar" caso não receba tráfego por algum tempo.


---

### Dicas de Engenheiro Sênior para os próximos passos (Evolução do Produto):

1. **Persistência de Dados no Render:** Como o SQLite salva os dados em um arquivo local, a cada deploy no Render, o banco será apagado (pois o Render usa containers efêmeros). Para a versão V2 (Produção Real), substitua o SQLite por um banco na nuvem (como o PostgreSQL gratuito oferecido pela Supabase, Render ou ElephantSQL) utilizando a biblioteca `psycopg2` ou `SQLAlchemy`.
2. **Evolução do Modelo:** O modelo do TensorFlow inserido no código treina "on-the-fly"
