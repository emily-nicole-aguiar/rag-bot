# ===============================================
# SCRIPT OFFLINE OTIMIZADO: SQL SERVER + RAG + Chroma + Gemini
# ===============================================

import pyodbc
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import spacy
from numpy import array
from google import genai
import requests

# ------------------------------
# 1️⃣ Conexão SQL Server
# ------------------------------
conn_str = (
    "Driver={SQL Server};"
    "Server=EMILYNICOLE;"
    "Database=datasets;"
    "Trusted_Connection=yes;"
)
conn = pyodbc.connect(conn_str)

tables = ["TabelaOriginal", "nutrition_cf", "GraduateEmployment", "game_data_all"]
samples = {}

for table in tables:
    query = f"SELECT TOP 5 * FROM [dbo].[{table}]"
    samples[table] = pd.read_sql(query, conn)

conn.close()

# ------------------------------
# 2️⃣ Descrições manuais das colunas
# ------------------------------
descricao_colunas_todas = {
    # (Mesma definição que você já tem)
    "TabelaOriginal": {
        "Modelo": "Código do modelo do pedido, por exemplo 'XYZ123'.",
        "País": "País de origem do pedido.",
        "Data_Entrega": "Data prevista de entrega do pedido.",
        "Status": "Status atual do pedido (Done, Completed, Analysis, etc).",
        "Operadora": "Operadora responsável pela entrega."
    },
    "nutrition_cf": {
        "Food": "Nome do alimento.",
        "Associativity": "Informação sobre combinação alimentar.",
        "Region": "Região de origem do alimento.",
        "Type": "Tipo do alimento (Ex: Fruta, Legume, Laticínio).",
        "Category": "Categoria nutricional.",
        "Allergy": "Indicação de alergênicos.",
        "Serving": "Porção recomendada.",
        "Total_Weight_gms": "Peso total em gramas.",
        "Energy_kcal": "Calorias.",
        "Proteins": "Proteínas em gramas.",
        "Carbohydrates": "Carboidratos em gramas.",
        "Fats": "Gorduras em gramas.",
        "Fiber": "Fibras em gramas.",
        "Carbon_Footprint_kg_CO2e": "Pegada de carbono em kg CO2 equivalente.",
        "Ingredients": "Ingredientes do alimento."
    },
    "GraduateEmployment": {
        "year": "Ano da graduação.",
        "university": "Nome da universidade.",
        "school": "Nome da escola/faculdade.",
        "degree": "Tipo de diploma obtido.",
        "employment_rate_overall": "Taxa geral de emprego dos graduados.",
        "employment_rate_ft_perm": "Taxa de emprego em tempo integral/permanente.",
        "basic_monthly_mean": "Média salarial mensal básica.",
        "basic_monthly_median": "Mediana salarial mensal básica.",
        "gross_monthly_mean": "Média salarial mensal bruta.",
        "gross_monthly_median": "Mediana salarial mensal bruta.",
        "gross_mthly_25_percentile": "25º percentil salarial bruto.",
        "gross_mthly_75_percentile": "75º percentil salarial bruto."
    },
    "game_data_all": {
        "game": "Nome do jogo.",
        "release": "Data de lançamento.",
        "primary_genre": "Gênero principal do jogo.",
        "store_genres": "Gêneros disponíveis nas lojas.",
        "publisher": "Editora do jogo.",
        "developer": "Desenvolvedora do jogo."
    }
}

# ------------------------------
# 3️⃣ Inicializar Spacy + função de embeddings
# ------------------------------
nlp = spacy.load("en_core_web_md")  # 100% offline

def spacy_embedding(text):
    return array(nlp(text).vector).tolist()

embedding_fn = embedding_functions.ExternalEmbeddingFunction(
    embed_fn=spacy_embedding,
    embedding_dim=nlp.vocab.vectors_length
)

# ------------------------------
# 4️⃣ Inicializar Chroma
# ------------------------------
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="colunas_db",
    embedding_function=embedding_fn
)

# Otimização: adiciona **uma entrada por tabela**, concatenando descrições
for tabela, col_dict in descricao_colunas_todas.items():
    all_desc = " ".join([f"{col}: {desc}" for col, desc in col_dict.items()])
    doc_id = f"{tabela}"
    collection.add(
        documents=[all_desc],
        ids=[doc_id],
        metadatas=[{"tabela": tabela}]
    )

# ------------------------------
# 5️⃣ Seleção da tabela/coluna via Chroma
# ------------------------------
def selecionar_tabela_e_contexto(pergunta):
    resultado = collection.query(
        query_texts=[pergunta],
        n_results=1
    )
    tabela = resultado['metadatas'][0][0]['tabela']
    contexto = resultado['documents'][0][0]
    return tabela, contexto

# ------------------------------
# 6️⃣ Configuração da LLM Gemini
# ------------------------------
GEMINI_API_KEY = ""
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------------------
# 7️⃣ Função final de resposta
# ------------------------------
def responder(pergunta):
    tabela, contexto = selecionar_tabela_e_contexto(pergunta)
    prompt = f"Pergunta: {pergunta}\nTabela: {tabela}\nContexto: {contexto}"

    # Mistral comentada
    # if modelo == "mistral":
    #     ...

    # Apenas Gemini
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Erro ao chamar a API Gemini: {e}"

# ------------------------------
# 8️⃣ Teste
# ------------------------------
pergunta = "Quais países tiveram status OK nos pedidos?"
resposta = responder(pergunta)
print("Pergunta:", pergunta)
print("Resposta:", resposta)
