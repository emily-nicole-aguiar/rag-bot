# Requisitos: Instale as dependências necessárias localmente
# pip install chromadb scikit-learn pyodbc google-generativeai pandas python-dotenv

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pyodbc
import google.generativeai as genai
import pandas as pd
import os
import pickle  # Para persistir o vectorizer

# Configurações
# Para Gemini API: Crie um arquivo .env na raiz do projeto com: GEMINI_API_KEY=sua-chave-aqui
# Ou defina como variável de ambiente: export GEMINI_API_KEY='sua-chave-aqui'
from dotenv import load_dotenv
load_dotenv()  # Carrega .env se existir

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Defina GEMINI_API_KEY no .env ou como variável de ambiente!")
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-2.5-flash')  # Modelo gratuito

# Conexão SQL Server local (Windows Authentication)
SERVER = 'EMILYNICOLE'
DATABASE = 'datasets'
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'

# Inicializar ChromaDB local (persistente em ./chroma_db)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="schema_collection")

# Descrições das tabelas e colunas (baseado no schema fornecido; adicione descrições reais se disponíveis)
schema_documents = [
    {
        "id": "table_1",
        "content": """
        Tabela: TabelaOriginal. Descrição: Contém dados de modelos por país, data de entrega, status e operadora. 
        Colunas: 
        - Modelo (modelo do item), 
        - País (país de origem; valores em inglês, ex: 'France' para França, 'Brazil' para Brasil, 'USA' para Estados Unidos),
        - Data_Entrega (data de entrega), 
        - Status (status do pedido, ex: Done, Completed, Analysis, etc), 
        - Operadora (operadora responsável).
        """
    },
    {
        "id": "table_2",
        "content": "Tabela: nutrition_cf. Descrição: Informações nutricionais e pegada de carbono de alimentos. Colunas: Food (nome do alimento), Associativity (associatividade), Region (região), Type (tipo), Category (categoria), Allergy (alergênicos), Serving (porção), Total_Weight_gms (peso total em gramas), Energy_kcal (energia em kcal), Proteins (proteínas), Carbohydrates (carboidratos), Fats (gorduras), Fiber (fibra), Carbon_Footprint_kg_CO2e (pegada de carbono em kg CO2e), Ingredients (ingredientes)."
    },
    {
        "id": "table_3",
        "content": """
        Tabela: GraduateEmployment. Descrição: Dados anuais de emprego de graduados universitários em Singapura (fonte: dados oficiais).
        Colunas detalhadas:
        - year: Ano de graduação (ex: 2023).
        - university: Nome da universidade (ex: National University of Singapore).
        - school: Faculdade ou escola dentro da universidade (ex: School of Computing).
        - degree: Tipo de grau (ex: Bachelor's in Computer Science).
        - employment_rate_overall: Taxa de emprego geral em % (média de todos os setores).
        - employment_rate_ft_perm: Taxa de emprego em tempo integral e permanente em %.
        - basic_monthly_mean: Média salarial básica mensal em SGD.
        - basic_monthly_median: Mediana salarial básica mensal em SGD.
        - gross_monthly_mean: Média salarial bruta mensal em SGD (inclui bônus).
        - gross_monthly_median: Mediana salarial bruta mensal em SGD.
        - gross_mthly_25_percentile: Percentil 25 do salário bruto mensal em SGD.
        - gross_mthly_75_percentile: Percentil 75 do salário bruto mensal em SGD.
        """
    },
    {
        "id": "table_4",
        "content": "Tabela: game_data_all. Descrição: Dados de jogos, incluindo lançamento e gêneros. Colunas: game (nome do jogo), release (data de lançamento), primary_genre (gênero principal), store_genres (gêneros da loja), publisher (editora), developer (desenvolvedor)."
    }
]

# Arquivos para persistir TF-IDF (local)
TFIDF_MODEL_PATH = "./tfidf_model.pkl"

# Função para popular o ChromaDB com embeddings TF-IDF locais (execute uma vez)
def populate_chroma_and_tfidf():
    documents = [doc["content"] for doc in schema_documents]
    ids = [doc["id"] for doc in schema_documents]
    
    # Preparar TF-IDF localmente (sem downloads)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)  # Sem stop_words para português; ajuste se necessário
    tfidf_matrix = vectorizer.fit_transform(documents)
    embeddings = tfidf_matrix.toarray().tolist()  # Converter para dense list para ChromaDB
    
    # Persistir vectorizer para uso futuro
    with open(TFIDF_MODEL_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Adicionar ao ChromaDB com embeddings locais (sem download de modelo)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    
    print("ChromaDB populado com embeddings TF-IDF locais.")

# Chame populate_chroma_and_tfidf() na primeira execução
# populate_chroma_and_tfidf()

# Função para carregar TF-IDF persistido
def load_tfidf():
    with open(TFIDF_MODEL_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Função para consultar RAG usando ChromaDB com TF-IDF (100% local)
def query_rag(user_question, n_results=2):
    try:
        vectorizer = load_tfidf()
    except FileNotFoundError:
        print("Execute populate_chroma_and_tfidf() primeiro!")
        return ""
    
    query_vec = vectorizer.transform([user_question]).toarray().tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=n_results
    )
    relevant_schema = "\n".join(results['documents'][0])
    return relevant_schema

# Função para gerar SQL via LLM (Gemini)
def generate_sql(user_question, relevant_schema):
    prompt = f"""
    Com base no schema das tabelas abaixo, gere uma consulta SQL válida para o banco de dados [datasets] no SQL Server.
    Use apenas as tabelas e colunas relevantes. A consulta deve ser otimizada e segura (sem injeção).

    IMPORTANTE: Considere variações linguísticas. Ex: Se a pergunta menciona 'França', use 'France' no SQL (pois os dados estão em inglês). 
    Mapeie termos comuns: França=France, Brasil=Brazil, EUA=USA/United States.
    
    Schema relevante:
    {relevant_schema}
    
    Pergunta do usuário: {user_question}
    
    Responda APENAS com a consulta SQL, nada mais. Exemplo: SELECT * FROM [dbo].[TabelaOriginal] WHERE [País] = 'Brasil';

    Nota: RETORNE SOMENTE A CONSULTA SQL, NADA MAIS. NÃO UTILIZE NENHUM TIPO DE MARCADOR OU FORMATAÇÃO.
    """
    
    response = MODEL.generate_content(prompt)
    sql_query = response.text.strip()
    return sql_query

# Função para executar SQL no SQL Server local
def execute_sql(sql_query):
    try:
        conn = pyodbc.connect(conn_str)
        df = pd.read_sql(sql_query, conn)
        conn.close()
        return df.to_json(orient='records', indent=2)  # Retorna como JSON para facilitar
    except Exception as e:
        return f"Erro na execução SQL: {str(e)}"

# Função para tratar resposta via LLM
def treat_response(sql_results, user_question):
    prompt = f"""
    Aqui estão os resultados da consulta SQL:
    {sql_results}
    
    Pergunta original: {user_question}
    
    Forneça uma resposta clara, concisa e amigável em português, explicando os dados de forma tratada. Se não houver dados, diga isso educadamente.
    """
    
    response = MODEL.generate_content(prompt)
    return response.text.strip()

# Pipeline principal
def rag_pipeline(user_question):
    # 1. Consultar RAG para schema relevante (TF-IDF via ChromaDB local)
    relevant_schema = query_rag(user_question)
    
    # 2. Gerar SQL via LLM
    sql_query = generate_sql(user_question, relevant_schema)
    print(f"SQL gerado: {sql_query}")  # Para debug
    
    # 3. Executar SQL
    results = execute_sql(sql_query)
    
    # 4. Tratar resposta via LLM
    final_answer = treat_response(results, user_question)
    
    return final_answer

# Exemplo de uso
if __name__ == "__main__":
    # Descomente para popular (rode uma vez)
    populate_chroma_and_tfidf()
    
    question = "Qual o país que está com mais entrega em análise"
    answer = rag_pipeline(question)
    print(f"Resposta: {answer}")