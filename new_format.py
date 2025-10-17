# Requisitos:
# pip install chromadb sentence-transformers pyodbc google-generativeai pandas python-dotenv sqlalchemy

import chromadb
from chromadb.utils import embedding_functions
import pyodbc
import google.generativeai as genai
import pandas as pd
import os
import datetime
from dotenv import load_dotenv
import time  # Para profiling
from sqlalchemy import create_engine

# ===========================
# 🔹 CONFIGURAÇÕES INICIAIS
# ===========================

load_dotenv()  # Carrega .env se existir

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Defina GEMINI_API_KEY no .env ou como variável de ambiente!")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(
    "gemini-2.5-flash",  # Mais rápido; se preferir original, mude para "gemini-2.5-flash"
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=2048,  # Aumentado para evitar finish_reason=2 (MAX_TOKENS)
        temperature=0.1  # Menos criatividade, mais velocidade
    )
)

# SQL Server (Windows Authentication) para SQLAlchemy
SERVER = "EMILYNICOLE"
DATABASE = "datasets"
# String de conexão ajustada para SQLAlchemy (mssql+pyodbc)
engine_str = f"mssql+pyodbc://@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(engine_str)

# Inicializa o ChromaDB persistente local
client = chromadb.PersistentClient(path="./chroma_db")

# Embedding function local (modelo já baixado)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="modelos/all-MiniLM-L6-v2"  # Caminho local do modelo
)

# Coleções do ChromaDB
collection = client.get_or_create_collection(
    name="schema_collection_v2",
    embedding_function=embedding_fn
)
history_collection = client.get_or_create_collection(
    name="history_collection_v2",
    embedding_function=embedding_fn
)

# ===========================
# 🔹 SCHEMAS DAS TABELAS
# ===========================

schema_documents = [
    {
        "id": "table_1",
        "content": """
        Tabela: TabelaOriginal. Descrição: Contém dados de modelos por país, data de entrega, status e operadora. 
        Colunas: 
        - Modelo (modelo do item), 
        - País (país de origem; valores em inglês, ex: 'France', 'Brazil', 'USA'),
        - Data_Entrega (data de entrega), 
        - Status (status do pedido, ex: Done, Completed, Analysis), 
        - Operadora (operadora responsável).
        """
    },
    {
        "id": "table_2",
        "content": "Tabela: nutrition_cf. Descrição: Informações nutricionais e pegada de carbono de alimentos. Colunas: Food, Region, Type, Category, Allergy, Serving, Weight_g, Energy_kcal, Proteins, Carbohydrates, Fats, Fiber, Carbon_Footprint_kg_CO2e, Ingredients."
    },
    {
        "id": "table_3",
        "content": """
        Tabela: GraduateEmployment. Dados anuais de emprego de graduados em Singapura.
        Colunas: year, university, school, degree, employment_rate_overall, employment_rate_ft_perm,
        basic_monthly_mean, basic_monthly_median, gross_monthly_mean, gross_monthly_median,
        gross_mthly_25_percentile, gross_mthly_75_percentile.
        """
    },
    {
        "id": "table_4",
        "content": "Tabela: game_data_all. Descrição: Dados de jogos, incluindo nome, data de lançamento, gênero, editora e desenvolvedor."
    }
]

# ===========================
# 🔹 POPULAR SCHEMA (execute uma vez)
# ===========================

def populate_chroma():
    existing = collection.count()
    if existing > 0:
        print(f"⚠️ ChromaDB já possui {existing} documentos. Pular população.")
        return

    documents = [doc["content"] for doc in schema_documents]
    ids = [doc["id"] for doc in schema_documents]

    collection.add(documents=documents, ids=ids)
    print("✅ Schemas adicionados ao ChromaDB com embeddings locais (SentenceTransformer).")

# ===========================
# 🔹 CONSULTA RAG (Schema + Histórico)
# ===========================

def query_rag(user_question, n_results=2):
    schema_results = collection.query(query_texts=[user_question], n_results=n_results)
    schema_relevant = "\n".join(schema_results["documents"][0]) if schema_results["documents"] else ""
    
    # Truncar contexto mais agressivamente para evitar prompts longos
    if len(schema_relevant) > 600:
        schema_relevant = schema_relevant[:600] + "\n... (truncado)"

    history_results = history_collection.query(query_texts=[user_question], n_results=1)  # Reduzido para 1 para menos contexto
    history_relevant = "\n".join(history_results["documents"][0]) if history_results["documents"] else ""
    
    if len(history_relevant) > 300:
        history_relevant = history_relevant[:300] + "\n... (truncado)"

    full_context = f"Schema relevante:\n{schema_relevant}\n\nHistórico recente:\n{history_relevant}"
    return full_context

# ===========================
# 🔹 GERAR SQL VIA GEMINI
# ===========================

def generate_sql(user_question, relevant_context):
    # Detecta tipo de query para ajustar
    is_report = "relatório" in user_question.lower() or "report" in user_question.lower() or "listar" in user_question.lower()
    report_hint = "Para relatórios ou listas, selecione todas as colunas relevantes (ex: SELECT * ou liste Modelo, Data_Entrega, Status, Operadora) e use GROUP BY se necessário para agregações como contagens por país/mês." if is_report else ""
    
    # Prompt mais curto
    prompt = f"""
    Gere SQL válido para [datasets] (SQL Server). Use tabelas/colunas relevantes do contexto.
    {report_hint}
    Mapeie países: França=France, Brasil=Brazil, EUA=USA.

    Contexto: {relevant_context}

    Pergunta: {user_question}

    Responda APENAS com SQL. Ex: SELECT Modelo, Data_Entrega, Status, Operadora, COUNT(*) as Total FROM [dbo].[TabelaOriginal] GROUP BY Modelo, Data_Entrega, Status, Operadora;

    SOMENTE SQL, sem formatação.
    """

    try:
        response = MODEL.generate_content(prompt)
        
        # Verificar se há partes válidas
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            sql_query = response.candidates[0].content.parts[0].text.strip()
        else:
            finish_reason = response.candidates[0].finish_reason if response and response.candidates else "None"
            print(f"⚠️ Erro na geração SQL: finish_reason={finish_reason}. Usando fallback.")
            # Fallback melhorado: Detecta palavras-chave na pergunta
            if "país" in user_question.lower() and "mês" in user_question.lower():
                sql_query = "SELECT Modelo, País, YEAR(Data_Entrega) as Ano, MONTH(Data_Entrega) as Mês, COUNT(*) as Total_Entregas FROM [dbo].[TabelaOriginal] GROUP BY Modelo, País, YEAR(Data_Entrega), MONTH(Data_Entrega) ORDER BY Total_Entregas DESC;"
            elif "Brasil" in user_question and "2025" in user_question:
                sql_query = "SELECT Modelo, Data_Entrega, Status, Operadora FROM [dbo].[TabelaOriginal] WHERE [País] = 'Brazil' AND YEAR(Data_Entrega) = 2025;"
            else:
                sql_query = "SELECT TOP 100 * FROM [dbo].[TabelaOriginal];"  # Limita para evitar dumps grandes
        return sql_query
    except Exception as e:
        print(f"❌ Erro na API Gemini: {str(e)}")
        return "SELECT TOP 100 * FROM [dbo].[TabelaOriginal];"

# ===========================
# 🔹 EXECUTAR SQL LOCALMENTE
# ===========================

def execute_sql(sql_query):
    try:
        # Usa SQLAlchemy para evitar warning do pandas
        df = pd.read_sql(sql_query, engine)
        # Limita JSON se df grande (evita prompts longos)
        if len(df) > 50:
            df_limited = df.head(50)  # Primeiros 50 rows
            json_results = df_limited.to_json(orient="records", indent=2) + f"\n... (total de {len(df)} registros, mostrando primeiros 50)"
        else:
            json_results = df.to_json(orient="records", indent=2)
        return json_results
    except Exception as e:
        return f"Erro na execução SQL: {str(e)}"

# ===========================
# 🔹 TRATAR RESPOSTA COM GEMINI
# ===========================

def treat_response(sql_results, user_question):
    # Trunca resultados para evitar MAX_TOKENS no prompt
    if len(sql_results) > 1200:
        sql_results_trunc = sql_results[:1200] + "\n... (dados truncados para processamento; total aproximado: " + str(len(sql_results)) + " chars)"
    else:
        sql_results_trunc = sql_results
    
    prompt = f"""
    Resultados SQL (filtrados pela pergunta):
    {sql_results_trunc}

    Pergunta: {user_question}

    Explique em PT-BR, claro e breve, como relatório.
    - Liste/resuma principais (ex: top modelos por país/mês).
    - Se muitos dados, destaque totais/agregações.
    - Sem dados: "Não encontrados dados."
    - Assuma resultados completos; não valide filtros.
    """

    try:
        response = MODEL.generate_content(prompt)
        
        # Verificação para .text
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            final_text = response.candidates[0].content.parts[0].text.strip()
        else:
            finish_reason = response.candidates[0].finish_reason if response and response.candidates else "None"
            print(f"⚠️ Erro na tratamento de resposta: finish_reason={finish_reason}. Usando resumo simples.")
            # Fallback melhorado
            if "Erro" in sql_results:
                final_text = f"Não foi possível processar: {sql_results}"
            elif "[ " in sql_results:  # JSON array
                # Extrai resumo simples
                lines = sql_results.split('\n')
                summary = f"Resumo para '{user_question}': Encontrados {len([l for l in lines if l.strip() and 'Modelo' in l])} registros. Exemplo: {lines[2:5]}..."  # Aproxima
            else:
                final_text = f"Resultados parciais: {sql_results[:300]}..."
        return final_text
    except Exception as e:
        print(f"❌ Erro na API Gemini (treat_response): {str(e)}")
        return f"Resposta temporária: {sql_results_trunc[:300]}..."

# ===========================
# 🔹 ARMAZENAR HISTÓRICO
# ===========================

def store_history(user_question, sql_query, sql_results, final_answer):
    # Trunca para histórico
    sql_results_trunc = sql_results[:400] + "..." if len(sql_results) > 400 else sql_results
    history_doc = f"""
    Pergunta: {user_question}
    SQL: {sql_query}
    Resultados (resumo): {sql_results_trunc}
    Resposta: {final_answer}
    """

    history_id = f"query_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    history_collection.add(documents=[history_doc], ids=[history_id])
    print(f"🧠 Histórico salvo (ID: {history_id})")

# ===========================
# 🔹 FUNÇÃO AUXILIAR PARA MEDIR TEMPO
# ===========================

def time_it(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"⏱️ {func.__name__} levou {elapsed:.2f}s")
    return result

# ===========================
# 🔹 PIPELINE PRINCIPAL RAG
# ===========================

def rag_pipeline(user_question):
    relevant_context = time_it(query_rag, user_question)
    sql_query = time_it(generate_sql, user_question, relevant_context)
    print(f"💬 SQL gerado: {sql_query}")

    results = time_it(execute_sql, sql_query)
    final_answer = time_it(treat_response, results, user_question)

    store_history(user_question, sql_query, results, final_answer)
    return final_answer

# ===========================
# 🔹 EXECUÇÃO DE EXEMPLO
# ===========================

if __name__ == "__main__":
    # populate_chroma()

    # Use a pergunta do erro para teste
    question = "Poderia me listar quais modelos tiveram mais entregas por país e mês?"
    answer = rag_pipeline(question)
    print(f"\n✅ Resposta: {answer}")