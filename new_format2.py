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
import time
from sqlalchemy import create_engine
import json

# ==============================================================================
# 🔹 NOVO SCRIPT COM SUGESTÕES IMPLEMENTADAS
# ==============================================================================
#
# Principais Alterações:
# 1. METADADOS ENRIQUECIDOS: Os schemas das tabelas agora são muito mais
#    detalhados, incluindo tipos de dados, exemplos e, crucialmente,
#    instruções explícitas de RELACIONAMENTO para guiar a criação de JOINs.
#
# 2. CACHE SEMÂNTICO: A pipeline agora verifica se uma pergunta muito similar
#    já foi feita. Se sim, reutiliza o SQL gerado anteriormente para economizar
#    tempo e custos de API.
#
# 3. CAMADA DE SEGURANÇA: Uma nova função `generate_safe_sql` foi adicionada
#    para bloquear comandos SQL perigosos (UPDATE, DELETE) e para adicionar
#    automaticamente um `TOP 1000` a todas as consultas, prevenindo sobrecarga.
#
# 4. PROMPT OTIMIZADO: O prompt enviado ao Gemini foi reestruturado com tags
#    para melhorar a clareza e a precisão da geração de SQL.
#
# ==============================================================================


# ===========================
# 🔹 CONFIGURAÇÕES INICIAIS
# ===========================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Defina GEMINI_API_KEY no .env ou como variável de ambiente!")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=2048,
        temperature=0.1
    )
)

SERVER = "EMILYNICOLE"
DATABASE = "datasets"
engine_str = f"mssql+pyodbc://@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(engine_str)

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="modelos/all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="schema_collection_v3_enriched", # Nova versão da coleção
    embedding_function=embedding_fn
)
history_collection = client.get_or_create_collection(
    name="history_collection_v3",
    embedding_function=embedding_fn
)

# ==============================================================================
# 🔹 ENRIQUECIMENTO DE METADADOS: SCHEMAS DAS TABELAS (NOVO FORMATO)
# ==============================================================================

# Para demonstrar JOINs, criamos uma segunda tabela hipotética: CadastroOperadoras
schema_documents = [
    {
        "id": "table_TabelaOriginal",
        "content": """
        Tabela: TabelaOriginal. 
        Descrição: Tabela principal (fato) que registra cada entrega de um modelo em um país. Use esta tabela para perguntas sobre quantidades, datas e status.
        Colunas: 
        - Modelo (TEXTO): O nome do modelo do item entregue. Ex: 'ModeloX', 'ProdutoY'.
        - País (TEXTO): O país de destino da entrega. Mapeie nomes em português para inglês (ex: Brasil -> 'Brazil', França -> 'France').
        - Data_Entrega (DATA): A data exata em que a entrega foi realizada. Use funções como YEAR() e MONTH() para agregar por período.
        - Status (TEXTO): A situação atual da entrega. Valores comuns: 'Done', 'Completed', 'Analysis'.
        - id_operadora (INTEIRO): Chave estrangeira que identifica a operadora. Use esta coluna para conectar com a tabela 'CadastroOperadoras'.
        
        RELACIONAMENTOS:
        - Para saber o NOME da operadora, faça um JOIN com a tabela 'CadastroOperadoras' usando: TabelaOriginal.id_operadora = CadastroOperadoras.id.
        """
    },
    {
        "id": "table_CadastroOperadoras",
        "content": """
        Tabela: CadastroOperadoras.
        Descrição: Tabela de cadastro (dimensão) com os nomes das operadoras de logística.
        Colunas:
        - id (INTEIRO): Chave primária única da operadora.
        - nome_operadora (TEXTO): O nome comercial da operadora. Ex: 'LogiFast', 'RapidTrans', 'EntregaGlobal'.
        
        RELACIONAMENTOS:
        - Esta tabela se conecta à 'TabelaOriginal' pela coluna 'id'. Use-a para buscar o nome da operadora a partir de um 'id_operadora'.
        """
    },
    {
        "id": "table_GraduateEmployment",
        "content": """
        Tabela: GraduateEmployment.
        Descrição: Contém dados anuais sobre o emprego de graduados em Singapura, incluindo taxas de emprego e salários por universidade e curso.
        Colunas:
        - year (INTEIRO): Ano da pesquisa.
        - university (TEXTO): Nome da universidade.
        - degree (TEXTO): Nome do curso/graduação.
        - employment_rate_overall (NUMÉRICO): Taxa de emprego geral (de 0 a 100).
        - basic_monthly_mean (NUMÉRICO): Salário mensal básico médio.
        - gross_monthly_median (NUMÉRICO): Mediana do salário mensal bruto.
        """
    }
]

# ===========================
# 🔹 POPULAR SCHEMA (execute uma vez)
# ===========================

def populate_chroma():
    existing = collection.count()
    if existing > 0:
        print(f"⚠️ ChromaDB (schema) já possui {existing} documentos. Pular população.")
        return

    documents = [doc["content"] for doc in schema_documents]
    ids = [doc["id"] for doc in schema_documents]

    collection.add(documents=documents, ids=ids)
    print("✅ Schemas enriquecidos adicionados ao ChromaDB.")

# ==============================================================================
# 🔹 CONSULTA RAG COM CACHE SEMÂNTICO
# ==============================================================================

def query_rag_with_cache(user_question, n_results=3, cache_threshold=0.15):
    # Passo 1: Buscar no histórico por uma pergunta muito similar para usar como cache
    history_results = history_collection.query(
        query_texts=[user_question],
        n_results=1,
        include=["documents", "distances"]  # Importante: pedir as distâncias
    )

    cached_sql = None
    # Verificação segura: se há documentos e distâncias válidas
    if (history_results["documents"] and 
        len(history_results["documents"][0]) > 0 and 
        history_results["distances"] and 
        len(history_results["distances"][0]) > 0 and 
        history_results["distances"][0][0] < cache_threshold):
        print(f"🔍 Encontrada pergunta similar no histórico (distância: {history_results['distances'][0][0]:.3f}). Ativando cache.")
        history_doc = history_results["documents"][0][0]
        # Extrai o SQL do documento de histórico
        if "SQL:" in history_doc:
            cached_sql = history_doc.split("SQL:")[1].split("\n")[0].strip()
            # Retornamos o SQL do cache e pulamos o resto da busca de contexto
            return "", cached_sql

    # Passo 2: Se não há cache, buscar contexto do schema e do histórico para gerar um novo SQL
    schema_results = collection.query(query_texts=[user_question], n_results=n_results)
    schema_relevant = "\n".join(schema_results["documents"][0]) if schema_results["documents"] else ""

    history_context_docs = history_collection.query(query_texts=[user_question], n_results=1)
    history_relevant = "\n".join(history_context_docs["documents"][0]) if history_context_docs["documents"] else "Nenhum histórico relevante."

    full_context = f"Schema relevante:\n{schema_relevant}\n\nExemplo de pergunta/SQL recente:\n{history_relevant}"
    return full_context, None # Retorna o contexto completo e None para o cache

# ==============================================================================
# 🔹 GERAR SQL VIA GEMINI (COM PROMPT OTIMIZADO)
# ==============================================================================

def generate_sql(user_question, relevant_context):
    prompt = f"""
    Você é um especialista em gerar código T-SQL para SQL Server.
    Sua tarefa é gerar uma ÚNICA query SQL válida baseada na pergunta do usuário e no contexto do schema fornecido.

    <instruções>
    1.  Use APENAS as tabelas e colunas descritas no contexto.
    2.  Preste atenção especial às seções de RELACIONAMENTOS para criar JOINs corretos quando necessário.
    3.  Mapeie nomes em português para seus equivalentes em inglês (ex: França -> 'France', Brasil -> 'Brazil').
    4.  Responda APENAS com o código SQL, sem explicações, comentários ou formatação ```sql.
    </instruções>

    <contexto_schema>
    {relevant_context}
    </contexto_schema>

    <pergunta_usuario>
    {user_question}
    </pergunta_usuario>

    SQL Gerado:
    """

    try:
        response = MODEL.generate_content(prompt)
        sql_query = response.text.strip()
        # Limpeza final para remover markdown que o modelo às vezes insiste em adicionar
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        return sql_query.strip()
    except Exception as e:
        print(f"❌ Erro na API Gemini (generate_sql): {str(e)}")
        return "SELECT 'Erro ao gerar SQL' AS Error;"

# ==============================================================================
# 🔹 NOVA CAMADA DE SEGURANÇA PARA SQL
# ==============================================================================

def generate_safe_sql(user_question, relevant_context):
    raw_sql = generate_sql(user_question, relevant_context)

    # Regra 1: Bloquear palavras-chave perigosas
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "CREATE", "EXEC"]
    for keyword in forbidden_keywords:
        if keyword in raw_sql.upper():
            print(f"⚠️ ALERTA DE SEGURANÇA: SQL bloqueado por conter '{keyword}': {raw_sql}")
            return "SELECT 'Comando não permitido detectado. Apenas consultas SELECT são autorizadas.' AS SecurityError;"

    # Regra 2: Forçar um limite de resultados (TOP para SQL Server) se não houver um
    if "SELECT" in raw_sql.upper() and "LIMIT" not in raw_sql.upper() and "TOP" not in raw_sql.upper():
        # Adiciona TOP 1000 após o primeiro SELECT
        raw_sql = raw_sql.replace("SELECT", "SELECT TOP 1000", 1)
        print("🛡️ Adicionado 'TOP 1000' por segurança à query.")

    return raw_sql


# ===========================
# 🔹 EXECUTAR SQL LOCALMENTE (sem alterações)
# ===========================
def execute_sql(sql_query):
    try:
        df = pd.read_sql(sql_query, engine)
        if len(df) > 50:
             json_results = df.head(50).to_json(orient='records', indent=2) + f"\n... (resultados truncados, mostrando 50 de {len(df)} registros)"
        else:
             json_results = df.to_json(orient='records', indent=2)
        return json_results
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ===========================
# 🔹 TRATAR RESPOSTA COM GEMINI (sem alterações)
# ===========================
def treat_response(sql_results, user_question):
    if len(sql_results) > 1500:
        sql_results = sql_results[:1500] + "\n... (dados truncados)"

    prompt = f"""
    Com base nos seguintes dados em formato JSON, responda à pergunta do usuário de forma clara e concisa em português.
    - Se os dados forem uma lista, resuma os principais pontos ou os top 5 resultados.
    - Se houver uma mensagem de erro, explique o erro de forma amigável.
    - Não mencione que os dados vieram de um JSON ou SQL. Apenas apresente a resposta.

    <dados_json>
    {sql_results}
    </dados_json>

    <pergunta_original>
    {user_question}
    </pergunta_original>

    Resposta final:
    """
    try:
        response = MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Erro na API Gemini (treat_response): {str(e)}")
        return f"Houve um problema ao processar os resultados. Dados brutos: {sql_results}"


# ===========================
# 🔹 ARMAZENAR HISTÓRICO
# ===========================
def store_history(user_question, sql_query):
    # Não precisamos mais salvar os resultados e a resposta, apenas o par pergunta/SQL
    history_doc = f"Pergunta: {user_question}\nSQL: {sql_query}"
    history_id = f"query_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    history_collection.add(documents=[history_doc], ids=[history_id])
    print(f"🧠 Histórico salvo (ID: {history_id})")

# ===========================
# 🔹 FUNÇÃO AUXILIAR PARA MEDIR TEMPO
# ===========================
def time_it(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"⏱️  {func.__name__:<25} levou {elapsed:.2f}s")
    return result

# ==============================================================================
# 🔹 PIPELINE PRINCIPAL RAG (ATUALIZADA)
# ==============================================================================

def rag_pipeline(user_question):
    print("\n" + "="*50)
    print(f"🚀 Iniciando pipeline para a pergunta: '{user_question}'")
    print("="*50)

    # 1. Consultar RAG e verificar cache
    relevant_context, cached_sql = time_it(query_rag_with_cache, user_question)

    if cached_sql:
        print("⚡ Cache Semântico Ativado! Reutilizando SQL.")
        sql_query = cached_sql
    else:
        print("🧠 Cache não ativado. Gerando novo SQL com segurança.")
        # 2. Gerar SQL de forma segura
        sql_query = time_it(generate_safe_sql, user_question, relevant_context)

    print(f"💬 SQL a ser executado: {sql_query}")

    # 3. Executar SQL
    results = time_it(execute_sql, sql_query)
    
    # 4. Tratar resposta para o usuário
    final_answer = time_it(treat_response, results, user_question)

    # 5. Salvar no histórico apenas se a query for nova
    if not cached_sql and "Error" not in sql_query:
        time_it(store_history, user_question, sql_query)
    
    return final_answer

# ===========================
# 🔹 EXECUÇÃO DE EXEMPLO
# ===========================

if __name__ == "__main__":
    # Primeira execução irá popular o ChromaDB com os novos schemas
    populate_chroma()

    # --- TESTE 1: Pergunta que requer um JOIN ---
    # O sistema deve usar os metadados enriquecidos para juntar TabelaOriginal com CadastroOperadoras
    question1 = "Quais os 5 modelos mais entregues pela operadora Movistar no Brasil?"
    answer1 = rag_pipeline(question1)
    print(f"\n✅ Resposta Final 1:\n{answer1}")

    # --- TESTE 2: Pergunta simples sobre outra tabela ---
    question2 = "Qual universidade teve a maior média de notas em 2013?"
    answer2 = rag_pipeline(question2)
    print(f"\n✅ Resposta Final 2:\n{answer2}")
    
    # --- TESTE 3: Repetir uma pergunta similar para testar o CACHE SEMÂNTICO ---
    # A pipeline deve detectar a similaridade e reutilizar o SQL da pergunta 2
    question3 = "qual a universidade com maior desempenho em 2019?"
    answer3 = rag_pipeline(question3)
    print(f"\n✅ Resposta Final 3 (do cache):\n{answer3}")