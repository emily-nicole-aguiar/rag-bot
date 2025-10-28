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
# 🔹 NOVO SCRIPT COM SUGESTÕES IMPLEMENTADAS (v2)
# ==============================================================================
#
# Principais Alterações:
# 1. LOOP DE AUTO-CORREÇÃO (NOVO): Se a execução do SQL falhar, a pipeline
#    captura a mensagem de erro e pede ao LLM para corrigir o SQL.
#    Isso é feito na `rag_pipeline` e `generate_sql`.
#
# 2. GLOSSÁRIO NA RESPOSTA (NOVO): O contexto do schema (glossário) buscado
#    no RAG é agora passado para a função `treat_response` para que a
#    explicação final ao usuário seja mais precisa e rica em contexto.
#
# 3. MUDANÇAS DE FUNÇÃO:
#    - `execute_sql` agora retorna (resultado, erro) para controle.
#    - `generate_sql` aceita `failed_sql` e `error_message` para correção.
#    - `query_rag_with_cache` foi otimizado para sempre retornar o
#      contexto do schema, mesmo quando o cache é ativado.
#    - `rag_pipeline` orquestra o novo loop de correção.
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
    "gemini-1.5-flash",
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
    name="schema_collection_v3_enriched",
    embedding_function=embedding_fn
)
history_collection = client.get_or_create_collection(
    name="history_collection_v3",
    embedding_function=embedding_fn
)

# ===========================
# 🔹 SCHEMAS DAS TABELAS
# ===========================

# (Schemas enriquecidos mantidos da versão anterior)
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
# 🔹 POPULAR SCHEMA
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
# 🔹 CONSULTA RAG (OTIMIZADA PARA SEMPRE RETORNAR SCHEMA)
# ==============================================================================

def query_rag_with_cache(user_question, n_results=3, cache_threshold=0.15):
    # Passo 1: Buscar SEMPRE o contexto do schema.
    # Isso é necessário para o "Glossário" na etapa final.
    schema_results = collection.query(query_texts=[user_question], n_results=n_results)
    schema_relevant = "\n".join(schema_results["documents"][0]) if schema_results["documents"] else ""

    # Passo 2: Buscar no histórico para tentar o cache.
    history_results = history_collection.query(
        query_texts=[user_question],
        n_results=1,
        include=["documents", "distances"]
    )

    cached_sql = None
    # Se encontramos um resultado e a distância é muito pequena (alta similaridade)...
    if history_results["documents"] and history_results["distances"][0][0] < cache_threshold:
        print(f"🔍 Encontrada pergunta similar no histórico (distância: {history_results['distances'][0][0]:.3f}). Ativando cache.")
        history_doc = history_results["documents"][0][0]
        if "SQL:" in history_doc:
            cached_sql = history_doc.split("SQL:")[1].split("\n")[0].strip()
            # Retornamos o schema (para o glossário) e o SQL do cache
            return schema_relevant, cached_sql

    # Passo 3: Se não há cache, montar o contexto completo (schema + histórico)
    history_context_docs = history_results["documents"][0] if history_results["documents"] else []
    history_relevant = "\n".join(history_context_docs) if history_context_docs else "Nenhum histórico relevante."
    
    full_context = f"Schema relevante:\n{schema_relevant}\n\nExemplo de pergunta/SQL recente:\n{history_relevant}"
    
    return full_context, None # Retorna o contexto completo e None para o cache

# ==============================================================================
# 🔹 GERAR SQL (ATUALIZADO PARA AUTO-CORREÇÃO)
# ==============================================================================

def generate_sql(user_question, relevant_context, failed_sql=None, error_message=None):
    
    # Se failed_sql for fornecido, o modo é "CORREÇÃO"
    if failed_sql:
        prompt = f"""
        Você é um especialista em corrigir T-SQL para SQL Server.
        A query anterior falhou. Por favor, corrija-a.

        <contexto_schema>
        {relevant_context}
        </contexto_schema>

        <query_com_erro>
        {failed_sql}
        </query_com_erro>

        <mensagem_de_erro_sql>
        {error_message}
        </mensagem_de_erro_sql>

        <pergunta_usuario_original>
        {user_question}
        </pergunta_usuario_original>

        Instruções de Correção:
        1.  Analise o <mensagem_de_erro_sql> e o <contexto_schema>.
        2.  O erro é provavelmente um nome de coluna ou tabela incorreto. Use o contexto para encontrar o nome correto.
        3.  Responda APENAS com o código SQL corrigido, sem explicações.

        SQL Corrigido:
        """
    else:
        # Modo padrão "GERAÇÃO"
        prompt = f"""
        Você é um especialista em gerar código T-SQL para SQL Server.
        Sua tarefa é gerar uma ÚNICA query SQL válida baseada na pergunta do usuário e no contexto do schema fornecido.

        <instruções>
        1.  Use APENAS as tabelas e colunas descritas no contexto.
        2.  Preste atenção especial às seções de RELACIONAMENTOS para criar JOINs corretos.
        3.  Responda APENAS com o código SQL, sem explicações ou formatação ```sql.
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
        # Limpeza final
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        return sql_query.strip()
    except Exception as e:
        print(f"❌ Erro na API Gemini (generate_sql): {str(e)}")
        return "SELECT 'Erro ao gerar SQL' AS Error;"

# ==============================================================================
# 🔹 CAMADA DE SEGURANÇA (ATUALIZADA)
# ==============================================================================

def generate_safe_sql(user_question, relevant_context, failed_sql=None, error_message=None):
    # Passa todos os argumentos para a função de geração
    raw_sql = generate_sql(user_question, relevant_context, failed_sql, error_message)

    # Regras de segurança são aplicadas em AMBAS as gerações (inicial e correção)
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "CREATE", "EXEC"]
    for keyword in forbidden_keywords:
        if keyword in raw_sql.upper():
            print(f"⚠️ ALERTA DE SEGURANÇA: SQL bloqueado por conter '{keyword}': {raw_sql}")
            return "SELECT 'Comando não permitido detectado.' AS SecurityError;"

    # Forçar TOP 1000 apenas se não for uma correção (correções podem já tê-lo)
    if not failed_sql and "SELECT" in raw_sql.upper() and "LIMIT" not in raw_sql.upper() and "TOP" not in raw_sql.upper():
        raw_sql = raw_sql.replace("SELECT", "SELECT TOP 1000", 1)
        print("🛡️ Adicionado 'TOP 1000' por segurança à query.")

    return raw_sql


# ==============================================================================
# 🔹 EXECUTAR SQL LOCALMENTE (ATUALIZADO PARA RETORNAR ERRO)
# ==============================================================================
def execute_sql(sql_query):
    try:
        df = pd.read_sql(sql_query, engine)
        
        if len(df) > 50:
             json_results = df.head(50).to_json(orient='records', indent=2) + f"\n... (resultados truncados, mostrando 50 de {len(df)} registros)"
        else:
             json_results = df.to_json(orient='records', indent=2)
        
        # Sucesso: Retorna os resultados e None para o erro
        return json_results, None
    except Exception as e:
        # Falha: Retorna None para os resultados e a string do erro
        error_message = str(e)
        print(f"❌ Erro ao executar SQL: {error_message}")
        return None, error_message

# ==============================================================================
# 🔹 TRATAR RESPOSTA COM GEMINI (ATUALIZADO COM GLOSSÁRIO)
# ==============================================================================
def treat_response(sql_results, user_question, schema_context):
    if len(sql_results) > 1500:
        sql_results = sql_results[:1500] + "\n... (dados truncados)"
        
    if len(schema_context) > 1000:
        schema_context = schema_context[:1000] + "\n... (glossário truncado)"

    prompt = f"""
    Com base nos dados JSON, responda à pergunta do usuário.
    Use o "Glossário" para entender o significado das colunas e explicar a resposta corretamente em português.

    <glossario_de_dados>
    {schema_context}
    (Ex: 'gross_monthly_median' é a mediana do salário mensal bruto.)
    </glossario_de_dados>

    <dados_json>
    {sql_results}
    </dados_json>

    <pergunta_original>
    {user_question}
    </pergunta_original>

    Instruções da Resposta:
    - Responda de forma clara e concisa.
    - Use o glossário para explicar os termos técnicos (ex: "A mediana do salário bruto (gross_monthly_median) foi...").
    - Se houver uma mensagem de erro nos dados, explique o erro de forma amigável.
    - Não mencione JSON ou SQL.

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
# 🔹 PIPELINE PRINCIPAL RAG (COM LOOP DE AUTO-CORREÇÃO)
# ==============================================================================

def rag_pipeline(user_question):
    print("\n" + "="*50)
    print(f"🚀 Iniciando pipeline para a pergunta: '{user_question}'")
    print("="*50)

    # 1. Consultar RAG e verificar cache.
    # 'relevant_context' sempre conterá o schema para o 'Glossário'.
    relevant_context, cached_sql = time_it(query_rag_with_cache, user_question)

    if cached_sql:
        print("⚡ Cache Semântico Ativado! Reutilizando SQL.")
        sql_query = cached_sql
    else:
        print("🧠 Cache não ativado. Gerando novo SQL com segurança.")
        # 2. Gerar SQL de forma segura
        sql_query = time_it(generate_safe_sql, user_question, relevant_context)

    print(f"💬 SQL (Tentativa 1): {sql_query}")

    # 3. Executar SQL (Tentativa 1)
    results, error = time_it(execute_sql, sql_query)

    # 4. LOOP DE AUTO-CORREÇÃO
    if error:
        print(f"⚠️ SQL (Tentativa 1) falhou: {error}")
        print("🔄 Iniciando tentativa de auto-correção...")
        
        # Tenta gerar um SQL corrigido
        sql_query = time_it(
            generate_safe_sql,
            user_question,
            relevant_context,
            failed_sql=sql_query,
            error_message=error
        )
        
        print(f"💬 SQL (Tentativa 2 - Corrigido): {sql_query}")
        
        # 5. Executar SQL (Tentativa 2)
        results, error = time_it(execute_sql, sql_query)
        
        # 6. Se falhar novamente, desiste e passa o erro para o usuário
        if error:
            print(f"❌ Auto-correção falhou: {error}")
            results = json.dumps({
                "error": f"Falha ao executar a query corrigida. Erro: {error}",
                "sql_tentada": sql_query
            }, indent=2)

    # 7. Tratar resposta para o usuário (agora com o glossário)
    # 'relevant_context' é o 'schema_relevant' do RAG
    final_answer = time_it(treat_response, results, user_question, relevant_context)

    # 8. Salvar no histórico apenas se for novo, bem-sucedido e não do cache
    if not cached_sql and not error:
        time_it(store_history, user_question, sql_query)
    
    return final_answer

# ===========================
# 🔹 EXECUÇÃO DE EXEMPLO
# ===========================

if __name__ == "__main__":
    populate_chroma()

    # --- TESTE 1: Pergunta que requer um JOIN ---
    question1 = "Quais os 5 modelos mais entregues pela operadora EntregaGlobal no Brasil?"
    answer1 = rag_pipeline(question1)
    print(f"\n✅ Resposta Final 1:\n{answer1}")

    # --- TESTE 2: Pergunta que usa o glossário ---
    question2 = "Qual universidade teve a maior gross_monthly_median em 2019?"
    answer2 = rag_pipeline(question2)
    print(f"\n✅ Resposta Final 2:\n{answer2}")
    
    # --- TESTE 3: Teste de CACHE SEMÂNTICO ---
    question3 = "qual a universidade com maior salario bruto mediano em 2019?"
    answer3 = rag_pipeline(question3)
    print(f"\n✅ Resposta Final 3 (do cache):\n{answer3}")
    
    # --- TESTE 4: Teste de AUTO-CORREÇÃO DE SQL ---
    # Pergunta com nome de coluna errado ('employ_rate' em vez de 'employment_rate_overall')
    # Isso deve forçar a falha do SQL e ativar o loop de correção.
    question4 = "Qual o 'employ_rate' médio da universidade NUS?"
    answer4 = rag_pipeline(question4)
    print(f"\n✅ Resposta Final 4 (Auto-Corrigida):\n{answer4}")
