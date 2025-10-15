# Pipeline RAG para Consulta em Banco de Dados SQL Server

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Made with ❤️ by Emily](https://img.shields.io/badge/Made%20with-❤️-red)](https://github.com/emily-nicole)

## 📖 Introdução

Este repositório contém um **pipeline de Retrieval-Augmented Generation (RAG)** em Python para responder perguntas em linguagem natural sobre dados armazenados em um banco de dados **SQL Server local**. O sistema usa **ChromaDB** com **embeddings TF-IDF locais** (100% offline) para recuperar metadados do schema (tabelas e colunas), permitindo que uma LLM (Google Gemini, gratuita via API) gere consultas SQL precisas, execute-as no banco e retorne respostas tratadas em português.

### Funcionalidades Principais
- **RAG Local**: Busca schema relevante sem internet (usa scikit-learn para TF-IDF).
- **Geração de SQL**: LLM mapeia perguntas para SQL, lidando com variações linguísticas (ex: "França" → "France").
- **Execução Segura**: Conecta ao SQL Server via pyodbc (Windows Auth).
- **Histórico Conversacional**: Armazena perguntas/respostas no RAG para contexto em interações futuras.
- **Tratamento de Respostas**: LLM formata resultados em texto amigável.
- **Escalável e Leve**: Suporta até ~1.000-10.000 entradas de histórico sem lentidão (veja [Otimização](#-otimização-e-limites)).

**Exemplo de Uso**:
- Pergunta: "Qual o país com mais entregas em análise?"
- Resposta: "O país com mais entregas em análise é o Brasil, com 15 registros."

### Arquitetura
```
Usuário → Pergunta → [RAG: Schema + Histórico] → LLM (SQL) → Execução SQL → LLM (Resposta) → Armazenar Histórico
```

- **RAG**: ChromaDB (vetor DB persistente).
- **LLM**: Gemini API (gratuita).
- **DB**: SQL Server local (`datasets`).

## 🚀 Instalação

### Pré-requisitos
- **Python 3.8+**: [Baixe aqui](https://www.python.org/downloads/).
- **SQL Server Local**: Server `EMILYNICOLE`, DB `datasets` (Windows Auth). Teste conexão via SSMS.
- **Ambiente Virtual**: Recomendado (ex: `python -m venv .venv; source .venv/bin/activate` no Linux/Mac ou `.venv\Scripts\activate` no Windows).
- **Driver ODBC**: Instale [ODBC Driver 17 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).

### Passos
1. Clone o repositório:
   ```
   git clone <seu-repo-url>
   cd chatbot
   ```

2. Instale dependências:
   ```
   pip install chromadb scikit-learn pyodbc google-generativeai pandas python-dotenv
   ```

3. Configure a API Gemini:
   - Crie chave gratuita no [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Crie `.env` na raiz:
     ```
     GEMINI_API_KEY=sua-chave-aqui
     ```
   - (Opcional) Defina no terminal: `$env:GEMINI_API_KEY='sua-chave'` (PowerShell).

4. Popule o RAG (rode o script uma vez):
   ```
   python main.py  # Descomente populate_chroma_and_tfidf() no if __name__
   ```

## 📁 Estrutura do Código

```
chatbot/
├── main.py                  # Script principal com pipeline
├── .env                     # Configs (não commite!)
├── chroma_db/               # Pasta gerada: ChromaDB persistente
├── tfidf_model.pkl          # Vectorizer TF-IDF salvo
└── README.md                # Este arquivo
```

### Funções Principais
- **`populate_chroma_and_tfidf()`**: Prepara embeddings TF-IDF e popula schemas no ChromaDB (rode uma vez).
- **`query_rag(user_question)`**: Recupera schema + histórico relevante via similaridade TF-IDF.
- **`generate_sql(user_question, context)`**: LLM gera SQL com prompt engenheirado (inclui mapeamentos linguísticos).
- **`execute_sql(sql_query)`**: Executa no SQL Server e retorna JSON.
- **`treat_response(results, question)`**: LLM formata resposta em PT-BR.
- **`store_history(...)`**: Armazena query no histórico (nova coleção).
- **`rag_pipeline(user_question)`**: Fluxo completo (RAG → SQL → Exec → Resposta → Histórico).

### Schema Hardcoded
Em `schema_documents`: Descrições de 4 tabelas (TabelaOriginal, nutrition_cf, GraduateEmployment, game_data_all). Expanda adicionando dicts com `id` e `content` (descrições ricas com colunas/exemplos).

## 🛠️ Uso

### Execução Básica
1. Rode o script:
   ```
   python main.py
   ```
   - Isso popula o RAG (se não feito) e testa uma pergunta exemplo.

2. Exemplo Interativo (adicione no `if __name__`):
   ```python
   while True:
       question = input("Faça uma pergunta: ")
       if question.lower() == 'sair':
           break
       answer = rag_pipeline(question)
       print(f"Resposta: {answer}\n")
   ```

### Exemplos de Perguntas
- "Quantos modelos na França?" → Gera `SELECT COUNT(*) FROM [dbo].[TabelaOriginal] WHERE [País] = 'France';`.
- "Qual a taxa de emprego em 2023?" → Recupera schema de GraduateEmployment.
- "Histórico": Perguntas sequenciais usam contexto anterior para consistência.

### Customizações
- **Adicionar Tabelas**: Edite `schema_documents`, rode `populate_chroma_and_tfidf()`.
- **Mapeamentos Linguísticos**: Expanda no prompt de `generate_sql` (ex: "Alemanha=Germany").
- **Modelo LLM**: Mude `gemini-1.5-flash` para `gemini-1.5-pro` (mais preciso, mas quota limitada).

## 🔍 Componente RAG: Detalhes Técnicos

O RAG é **100% local** (sem modelos pré-treinados ou internet para busca/embeddings).

### População (`populate_chroma_and_tfidf`)
- Vetoriza schemas com **TF-IDF** (scikit-learn): Converte texto em vetores baseados em frequência de termos.
- Armazena em ChromaDB: `add(documents, embeddings, ids)`.
- Persiste vectorizer em pickle para queries.

### Consulta (`query_rag`)
- Vetoriza pergunta: `vectorizer.transform([question])`.
- Busca paralela: Schemas (top 2) + Histórico (top 2).
- Similaridade: Cosseno (padrão do ChromaDB).
- Retorno: Contexto concatenado para LLM.

### Histórico (`store_history` e `history_collection`)
- Armazena: Pergunta + SQL + resumo resultados + resposta.
- Embedding: Baseado na pergunta (reusa TF-IDF).
- ID: Timestamp único.
- Uso: Recuperado em queries futuras para contexto (ex: "Evite erros de mapeamento passados").

### Limites e Otimização
- **Armazenamento**: ~5-15 KB/entry. Suporta 1k-10k sem lentidão.
- **Pruning**: Adicione na `store_history`:
  ```python
  if history_collection.count() > 500:
      all_ids = history_collection.get()['ids']
      old_ids = all_ids[:-100]  # Mantém últimas 100
      history_collection.delete(ids=old_ids)
  ```
- **Performance**: <1s/query para corpus pequeno. Monitore com `print(history_collection.count())`.

## ⚠️ Troubleshooting

| Problema | Causa | Solução |
|----------|-------|---------|
| **dotenv parsing** | `.env` inválido | Verifique sintaxe (sem espaços em `= `). |
| **ChromaDB embeddings error** | Nesting extra | Use código atualizado (embeddings direto). |
| **pyodbc ConnectionError** | Driver ausente | Instale ODBC 17; teste `pyodbc.connect(conn_str)`. |
| **Pandas Warning (SQLAlchemy)** | pyodbc não recomendado | Ignore ou migre para SQLAlchemy (`pip install sqlalchemy`). |
| **Gemini API falha** | Chave/quota | Teste chave; verifique [AI Studio](https://aistudio.google.com). |
| **RAG schema errado** | TF-IDF fraco | Enriqueça `content` com sinônimos; aumente `n_results=3`. |
| **Histórico vazio** | Primeira execução | Rode 2+ perguntas; cheque `print(relevant_context)`. |
| **SQL inválido** | Mismatch tabela | Adicione `[dbo].` no prompt exemplo. |


## 📚 Referências
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Gemini API](https://ai.google.dev/)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- **Autor**: Emily Nicole.
