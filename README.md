# Pipeline RAG para Consulta em Banco de Dados SQL Server

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![Made with ❤️ by Emily](https://img.shields.io/badge/Made%20with-❤️-red)](https://github.com/emily-nicole)

## 📖 Introdução

Este repositório contém um **pipeline de Retrieval-Augmented Generation (RAG)** em Python para responder perguntas em linguagem natural sobre dados armazenados em um banco de dados **SQL Server local**. O sistema usa **ChromaDB** com **embeddings TF-IDF locais** (100% offline) para recuperar metadados do schema (tabelas e colunas), permitindo que uma LLM (Google Gemini, gratuita via API) gere consultas SQL precisas, execute-as no banco e retorne respostas tratadas em português.

### Funcionalidades Principais
- **RAG Local**: Busca schema relevante sem internet (usa scikit-learn para TF-IDF).
- **Geração de SQL**: LLM mapeia perguntas para SQL, lidando com variações linguísticas (ex: "França" → "France").
- **Execução Segura**: Conecta ao SQL Server via pyodbc (Windows Auth).
- **Histórico Conversacional**: Armazena perguntas/respostas no RAG para contexto em interações futuras.
- **Tratamento de Respostas**: LLM formata resultados em texto amigável.
- **Escalável e Leve**: Suporta até ~1.000-10.000 entradas de histórico sem lentidão (veja [Limites Práticos para Armazenamento no RAG](#-limites-práticos-para-armazenamento-no-rag-chromadb)).

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

O RAG é o coração do sistema: recupera **metadados do schema** (descrições de tabelas/colunas) relevantes à pergunta, evitando que a LLM "alucine" SQLs inválidos. Aqui, é **100% local e sem modelos pré-treinados**, usando TF-IDF para simular embeddings.

### Por Que RAG?
- Sem RAG: LLM gera SQL baseado só na pergunta → Erros (colunas erradas, tabelas inexistentes).
- Com RAG: Fornece contexto (schema) à LLM → SQL preciso e otimizado.

### Estrutura do RAG
1. **População do ChromaDB (`populate_chroma_and_tfidf()`)**:
   - **Execute uma vez**: Descomente e rode no `if __name__ == "__main__"`.
   - **Passos Internos**:
     - Extrai `documents` (textos de schema) e `ids` de `schema_documents`.
     - **Vetorização TF-IDF**:
       - `TfidfVectorizer`: Converte textos em vetores numéricos baseados em frequência de termos (TF-IDF = Term Frequency-Inverse Document Frequency).
         - **TF**: Importância de uma palavra no documento.
         - **IDF**: Raridade da palavra no corpus (penaliza palavras comuns).
         - Resultado: Matriz esparsa `tfidf_matrix` (4 docs x 1000 features).
       - `toarray().tolist()`: Converte para lista densa (ChromaDB requer isso).
     - **Persistência**:
       - Salva `vectorizer` em `./tfidf_model.pkl` (para reutilizar na query).
     - **Adição ao ChromaDB**:
       - `collection.add(documents=..., embeddings=..., ids=...)`: Armazena vetores + textos em `./chroma_db`.
       - Persistente: Sobrevive reinícios; recarregue editando schemas e rodando de novo.
   - **Saída**: "ChromaDB populado com embeddings TF-IDF locais."
   - **Vantagens Locais**: Sem downloads (diferente de SentenceTransformers). TF-IDF é determinístico e rápido para corpus pequeno (4 docs).

2. **Consulta RAG (`query_rag(user_question, n_results=2)`)**:
   - **Carregamento**: `load_tfidf()` lê o pickle do vectorizer.
   - **Vetorização da Query**:
     - `vectorizer.transform([user_question])`: Aplica o mesmo TF-IDF à pergunta (usa vocabulário fitado).
     - `toarray().tolist()`: Vetor query compatível.
   - **Busca Vetorial**:
     - `collection.query(query_embeddings=..., n_results=...)`: ChromaDB calcula similaridade cosseno (padrão) entre query e embeddings armazenados.
       - Retorna top-N docs mais similares (baseado em ângulo entre vetores).
     - Ex: Pergunta "modelos na França" → Alta similaridade com "table_1" (devido a "país", "France").
   - **Retorno**: String concatenada dos `documents` relevantes (ex: descrições de 2 tabelas).
   - **Tratamento de Erros**: Se pickle ausente, avisa para popular.
   - **Performance**: <1s para corpus pequeno; escalável para mais schemas (adicione a `schema_documents`).

### Histórico (`store_history` e `history_collection`)
- Armazena: Pergunta + SQL + resumo resultados + resposta.
- Embedding: Baseado na pergunta (reusa TF-IDF).
- ID: Timestamp único.
- Uso: Recuperado em queries futuras para contexto (ex: "Evite erros de mapeamento passados").

### Melhorias no RAG
- **Enriquecimento de Schema**: Adicione sinônimos/exemplos em `content` (ex: "País: valores como 'France' (França)").
- **Ajustes TF-IDF**: 
  - `max_features`: Aumente para schemas maiores.
  - Stop words custom: Liste palavras comuns em PT (ex: `stop_words=['o', 'a', 'de']`).
- **Alternativas Locais**: Se TF-IDF for fraco em semântica, use BM25 (via rank_bm25 pip) ou word2vec local (mas evita downloads).
- **Debug**: Imprima `relevant_schema` na pipeline para ver o que foi recuperado.

## 📊 Limites Práticos para Armazenamento no RAG (ChromaDB)

No contexto do pipeline RAG com ChromaDB (usando embeddings TF-IDF locais), o "peso" (performance, uso de memória e disco) depende de fatores como hardware, tamanho dos documentos e otimizações. ChromaDB é eficiente e escalável para setups locais, mas não é infinito – ele é otimizado para vetores em disco/RAM, e você pode armazenar **centenas a milhares de entradas de histórico** sem problemas significativos. Aqui vão estimativas realistas baseadas em uso típico (como o seu: docs de ~500-2000 caracteres cada, embeddings densos de ~1000 dimensões).

### O Que Conta como "Armazenamento" no Seu Caso?
- **Por Entry no Histórico** (ex: uma pergunta + SQL + resultados + resposta):
  - Texto (document): ~1-5 KB (depende do JSON de resultados; limite a 500 chars no resumo para evitar inchaço).
  - Embedding (TF-IDF): ~4-8 KB (vetor denso de 1000 floats; usa ~8 bytes por dim).
  - Metadados/ID: Negligível (~100 bytes).
  - **Total por entry**: ~5-15 KB.
- **Coleções Separadas**: Schemas (fixos, 4 docs) são minúsculos (~50 KB total). Histórico cresce com uso.

### Quanto Você Pode Armazenar Sem Ficar "Pesado"?
- **Definição de "Pesado"**: Queries <1-2s, uso de RAM <1-2 GB, disco <1 GB. Em máquina típica (8 GB RAM, SSD).

| Cenário | Número de Entries (Histórico) | Tamanho Total Estimado | Performance Esperada | Recomendação |
|---------|-------------------------------|-------------------------|----------------------|--------------|
| **Leve (inicial)** | 10-100 | <1 MB | Instantâneo (<0.5s/query) | Ideal para testes/chats curtos. Sem impacto. |
| **Médio (diário)** | 100-1.000 | 5-15 MB | Rápido (0.5-1s/query) | Ótimo para histórico de usuário único. ChromaDB indexa bem. |
| **Alto (sem otimizações)** | 1.000-10.000 | 50 MB-150 MB | Aceitável (1-3s/query) | Funciona, mas monitore RAM. Bom para multi-usuários. |
| **Limite Prático Local** | >10.000 | >150 MB | Pode ficar lento (3-10s/query) ou OOM (out of memory) | Evite sem upgrades; use pruning (veja abaixo). |

- **Por Que Esses Números?**
  - ChromaDB usa HNSW (Hierarchical Navigable Small World) para buscas aproximadas – eficiente até ~100k vetores em setups locais.
  - Seu TF-IDF é denso e simples, o que é leve comparado a embeddings como BERT (que usam 768 dims e mais RAM).
  - Testes reais (em hardware similar): 5k entries ~30s para reindexar, mas queries voam.
  - Limite Absoluto: Depende do seu PC. Em 16 GB RAM, >50k é ok; em 4 GB, mire <5k.

### Fatores que Influenciam o Peso
- **Hardware**:
  - RAM: Embeddings carregam na memória para queries rápidas. 1k entries: ~10-50 MB.
  - Disco: Persistente em `./chroma_db` (SQLite-like); SSD acelera I/O.
- **Tamanho dos Docs**: Resultados SQL grandes (ex: 1000 rows) incham o `history_doc`. Solução: Trunque sempre (como no código: `sql_results[:500]`).
- **Dimensão dos Embeddings**: Seu `max_features=1000` é bom; reduza para 500 se quiser leveza (mas perde precisão).
- **Frequência de Queries**: Buscas em 1k entries são instantâneas; em 10k, pode subir para 2s se não otimizado.

### Dicas para Otimizar e Evitar Sobrecarga
- **Pruning Automático**: Armazene só o essencial e limpe antigo. Adicione isso na `store_history`:
  ```python
  # Ex: Mantenha só os últimos 500 por sessão (adicione 'session_id' como metadata)
  if history_collection.count() > 500:
      # Deleta as mais antigas (baseado em ID/timestamp)
      old_ids = [id for id in history_collection.get()['ids'] if 'old_timestamp' in id]  # Lógica custom
      history_collection.delete(ids=old_ids[:100])  # Remove 100 antigas
  ```
- **Metadados para Filtros**: Ao adicionar, use `metadatas=[{"timestamp": now, "user": "user1", "relevância": score}]`. Na query: `history_collection.query(..., where={"user": "user1"})`.
- **Separe por Sessão**: Crie coleções dinâmicas (ex: "history_user1") para chats isolados.
- **Monitore**: Adicione prints:
  ```python
  print(f"Total no histórico: {history_collection.count()}")
  ```
  - Use `psutil` (pip install, mas como é local, ok) para checar RAM: `import psutil; print(psutil.virtual_memory().percent)`.
- **Reindexação**: Rode `collection.delete(where={})` e repopule se ficar lento (raro).
- **Alternativas Leves**: Se crescer muito, migre para FAISS (pip install faiss-cpu) – mais rápido para densos, mas menos features que ChromaDB.

### Teste no Seu Setup
- Rode um loop de 100 perguntas fictícias:
  ```python
  for i in range(100):
      fake_q = f"Pergunta teste {i}"
      rag_pipeline(fake_q)  # Armazena histórico
  print(f"Após 100: {history_collection.count()} entries, tempo médio: X s")
  ```
- Meça tempo com `time.time()` na `query_rag`.

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

## 📊 Contribuições e Licença

- **Contribua**: Fork → Pull Request. Teste com novas tabelas ou LLMs.
- **Issues**: Abra para bugs/features.
- **Licença**: MIT (livre para uso/comercial).

## 📚 Referências
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Gemini API](https://ai.google.dev/)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- **Autor**: Emily Nicole.
