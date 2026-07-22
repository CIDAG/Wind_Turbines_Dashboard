# Wind Dataset Dashboard

Dashboard em Streamlit para comparar datasets abertos de turbinas eólicas. A aplicação carrega os CSVs disponíveis na pasta do projeto, padroniza as colunas principais e mostra cobertura de variáveis, compatibilidade entre bases, curvas de potência, séries temporais, status operacional e indicadores de qualidade.

## Estrutura principal

- `wind_dashboard.py`: aplicação Streamlit.
- `wind_dashboard_core/loaders.py`: cadastro dos datasets e conversão das colunas para um formato comum.
- `wind_dashboard_core/metrics.py`: métricas de cobertura, qualidade e compatibilidade.
- `wind_dashboard_core/charts.py`: gráficos Plotly usados no dashboard.
- `datasets2020-penmanshiel/`: arquivos Penmanshiel usados pelo loader automático.

Os CSVs precisam ficar no mesmo layout esperado pelo projeto. O dashboard detecta os datasets registrados quando encontra os arquivos principais de cada um.

## Como rodar localmente

### 1. Clonar ou baixar o projeto

Se estiver usando Git:

```powershell
git clone https://github.com/EnzoBaldinotti/WT-project.git
```

### 2. Dados: pacote Zenodo

O repositório distribui código, notebooks, documentação e os outputs derivados em `output/`. Os datasets de entrada (Penmanshiel, EDP, Kaggle e curvas de referência) são fornecidos separadamente pelo pacote `ZENODO.zip`, evitando que o clone do GitHub baixe vários gigabytes de dados.

> **DOI/link do Zenodo:** adicione o link público do registro aqui quando a publicação estiver concluída.

Após clonar o repositório, baixe `ZENODO.zip` e extraia **o conteúdo** do ZIP na raiz do projeto, preservando os nomes e a estrutura:

O resultado esperado é:

```text
WT-project/
├─ wind_dashboard.py
├─ Wind_Turbine_SCADA_tratado_consolidado.csv
├─ Location1.csv
├─ ... demais CSVs da raiz ...
├─ datasets2020-penmanshiel/
│  ├─ Turbine_Data_Penmanshiel_*.csv
│  └─ Status_Penmanshiel_*.csv
└─ output/merged_datasets/  # somente para reprodução completa
```

O ZIP não deve criar uma camada adicional, como `WT-project/dados-do-zenodo/Location1.csv`. Caso isso aconteça, mova o conteúdo extraído um nível acima. O dashboard usa caminhos relativos e funcionará em qualquer computador quando esse layout for preservado.

Para executar o dashboard, extraia os CSVs EDP tratados, Kaggle, curvas de referência e Penmanshiel. O pacote também contém EDP bruto, MetMast, logs e falhas para reproduzir os notebooks. Os CSVs em `output/merged_datasets/` permanecem versionados com Git LFS; para obter esses outputs prontos, execute:

```powershell
git lfs install
git lfs pull
```

Os outputs podem ser regenerados com `merge_development_merges.ipynb` quando os dados do Zenodo estiverem disponíveis.

### 3. Criar um ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação do ambiente, rode:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

No macOS ou Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar as dependências

```powershell
python -m pip install --upgrade pip
pip install streamlit pandas numpy plotly scikit-learn
```

### 5. Conferir os arquivos de dados

O dashboard procura os datasets cadastrados em `wind_dashboard_core/loaders.py`. Hoje os principais são:

- EDP SCADA: `Wind_Turbine_SCADA_tratado_consolidado.csv` ou os arquivos `Turbina_T01_data.csv`, `Turbina_T06_data.csv`, `Turbina_T07_data.csv`, `Turbina_T11_data.csv`.
- Kaggle Wind Sites: `Location1.csv`, `Location2.csv`, `Location3.csv`, `Location4.csv`.
- NREL/IEA Reference Turbines: arquivos de referência como `NREL_Reference_5MW_126 (1).csv`, `IEA_Reference_10MW_198.csv`, `DTU_Reference_v1_10MW_178.csv`.
- Penmanshiel: uma pasta no padrão `datasets*-penmanshiel` com arquivos `Turbine_Data_Penmanshiel_*.csv` e `Status_Penmanshiel_*.csv`.

Se algum dataset não aparecer no menu lateral, o arquivo esperado provavelmente não está na pasta correta.

### 6. Iniciar o dashboard

```powershell
streamlit run wind_dashboard.py
```

O Streamlit normalmente abre o navegador sozinho. Se não abrir, acesse:

```text
http://localhost:8501
```

Se a porta 8501 já estiver ocupada:

```powershell
streamlit run wind_dashboard.py --server.port 8502
```

Para parar a aplicação, volte ao terminal e pressione `Ctrl+C`.

## Como adicionar um novo dataset

O dashboard não lê CSVs soltos automaticamente. Cada dataset precisa de um loader próprio para evitar comparação errada entre colunas com nomes parecidos, mas significados diferentes.

Passos básicos:

1. Abra `wind_dashboard_core/loaders.py`.
2. Crie uma função `load_meu_dataset(root)` que leia o CSV e renomeie as colunas para o padrão canônico.
3. Cadastre o dataset em `DATASET_SPECS`.
4. Adicione a função no dicionário `LOADERS`.

Algumas colunas canônicas importantes:

- `entity`: turbina, local ou curva de referência.
- `timestamp`: data e hora da medição.
- `wind_speed_ms`: velocidade do vento em m/s.
- `power_kw`: potência em kW.
- `power_norm`: potência normalizada.
- `wind_direction_deg`: direção do vento.
- `temperature_c`: temperatura.
- `status`, `status_code`, `status_message`: informações de status operacional, quando existirem.

## Merge compatível de CSVs

O dashboard tem uma aba chamada `Merge CSVs`. Ela tenta encontrar o par de datasets mais compatível sem usar IA. A lógica é baseada no schema canônico criado pelos loaders e em regras fixas de chaveamento.

Como usar:

1. Selecione dois ou mais datasets na barra lateral.
2. Abra a aba `Merge CSVs`.
3. Escolha se o merge deve usar o recorte filtrado atual ou os datasets completos.
4. Veja o ranking de pares sugeridos.
5. Selecione um par, revise as flags e avisos.
6. Clique em `Gerar merge`.
7. Confira a prévia e baixe o CSV gerado.

O CSV gerado inclui colunas de rastreio no começo:

- `_merge_flag`: indica se o merge é recomendado, exige cuidado ou foi bloqueado.
- `_merge_compatibility_flag`: resume a compatibilidade como `HIGH`, `MEDIUM` ou `LOW`.
- `_merge_key_flag`: mostra qual chave foi usada.
- `_merge_strategy`: mostra a estratégia aplicada.
- `_merge_score`: pontuação calculada pelo dashboard.
- `_merge_presence`: indica se a chave veio dos dois datasets ou só de um lado, quando o tipo de join permite isso.

Flags principais:

- `MERGE_OK`: o dashboard encontrou uma chave boa o suficiente para gerar o merge.
- `MERGE_CAUTION`: o merge é possível, mas envolve agregação ou uma chave menos direta.
- `MERGE_BLOCKED`: o dashboard não encontrou uma chave determinística segura.

Flags de chave:

- `KEY_ENTITY_TIME`: usa `entity` mais uma janela de tempo.
- `KEY_TIME_WINDOW`: agrega por janela de tempo, sem entidade compartilhada.
- `KEY_WIND_BIN`: agrega por bins de velocidade do vento, normalmente a cada `0.5 m/s`.
- `KEY_METADATA`: usa metadados como `site`, `turbine_model`, `year`, `rated_power_kw` ou `rotor_diameter_m`.
- `KEY_NONE`: nenhuma chave segura foi encontrada.

Regras usadas:

- Se os datasets tiverem `timestamp` e entidades compatíveis, o dashboard tenta merge por entidade e janela temporal.
- Se tiverem `timestamp`, mas não compartilharem entidade, o dashboard agrega por janela temporal.
- Se um dataset for curva de referência sem timestamp, o dashboard compara por bins de vento.
- Se só houver metadados em comum, o merge é tratado como anotação, não como junção operacional linha a linha.
- Se a compatibilidade for baixa, o merge fica bloqueado.

Essa abordagem é determinística: o dashboard não tenta adivinhar significado de coluna. O significado entra pelo loader, e o merge usa apenas o schema canônico e regras explícitas.

## Outputs derivados de merge

O notebook `merge_development_merges.ipynb` gera os CSVs em `output/merged_datasets`.

Arquivos principais:

- `edp_gold_scada_failures_logs_metmast_empirical.csv`: base EDP enriquecida com falhas, logs, metmast e curva empírica. O output atual já agrega conflitos em `entity + timestamp` usando mediana numérica, equivalente à média nos conflitos duplicados de duas linhas, e remove colunas exatamente redundantes.
- `penmanshiel_scada_status_weak_labels.csv`: SCADA Penmanshiel com weak labels criados a partir dos status operacionais.
- `penmanshiel_status_failure_candidates.csv`: eventos de status Penmanshiel classificados como candidatos de falha. O output remove duplicatas exatas, `Comment` e `weak_failure_event`.
- `edp_penmanshiel_common_schema.csv`: concatenação padronizada entre EDP limpo e Penmanshiel, usando uma lista menor de colunas comuns para comparação/modelagem.
- `merge_artifacts_summary.csv`: resumo de linhas, colunas e contagens positivas dos labels.

Para regenerar esses arquivos, execute o notebook a partir da raiz do projeto. Depois confira `merge_artifacts_summary.csv` para validar se as contagens produzidas batem com o esperado.

## Problemas comuns

### `ModuleNotFoundError`

Ative o ambiente virtual e instale as dependências novamente:

```powershell
.\.venv\Scripts\Activate.ps1
pip install streamlit pandas numpy plotly scikit-learn
```

### Nenhum dataset aparece

Confira se os CSVs estão na raiz do projeto ou na pasta esperada. O dashboard só mostra datasets cadastrados em `DATASET_SPECS` e com arquivos encontrados.

### O dashboard fica lento

Reduza o valor de "Pontos por gráfico" na barra lateral. Alguns CSVs têm muitas linhas, e gráficos muito grandes podem pesar no navegador.
