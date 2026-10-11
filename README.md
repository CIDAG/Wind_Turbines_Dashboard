# Wind Dataset Dashboard

Dashboard em Streamlit para comparar datasets abertos de turbinas eólicas. A aplicação carrega os CSVs disponíveis em `data/`, padroniza as colunas principais e mostra cobertura de variáveis, compatibilidade entre bases, curvas de potência, séries temporais, status operacional e indicadores de qualidade.

## Estrutura principal

- `wind_dashboard.py`: aplicação Streamlit.
- `wind_dashboard_core/loaders.py`: cadastro dos datasets e conversão das colunas para um formato comum.
- `wind_dashboard_core/metrics.py`: métricas de cobertura, qualidade e compatibilidade.
- `wind_dashboard_core/charts.py`: gráficos Plotly usados no dashboard.
- `data/`: CSVs de entrada e tratados usados pelo dashboard, inclusive os arquivos Penmanshiel, diretamente nessa pasta.
- `notebooks/`: análises e geração dos CSVs derivados.
- `docs/`: notas técnicas adicionais.
- `output/`: resultados gerados e relatórios.

Os CSVs precisam ficar no mesmo layout esperado pelo projeto. O dashboard detecta os datasets registrados quando encontra os arquivos principais de cada um.

## Como rodar localmente

### 1. Clonar ou baixar o projeto

Se estiver usando Git:

```powershell
git clone https://github.com/CIDAG/Wind_Turbines_Dashboard.git
cd Wind_Turbines_Dashboard
```

### 2. Dados do Zenodo

O [registro Zenodo 21483443](https://zenodo.org/records/21483443) ([DOI 10.5281/zenodo.21483443](https://doi.org/10.5281/zenodo.21483443)) permanece com os CSVs individuais e o arquivo `datasets2020-penmanshiel.zip` separados. O repositório já inclui alguns CSVs tratados em `data/`; baixe os CSVs individuais necessários do Zenodo diretamente para `data/`, sem sobrescrever os existentes.

Para Penmanshiel, baixe [datasets2020-penmanshiel.zip](https://zenodo.org/records/21483443/files/datasets2020-penmanshiel.zip?download=1) para `data/` e, a partir da raiz do projeto, execute:

```powershell
python scripts/extract_penmanshiel.py data/datasets2020-penmanshiel.zip
```

O script extrai os CSVs Penmanshiel **diretamente em `data/`**, mesmo se houver uma pasta interna no ZIP. Ele não altera o arquivo baixado e não sobrescreve CSVs existentes. Uma extração comum pode preservar a pasta interna do ZIP; nesse caso, mova os CSVs dessa pasta para `data/` antes de executar o dashboard.

O resultado esperado é:

```text
Wind_Turbines_Dashboard/
├─ wind_dashboard.py
├─ data/
│  ├─ Wind_Turbine_SCADA_tratado_consolidado.csv
│  ├─ Location1.csv
│  ├─ ... demais CSVs de entrada ...
│  ├─ Turbine_Data_Penmanshiel_*.csv
│  └─ Status_Penmanshiel_*.csv
├─ scripts/extract_penmanshiel.py
├─ notebooks/
├─ docs/
└─ output/merged_datasets/
```

O dashboard identifica a raiz do projeto pela localização de `wind_dashboard.py` e procura esses arquivos diretamente em `data/`.

Para executar o dashboard, tenha em `data/` os CSVs EDP tratados, Kaggle, curvas de referência e Penmanshiel. O registro também contém EDP bruto, MetMast, logs e falhas para reproduzir os notebooks. Os CSVs já versionados em `output/merged_datasets/` usam Git LFS; para obter esses outputs prontos, execute:

```powershell
git lfs install
git lfs pull
```

Os outputs podem ser regenerados com `notebooks/merge_development_merges.ipynb` quando os dados do Zenodo estiverem disponíveis.

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
pip install "streamlit>=1.56" pandas numpy plotly scikit-learn
```

Para executar os notebooks com o mesmo ambiente virtual, instale também:

```powershell
pip install matplotlib seaborn scipy ipykernel nbformat nbclient
```

### 5. Conferir os arquivos de dados

O dashboard procura em `data/` os datasets cadastrados em `wind_dashboard_core/loaders.py`. Hoje os principais são:

- EDP SCADA: `Wind_Turbine_SCADA_tratado_consolidado.csv` ou os arquivos `Turbina_T01_data.csv`, `Turbina_T06_data.csv`, `Turbina_T07_data.csv`, `Turbina_T11_data.csv`.
- Kaggle Wind Sites: `Location1.csv`, `Location2.csv`, `Location3.csv`, `Location4.csv`.
- NREL/IEA Reference Turbines: arquivos de referência como `NREL_Reference_5MW_126 (1).csv`, `IEA_Reference_10MW_198.csv`, `DTU_Reference_v1_10MW_178.csv`.
- Penmanshiel: arquivos `Turbine_Data_Penmanshiel_*.csv` e `Status_Penmanshiel_*.csv` diretamente em `data/`.

Se algum dataset não aparecer no menu lateral, o arquivo esperado provavelmente não está na pasta correta.

O dashboard abre com uma base selecionada para limitar o uso inicial de memória. Selecione outras bases na barra lateral quando quiser compará-las.

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

As abas são calculadas quando abertas, usando o suporte do Streamlit 1.56 ou superior. As bases carregadas são compartilhadas pelo cache somente para leitura, e os filtros evitam cópias sucessivas dos dados. A contagem de duplicados considera todos os registros, mas divide o processamento em blocos para reduzir o uso de memória.

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

O notebook `notebooks/merge_development_merges.ipynb` gera os CSVs em `output/merged_datasets`.

Arquivos principais:

- `edp_gold_scada_failures_logs_metmast_empirical.csv`: base EDP enriquecida com falhas, logs, metmast e curva empírica. O output atual já agrega conflitos em `entity + timestamp` usando mediana numérica, equivalente à média nos conflitos duplicados de duas linhas, e remove colunas exatamente redundantes.
- `penmanshiel_scada_status_weak_labels.csv`: SCADA Penmanshiel com weak labels criados a partir dos status operacionais.
- `penmanshiel_status_failure_candidates.csv`: eventos de status Penmanshiel classificados como candidatos de falha. O output remove duplicatas exatas, `Comment` e `weak_failure_event`.
- `edp_penmanshiel_common_schema.csv`: concatenação padronizada entre EDP limpo e Penmanshiel, usando uma lista menor de colunas comuns para comparação/modelagem.
- `merge_artifacts_summary.csv`: resumo de linhas, colunas e contagens positivas dos labels.

Para regenerar esses arquivos, execute o notebook a partir da raiz do projeto ou de `notebooks/`. Depois confira `output/merged_datasets/merge_artifacts_summary.csv` para validar se as contagens produzidas batem com o esperado.

## Problemas comuns

### `ModuleNotFoundError`

Ative o ambiente virtual e instale as dependências novamente:

```powershell
.\.venv\Scripts\Activate.ps1
pip install streamlit pandas numpy plotly scikit-learn
```

### Nenhum dataset aparece

Confira se todos os CSVs, inclusive os arquivos Penmanshiel, estão diretamente em `data/`. O dashboard só mostra datasets cadastrados em `DATASET_SPECS` e com arquivos encontrados. Se os arquivos foram movidos enquanto o Streamlit estava aberto e a opção ainda não apareceu, pare a aplicação com `Ctrl+C` e execute `streamlit run wind_dashboard.py` novamente para recarregar o loader.

### O dashboard fica lento

Reduza o valor de "Pontos por gráfico" na barra lateral. Alguns CSVs têm muitas linhas, e gráficos muito grandes podem pesar no navegador.
