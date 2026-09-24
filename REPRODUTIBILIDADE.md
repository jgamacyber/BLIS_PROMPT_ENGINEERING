# Documento de Reprodutibilidade

Roteiro para reproduzir todos os resultados deste módulo na sua máquina, com sua chave da OpenRouter. Cada comando traz o que esperar e quanto custa.

---

## 1. Ambiente

### Requisitos

- Python **3.10 ou superior** (o código usa `X | None`, sintaxe de união de tipos do 3.10)
- Uma chave da OpenRouter: https://openrouter.ai/keys — só para os comandos da seção 4 em diante

```bash
python --version      # deve mostrar 3.10+
```

Se a sua distribuição usa `python3`, troque em todos os comandos.

### Instalação

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows (PowerShell)

pip install -r requirements.txt
```

Só duas dependências: `openai` e `python-dotenv`. Não há framework de prompting, nem `pandas`, nem `scikit-learn` — tudo que o módulo faz está no código do repositório.

### Configuração

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Edite o `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...sua_chave...
MODEL=openai/gpt-4o-mini
TEMPERATURE=0.0
K_FEWSHOT=3
```

> O `.env` está no `.gitignore` e nunca deve ser commitado.

**Por que temperatura 0:** queremos medir o efeito do *prompt*, não a variação da amostragem. Para medir variância, veja a seção 6.

---

## 2. Validação sem custo

Rode as duas suítes antes de gastar qualquer crédito. Nenhuma faz chamada de rede.

### 2.1 Testes offline — a lógica

```bash
python testes_offline.py
```

**Esperado:** `Todos os testes passaram.` — **127 verificações**, cobrindo montagem de prompts, extração e verificação de respostas, carregamento e validação do conjunto, roteamento heurístico, agregação dos relatórios e utilidades do APE.

### 2.2 Testes com LLM simulado — as chamadas de API

```bash
python testes_mock.py
```

**Esperado:** `Todos os testes passaram.` — **62 verificações**. Substitui o cliente da OpenRouter por um dublê e confere que as chamadas são montadas no formato certo (modelo, mensagens, temperatura), que as respostas são interpretadas corretamente, e que falhas de rede e respostas malformadas não derrubam o experimento.

> Esta suíte é a que mais economiza crédito: um parser frágil ou um erro de montagem de chamada só apareceria em produção — aqui aparece de graça.

Se algo falhar nas duas seções, é bug ou incompatibilidade de ambiente. Não vale seguir para os comandos pagos.

---

## 3. Exploração sem custo

Estes comandos não chamam a API.

### 3.1 Listar estratégias e conjunto

```bash
python main.py listar
```

**Esperado:** as 9 estratégias com descrição e artigo, e `28 perguntas em 6 categorias (abstencao=3 | aritmetica=6 | classificacao=5 | extracao=5 | factual=4 | logica=5) | 10 demonstrações few-shot`.

### 3.2 Validar o conjunto

```bash
python main.py validar
```

**Esperado:** `Nenhum problema encontrado.` e a confirmação de que nenhuma pergunta avaliada aparece como demonstração few-shot.

Este é o teste de integridade mais importante do módulo. Se uma pergunta do conjunto de avaliação também fosse demonstração, o few-shot estaria consultando gabarito e toda a comparação seria inválida. O comando sai com código 1 se detectar vazamento.

### 3.3 Ler os prompts antes de enviá-los

```bash
python main.py ver-prompt cot_few_shot --pergunta-id arit_01
python main.py ver-prompt zero_shot --pergunta-id arit_01
python main.py ver-prompt estruturada --pergunta-id extr_04
```

**Esperado:** o prompt renderizado com `[SYSTEM]` / `[USER]` e a estimativa de tamanho. Para `cot_few_shot` em `arit_01`, cerca de **244 tokens**, com três demonstrações contendo a cadeia de raciocínio antes de cada resposta.

A maior parte dos problemas de engenharia de prompts é visível aqui, antes de gastar um token.

### 3.4 Roteamento

```bash
python main.py rotear "Roger tem 5 bolas e compra 2 latas com 3 cada. Quantas bolas tem?"
python main.py rotear "Extraia o valor total: total de R$ 347,90"
python main.py rotear "Qual é a capital da Austrália?"
```

**Esperado:** categoria, confiança, estratégia escolhida e as regex que serviram de evidência.

Acurácia medida do classificador heurístico no conjunto: **96,4% (27/28)**. O único erro é `fato_02` ("Quantos parâmetros tem o GPT-3? Responda apenas o número"), classificado como `extracao` por causa de "apenas o número" — e sai com **33% de confiança**, que é o gatilho da escalada no modo híbrido.

Para reproduzir essa medição:

```bash
python -c "
from roteamento import classificar_heuristico
from dataset import carregar_perguntas
p = carregar_perguntas()
ok = sum(classificar_heuristico(q.pergunta)[0] == q.categoria for q in p)
print(f'{ok}/{len(p)} = {ok/len(p):.1%}')"
```

---

## 4. A comparação principal (**custa**)

```bash
python main.py comparar
```

Roda 8 estratégias × 28 perguntas = **224 chamadas**. O comando pede confirmação antes de começar (use `--sim` para pular).

**Custo estimado:** com `gpt-4o-mini`, prompts de 30 a 570 tokens e respostas de ~45 tokens, algo entre **US$ 0,03 e US$ 0,08**.

**Saída:** três tabelas.

1. **Geral** — acurácia, acertos, tokens por item, latência e erros por estratégia.
2. **Por categoria** — onde cada estratégia ganha e onde perde. É a visão que importa: a média geral esconde que o CoT tende a ajudar em raciocínio e não fazer diferença em classificação.
3. **Custo-benefício** — pontos percentuais ganhos por token extra, tomando o zero-shot como referência.

Os resultados vão para `resultados/execucao.json`.

### Versões menores, mais baratas

```bash
# Só uma categoria
python main.py comparar --categoria aritmetica

# Só algumas estratégias
python main.py comparar --estrategias zero_shot_instrucao few_shot cot_few_shot

# Perguntas específicas
python main.py comparar --ids arit_01 arit_02 clas_01
```

### Relatórios

```bash
python main.py relatorio                 # reimprime as tabelas, sem custo
python main.py relatorio --divergencias  # perguntas em que as estratégias discordam
python main.py relatorio --tudo          # exporta CSV + Markdown + HTML
```

`--tudo` gera `resultados/comparacao.csv`, `.md` e `.html`. Reimprimir relatórios **não custa nada** — lê o JSON salvo.

As **divergências** são a parte mais informativa: as perguntas que algumas estratégias acertam e outras erram. É onde está a explicação de *por que* uma técnica ajuda.

---

## 5. Sobre os números que você vai obter

**Este documento não traz uma tabela de acurácias de referência, de propósito.**

Os resultados dependem do modelo, da versão dele e da data. Publicar "few-shot = 82%" como número esperado levaria você a achar que algo está errado quando obtivesse 76%. O que se reproduz aqui é o **método**, não os valores.

O que é razoável esperar, com base nos artigos:

- **Aritmética e lógica:** CoT (`cot_few_shot`, `cot_zero_shot`) tende a liderar. É a tese central de Wei et al.
- **Classificação:** few-shot tende a ir bem, porque demonstra o conjunto fechado de rótulos. CoT costuma não ajudar e custa mais.
- **Extração e JSON:** `estruturada` tende a liderar, porque o que está em jogo é o formato, não o raciocínio.
- **Abstenção:** estratégias que declaram explicitamente a regra "não invente" devem ir melhor. Zero-shot puro costuma ir mal.
- **`zero_shot` puro** deve ficar atrás em quase tudo — inclusive porque não pede o formato `RESPOSTA:`, o que prejudica a extração. Isso é proposital: é o piso da comparação.

**Dois avisos de interpretação:**

1. **Parte do ganho é formato, não raciocínio.** Estratégias que pedem `RESPOSTA: <x>` acertam mais também por serem mais fáceis de avaliar. Antes de concluir que uma técnica "raciocina melhor", leia as saídas brutas (`relatorio --divergencias`, ou a seção 6 do notebook).

2. **CoT é habilidade emergente de escala.** Wei et al. relatam que abaixo de ~100B parâmetros o CoT produz cadeias fluentes porém ilógicas e fica **pior** que o prompting padrão. Se o seu CoT não ajudar, registre — é um resultado alinhado com o artigo, não um bug.

---

## 6. Determinismo e variância

**Determinístico, reproduz exatamente:**

- Montagem de todos os prompts
- Extração e verificação de respostas
- Classificação heurística do roteador
- Agregação e relatórios a partir de um `execucao.json` salvo
- Amostragem do APE (sementes fixas: 42 para propostas, 7 para subconjuntos)

**Varia entre execuções:**

- A geração do modelo, mesmo com `temperature=0.0` — APIs não garantem reprodutibilidade bit-a-bit
- O APE por definição: propõe com `temperature=0.9` para diversificar candidatas
- Mudanças de versão do modelo por trás do mesmo identificador

### Medir a variância

```bash
# no .env: TEMPERATURE=0.7
python main.py comparar --estrategias cot_zero_shot --repeticoes 3 --categoria aritmetica
```

Isso roda cada pergunta 3 vezes. Se a mesma estratégia oscila entre acerto e erro na mesma pergunta, diferenças pequenas entre estratégias são ruído.

**Para um relatório, rode 3 vezes e reporte média e desvio, não um número único.**

---

## 7. APE (**custa mais**)

```bash
python main.py ape
```

**Padrão:** 6 candidatas + 2 sementes, 2 rodadas de reamostragem, pontuação em 8 perguntas por rodada.

**Custo:** o comando imprime a estimativa antes e pede confirmação. Com os padrões, cerca de **200 a 250 chamadas**, ou **US$ 0,03 a 0,06**.

Versão mais barata:

```bash
python main.py ape --candidatas 4 --rodadas 1 --subconjunto 5
```

**Esperado:** as candidatas propostas, o score de cada uma, as variações reamostradas e o ranking final. A melhor instrução vai para `resultados/instrucao_ape.txt`.

Depois, compare a instrução descoberta com as escritas à mão:

```bash
python main.py comparar --estrategias ape zero_shot_instrucao few_shot cot_few_shot
```

> Se a instrução do APE **não** superar as escritas à mão, isso também é um resultado. O artigo reporta desempenho de nível humano em 24/24 tarefas de indução de instrução, mas com um orçamento de busca muito maior do que o usado aqui.

---

## 8. Roteamento aprendido

```bash
python main.py comparar          # precisa ter rodado antes
python main.py aprender-rotas    # não custa nada: lê o JSON salvo
```

Deriva a tabela categoria → estratégia dos resultados **medidos**, em vez de supor. Salva em `resultados/tabela_roteamento.json`.

`--criterio custo` (padrão) desempata pelo menor gasto em tokens; `--criterio acuracia` ignora o custo.

Depois, responda usando o roteador:

```bash
python main.py responder "Quantas maçãs sobram se eu usar 20 de 23 e comprar 6?"
python main.py responder "Classifique o sentimento: o produto quebrou em dois dias"
```

**Esperado:** a decisão de roteamento (categoria, confiança, estratégia) seguida da resposta. Cada pergunta usa a estratégia que se mostrou melhor para o tipo dela.

**Custo:** 1 chamada por pergunta (2 se usar `--metodo llm`).

---

## 9. Resumo de custos

Com `openai/gpt-4o-mini`, em setembro de 2026. Confira os preços atuais em https://openrouter.ai/models.

| Comando | Chamadas | Custo aprox. |
|---|---|---|
| `testes_offline.py` | 0 | **US$ 0** |
| `testes_mock.py` | 0 | **US$ 0** |
| `main.py listar` / `validar` / `ver-prompt` / `rotear` | 0 | **US$ 0** |
| `main.py relatorio` (qualquer flag) | 0 | **US$ 0** |
| `main.py aprender-rotas` | 0 | **US$ 0** |
| `main.py comparar --categoria aritmetica` | 48 | ~US$ 0,01 |
| `main.py comparar` (completo) | 224 | **US$ 0,03 – 0,08** |
| `main.py ape` (padrão) | ~200–250 | **US$ 0,03 – 0,06** |
| `main.py ape --candidatas 4 --rodadas 1 --subconjunto 5` | ~60 | ~US$ 0,01 |
| `main.py responder` | 1–2 | <US$ 0,001 |

Reproduzir tudo, uma vez, fica abaixo de **US$ 0,20**.

---

## 10. Problemas comuns

**`OPENROUTER_API_KEY não encontrada`**
O `.env` não existe ou está na pasta errada. Ele precisa ficar ao lado de `main.py`.

**`Resultados não encontrados`**
`relatorio` e `aprender-rotas` leem `resultados/execucao.json`. Rode `python main.py comparar` antes.

**`'ape' pedida mas nenhuma instrução otimizada existe`**
Rode `python main.py ape` antes de incluir `ape` na comparação.

**Erro 401 ou 402 da OpenRouter**
Chave inválida ou sem créditos: https://openrouter.ai/credits

**Rate limit (429)**
Reduza o escopo com `--categoria` ou `--ids`, ou rode por partes.

**Acurácia baixa em tudo, inclusive nas estratégias boas**
Olhe as saídas brutas com `python main.py relatorio --divergencias`. Costuma ser problema de extração: o modelo respondeu certo mas não seguiu o formato `RESPOSTA:`. O extrator tem fallbacks, mas modelos muito verbosos ainda escapam.

**`SyntaxError` com `|` em anotações de tipo**
Python anterior ao 3.10. Atualize o interpretador.

**O notebook não encontra os módulos**
Ele precisa ser aberto a partir da pasta do módulo, onde estão os `.py` e `data/`. A primeira célula verifica isso e avisa.

---

## 11. Usar seu próprio conjunto

1. Substitua `data/perguntas.json`:

```json
[
  {
    "id": "meu_01",
    "categoria": "minha_categoria",
    "pergunta": "sua pergunta",
    "esperado": "resposta esperada",
    "verificacao": "exato"
  }
]
```

Tipos de `verificacao`: `exato`, `contem`, `numerico`, `regex`, `json` (aqui `esperado` é a lista de chaves obrigatórias) e `llm` (juiz por LLM, para perguntas abertas).

2. Substitua `data/exemplos_fewshot.json` por demonstrações do seu domínio. Preencha o campo `raciocinio` — sem ele, o `cot_few_shot` degrada para few-shot comum.

3. **Rode `python main.py validar`.** Ele checa vazamento, ids duplicados e categorias sem demonstração correspondente.

4. Se criar categorias novas, adicione-as a `TABELA_PADRAO` e a `PADROES` em `roteamento.py` — ou simplesmente rode `aprender-rotas` depois da comparação, que a tabela sai dos dados.

**Inclua perguntas sem resposta possível** (categoria `abstencao`). Sem elas, não dá para distinguir um sistema que sabe responder de um que sempre responde.

---

## 12. Checklist de reprodução

- [ ] Python 3.10+ confirmado
- [ ] `venv` criado e ativado na pasta do módulo
- [ ] `pip install -r requirements.txt` sem erros
- [ ] `.env` criado a partir do `.env.example`, com a chave preenchida
- [ ] `python testes_offline.py` → 127 verificações, todas passam
- [ ] `python testes_mock.py` → 62 verificações, todas passam
- [ ] `python main.py validar` → nenhum vazamento
- [ ] `python main.py ver-prompt cot_few_shot` → prompt com cadeias de raciocínio
- [ ] Roteador heurístico → 27/28 = 96,4%
- [ ] `python main.py comparar` → três tabelas, resultados salvos
- [ ] `python main.py relatorio --tudo` → CSV, Markdown e HTML gerados
- [ ] `python main.py relatorio --divergencias` → saídas lidas à mão
- [ ] `python main.py aprender-rotas` → tabela derivada dos dados
- [ ] Números anotados **com o nome e a data do modelo usado**
