# BLIS — Módulo 02: Engenharia de Prompts

Script e notebook que recebem um conjunto de perguntas, rodam **versões diferentes de prompt** --> zero-shot, few-shot, com persona, com instrução estruturada, chain-of-thought <-- e **comparam as saídas**. Inclui roteamento de prompts e otimização automática de instrução.

Tudo implementado do zero, sem frameworks de prompting. As estratégias são fundamentadas nos artigos, não em folclore de internet — e cada uma é **medida** em vez de assumida.

## Artigos de referência

| Artigo | O que aparece no código |
|---|---|
| Vaswani et al. (2017), *Attention Is All You Need* | O Transformer e a janela de contexto — por que o tamanho e a posição do prompt importam |
| Brown et al. (2020), *Language Models are Few-Shot Learners* | As definições de zero-shot, one-shot e few-shot; aprendizado em contexto sem atualizar pesos |
| Wei et al. (2022), *Chain-of-Thought Prompting* | Demonstrações como triplas ⟨entrada, cadeia de pensamento, saída⟩ |
| Zhou et al. (2023), *LLMs are Human-Level Prompt Engineers* (APE) | Algoritmo 1: propor, pontuar, filtrar, reamostrar instruções |

## As estratégias comparadas

| Estratégia | O que faz | Artigo |
|---|---|---|
| `zero_shot` | Só a pergunta, sem instrução nem exemplos | Brown et al. |
| `zero_shot_instrucao` | Instrução em linguagem natural, sem demonstrações | Brown et al. |
| `one_shot` | Instrução + 1 demonstração | Brown et al. |
| `few_shot` | Instrução + K demonstrações | Brown et al. |
| `persona` | Papel atribuído ao modelo antes da tarefa | — |
| `estruturada` | Papel, tarefa, regras e formato, com delimitadores | — |
| `cot_zero_shot` | Pede raciocínio passo a passo, sem demonstrações | Kojima et al.; otimizável por APE |
| `cot_few_shot` | Demonstrações com cadeia de raciocínio | Wei et al. |
| `ape` | Instrução descoberta automaticamente por busca | Zhou et al. |

`persona` e `estruturada` não vêm de nenhum dos artigos — são práticas consolidadas entre praticantes. Estão aqui exatamente para serem confrontadas com as técnicas que têm respaldo empírico.

## Início rápido

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
cp .env.example .env               # preencha OPENROUTER_API_KEY

# Sem gastar nada:
python testes_offline.py                    # 127 verificações
python testes_mock.py                       # 62 verificações
python main.py listar                       # estratégias e conjunto
python main.py validar                      # checa vazamento
python main.py ver-prompt cot_few_shot      # lê o prompt antes de enviá-lo
python main.py rotear "Quantas maçãs sobraram?"

# Com a chave:
python main.py comparar                     # a matriz completa
python main.py relatorio --tudo             # CSV + Markdown + HTML
python main.py aprender-rotas               # deriva a tabela do que foi medido
python main.py ape                          # otimiza a instrução
python main.py responder "sua pergunta"     # responde com roteamento
```

Há também um `notebook.ipynb` para exploração interativa — o enunciado pedia "script ou notebook"; estão os dois, compartilhando os mesmos módulos.

## O conjunto de avaliação

28 perguntas em 6 categorias, com gabarito e forma de verificação:

| Categoria | Itens | O que testa |
|---|---|---|
| `aritmetica` | 6 | Raciocínio multi-passo — onde o CoT deve brilhar |
| `logica` | 5 | Dedução, ordenação, rastreio de estado |
| `classificacao` | 5 | Rótulo de um conjunto fechado |
| `extracao` | 5 | Retirar um dado de um texto; produzir JSON válido |
| `factual` | 4 | Conhecimento direto |
| `abstencao` | 3 | **Perguntas sem resposta possível** |

A categoria `abstencao` é a mais importante. Sem perguntas impossíveis no conjunto, não dá para distinguir um sistema que sabe responder de um que sempre responde.

**A verificação é determinística por padrão** — exata, numérica, regex ou validação de JSON. Não usa LLM como juiz na métrica principal: é grátis, é reprodutível, e um juiz LLM injetaria o próprio viés naquilo que estamos medindo. Há um juiz opcional para perguntas abertas, marcado como tal.

## Roteamento de prompts

Se nenhuma estratégia vence em tudo, usar uma só para todas as perguntas é deixar acurácia (ou dinheiro) na mesa. O roteador classifica a pergunta e escolhe a estratégia:

- **Classificador heurístico** — regex ponderada, grátis e determinístico. **96,4% de acerto** (27/28) no conjunto.
- **Classificador por LLM** — custa uma chamada, acerta mais em casos ambíguos.
- **Modo híbrido** — usa a heurística e só escala ao LLM quando a confiança é baixa.

O único erro da heurística (`fato_02`) sai com **33% de confiança** — ou seja, ela sinaliza a própria dúvida, que é o gatilho da escalada. Os pesos da regex não foram ajustados para zerar esse erro de propósito: ajustar a heurística até acertar o conjunto de teste seria sobreajuste, não melhoria.

A tabela de roteamento pode ser **aprendida dos resultados medidos** (`python main.py aprender-rotas`), em vez de escrita por intuição. Empates são resolvidos pelo menor custo em tokens: se duas estratégias acertam o mesmo, não há razão para pagar pela mais cara.

## APE — otimização automática da instrução

Implementa o Algoritmo 1 de Zhou et al.:

1. O LLM propõe instruções candidatas a partir de demonstrações (geração *forward*: "A instrução dada ao amigo era: ___")
2. Cada candidata é **executada de verdade** e pontuada pela acurácia (`f_exec`)
3. As melhores são filtradas
4. Variações semanticamente próximas são reamostradas (busca Monte Carlo)
5. Repete; devolve a de maior score

Duas instruções escritas à mão entram como **sementes**, para que a busca tenha uma referência humana a superar — é a comparação que o artigo faz.

## O que observar

1. **Chain-of-thought é habilidade emergente de escala.** Wei et al. observaram que modelos pequenos produzem cadeias "fluentes porém ilógicas" e ficam **piores** que o prompting padrão. Se o seu CoT não ajudar, isso é um resultado, não um bug.
2. **Parte do ganho é formato, não raciocínio.** Estratégias que pedem `RESPOSTA: <x>` acertam mais também porque ficam mais fáceis de avaliar. Leia as saídas brutas antes de concluir que uma técnica "raciocina melhor".
3. **Custo importa.** A tabela de custo-benefício mostra pontos percentuais ganhos por token extra. CoT few-shot costuma ser a mais cara; nem sempre compensa.
4. **Persona é a mais popular e a menos sustentada.** Nenhum dos quatro artigos a avalia. Veja o que acontece no seu experimento.
5. **Uma execução não é evidência.** Com `--repeticoes 3` e temperatura > 0 dá para ver quanto do resultado é ruído.

## Reprodução

Roteiro completo com custos e saídas esperadas: **[REPRODUTIBILIDADE.md](./REPRODUTIBILIDADE.md)**.

## Licença

MIT — ver [`LICENSE`](./LICENSE).

## Referências

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NIPS.
2. Brown, T. B. et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS.
3. Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
4. Zhou, Y. et al. (2023). *Large Language Models are Human-Level Prompt Engineers.* ICLR.
5. Kojima, T. et al. (2022). *Large Language Models are Zero-Shot Reasoners.* NeurIPS. — citado pelo APE; origem do CoT zero-shot.

Os PDFs não são versionados por questão de direitos autorais.
