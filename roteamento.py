"""
Roteamento de prompts: escolher a estratégia conforme o tipo da pergunta.

A premissa vem do próprio experimento: nenhuma estratégia vence em tudo. CoT
tende a ajudar em raciocínio e custar caro sem ganho em classificação; few-shot
disciplina o formato; zero-shot é o mais barato. Se isso é verdade, usar uma
única estratégia para todas as perguntas é deixar desempenho (ou dinheiro) na
mesa.

O roteador tem duas peças:

1. Um **classificador** que identifica a categoria da pergunta.
   - `heuristico`: regex e palavras-chave. Grátis, determinístico, e falha em
     casos ambíguos.
   - `llm`: uma chamada ao modelo. Custa e acerta mais.

2. Uma **tabela de roteamento** categoria → estratégia.
   - Pode ser escrita à mão (padrão razoável abaixo).
   - Ou **aprendida** dos resultados de `main.py comparar`, o que torna a
     decisão baseada em medição e não em intuição. É o modo recomendado.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from avaliacao import remover_acentos
from config import SETTINGS, Settings, get_client

CAMINHO_TABELA = Path("resultados/tabela_roteamento.json")


# --------------------------------------------------------------------------- #
# Tabela padrão (hipótese inicial, a ser substituída pela medição)
# --------------------------------------------------------------------------- #

TABELA_PADRAO: dict[str, str] = {
    "aritmetica": "cot_few_shot",     # raciocínio multi-passo: CoT (Wei et al.)
    "logica": "cot_zero_shot",        # raciocínio, mas sem precisar de exemplos
    "classificacao": "few_shot",      # rótulo fixo: demonstrar o conjunto de saídas
    "extracao": "estruturada",        # o que importa é o formato da saída
    "factual": "zero_shot_instrucao", # resposta direta: o mais barato serve
    "abstencao": "estruturada",       # a regra de "não invente" precisa ser explícita
    "geral": "zero_shot_instrucao",
}

ESTRATEGIA_PADRAO = "zero_shot_instrucao"


# --------------------------------------------------------------------------- #
# Classificador heurístico
# --------------------------------------------------------------------------- #

# Cada categoria tem padrões (regex, peso) aplicados ao texto sem acentos e
# minúsculo. Os pesos importam: um verbo imperativo que NOMEIA a tarefa
# ("extraia", "classifique") é um sinal muito mais forte do que a mera presença
# de números, que aparecem em quase todo tipo de pergunta. Sem isso, "Extraia o
# valor total de R$ 347,90" é classificada como aritmética.
PADROES: dict[str, list[tuple[str, float]]] = {
    "aritmetica": [
        (r"\bquant[oa]s?\b", 1.0),
        (r"\bqual\s+o\s+(pre|valor|total|custo)", 1.0),
        (r"\d+\s*[%+\-x*/]\s*\d+", 1.0),
        (r"\br\$\s*\d", 0.5),
        (r"\bdesconto\b", 1.0), (r"\baumento\b", 1.0),
        (r"\bsoma\b", 1.0), (r"\btotal\b", 0.5), (r"\bmedia\b", 1.0),
        (r"\bpor\s+cento\b", 1.0), (r"\d+/\d+", 1.0),
        (r"\bcada\b.*\d", 1.0), (r"\bquilometr", 1.0), (r"\blitros?\b", 1.0),
    ],
    "classificacao": [
        (r"\bclassifiqu", 3.0),      # verbo que nomeia a tarefa
        (r"\bcategoriz", 3.0),
        (r"\brotul", 2.0),
        (r"\bsentimento\b", 2.0),
        (r"\bpositivo\b.*\bnegativo\b", 2.0),
        (r"\bintencao\b", 1.5),
        (r"\bqual\s+categoria\b", 2.0),
        (r"\buma\s+destas\s+categorias\b", 2.0),
    ],
    "extracao": [
        (r"\bextra[ií]", 3.0),       # verbo que nomeia a tarefa
        (r"\bretire\b", 2.5),
        (r"\bidentifique\s+o\b", 2.0),
        (r"\bresponda\s+apenas\s+com\s+um\s+(objeto\s+)?json\b", 3.0),
        (r"\bjson\b", 2.0),
        (r"\bapenas\s+o\s+(valor|numero|nome|campo)", 2.0),
        (r"\bdo\s+texto\b", 1.5),
        (r"\bchaves\b", 1.0),
    ],
    "logica": [
        (r"\btodos?\s+os?\b.*\balguns?\b", 2.0),
        (r"\bnecessariamente\b", 2.0),
        (r"\bmais\s+(alt|baix|velh|nov)[oa]\b", 1.5),
        (r"\bconcatene\b", 2.5),
        (r"\bultimas?\s+letras?\b", 2.5),
        (r"\bdia\s+da\s+semana\b", 2.0),
        (r"\bse\s+.*\bentao\b", 1.5),
        (r"\bvira\s+a\s+moeda\b", 2.0),
        (r"\bordem\b", 0.5),
    ],
    "factual": [
        (r"\bqual\s+(e|foi)\s+a?\s*(capital|arquitetura|mecanismo)", 2.5),
        (r"\bquem\s+(propos|criou|escreveu|inventou)\b", 2.5),
        (r"\bem\s+que\s+ano\b", 2.0),
        (r"\bo\s+que\s+(e|significa)\b", 1.5),
        (r"\barquitetura\b", 1.0),
        (r"\bartigo\b", 0.5),
    ],
}

# Sinais de que a pergunta provavelmente não tem resposta conhecível.
PADROES_ABSTENCAO: list[tuple[str, float]] = [
    (r"\bficticia?\b", 3.0),
    (r"\bfaturamento\b.*\btrimestre\b", 2.5),
    (r"\bcor\s+favorita\b", 3.0),
    (r"\bquantos?\s+funcionarios\b", 2.0),
    (r"\bdepartamento\s+\w+\s+da\s+empresa\b", 2.0),
]


@dataclass
class Rota:
    """A decisão do roteador, com o rastro de como chegou nela."""

    pergunta: str
    categoria: str
    estrategia: str
    confianca: float = 0.0
    metodo: str = "heuristico"
    evidencias: list[str] = field(default_factory=list)

    def imprimir(self) -> None:
        print(f"  pergunta:   {self.pergunta[:64]}")
        print(f"  categoria:  {self.categoria} "
              f"({self.metodo}, confiança {self.confianca:.0%})")
        print(f"  estratégia: {self.estrategia}")
        if self.evidencias:
            print(f"  evidências: {', '.join(self.evidencias[:4])}")


def classificar_heuristico(pergunta: str) -> tuple[str, float, list[str]]:
    """
    Classifica por padrões textuais. Devolve (categoria, confiança, evidências).

    A confiança é a margem entre o primeiro e o segundo colocados — baixa
    confiança significa pergunta ambígua, e é o sinal para escalar ao
    classificador por LLM.
    """
    texto = remover_acentos(pergunta.lower())

    pontos: dict[str, float] = {}
    evidencias: dict[str, list[str]] = {}

    for categoria, padroes in PADROES.items():
        for padrao, peso in padroes:
            if re.search(padrao, texto):
                pontos[categoria] = pontos.get(categoria, 0) + peso
                evidencias.setdefault(categoria, []).append(padrao)

    for padrao, peso in PADROES_ABSTENCAO:
        if re.search(padrao, texto, re.IGNORECASE):
            pontos["abstencao"] = pontos.get("abstencao", 0) + peso
            evidencias.setdefault("abstencao", []).append(padrao)

    if not pontos:
        return "geral", 0.0, []

    ordenados = sorted(pontos.items(), key=lambda kv: kv[1], reverse=True)
    melhor, score = ordenados[0]
    segundo = ordenados[1][1] if len(ordenados) > 1 else 0.0

    total = sum(pontos.values())
    confianca = (score - segundo) / total if total else 0.0
    return melhor, min(1.0, max(0.0, confianca)), evidencias.get(melhor, [])


PROMPT_CLASSIFICADOR = """Classifique a pergunta abaixo em UMA destas categorias:

- aritmetica: exige cálculo numérico ou raciocínio matemático em vários passos
- logica: exige dedução, ordenação, rastreio de estado ou manipulação simbólica
- classificacao: pede para atribuir um rótulo de um conjunto fechado
- extracao: pede para retirar um dado específico de um texto dado, ou produzir JSON
- factual: pede um fato conhecido, de resposta direta
- abstencao: pede informação privada, inexistente ou impossível de saber

Pergunta: {pergunta}

Responda APENAS com um JSON:
{{"categoria": "<uma das acima>", "confianca": 0.0-1.0}}"""


def classificar_llm(
    pergunta: str, settings: Settings | None = None
) -> tuple[str, float]:
    """Classifica com uma chamada ao LLM. Cai no heurístico se falhar."""
    settings = settings or SETTINGS
    validas = set(PADROES) | {"abstencao", "geral"}

    try:
        client = get_client(settings)
        resposta = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "user", "content": PROMPT_CLASSIFICADOR.format(
                    pergunta=pergunta)}
            ],
            temperature=0.0,
            max_tokens=60,
        )
        texto = resposta.choices[0].message.content or ""
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            dados = json.loads(match.group(0))
            categoria = str(dados.get("categoria", "geral")).lower().strip()
            if categoria in validas:
                confianca = float(dados.get("confianca", 0.5))
                return categoria, min(1.0, max(0.0, confianca))
    except Exception as erro:  # noqa: BLE001
        print(f"  [roteador] classificação por LLM falhou ({erro}); usando heurística")

    categoria, confianca, _ = classificar_heuristico(pergunta)
    return categoria, confianca


# --------------------------------------------------------------------------- #
# Roteador
# --------------------------------------------------------------------------- #


class Roteador:
    """Escolhe a estratégia de prompt conforme a categoria da pergunta."""

    def __init__(
        self,
        tabela: dict[str, str] | None = None,
        metodo: str = "heuristico",
        limiar_escalada: float = 0.15,
        settings: Settings | None = None,
    ) -> None:
        self.tabela = dict(tabela or TABELA_PADRAO)
        self.metodo = metodo
        self.limiar_escalada = limiar_escalada
        self.settings = settings or SETTINGS

    def rotear(self, pergunta: str) -> Rota:
        evidencias: list[str] = []

        if self.metodo == "llm":
            categoria, confianca = classificar_llm(pergunta, self.settings)
            metodo = "llm"
        else:
            categoria, confianca, evidencias = classificar_heuristico(pergunta)
            metodo = "heuristico"

            # Escalada: se a heurística não está confiante, vale gastar uma
            # chamada para classificar melhor. Roteamento errado custa mais
            # caro do que a chamada de classificação.
            if self.metodo == "hibrido" and confianca < self.limiar_escalada:
                categoria, confianca = classificar_llm(pergunta, self.settings)
                metodo = "híbrido (escalou para LLM)"

        estrategia = self.tabela.get(categoria, ESTRATEGIA_PADRAO)
        return Rota(
            pergunta=pergunta,
            categoria=categoria,
            estrategia=estrategia,
            confianca=confianca,
            metodo=metodo,
            evidencias=evidencias,
        )

    def resumo(self) -> str:
        linhas = [f"  {c:<16} -> {e}" for c, e in sorted(self.tabela.items())]
        return "\n".join(linhas)


# --------------------------------------------------------------------------- #
# Aprender a tabela a partir dos resultados medidos
# --------------------------------------------------------------------------- #


def aprender_tabela(execucao, criterio: str = "acuracia") -> dict[str, str]:
    """
    Deriva a tabela de roteamento dos resultados de `main.py comparar`.

    É o fecho do experimento: em vez de supor qual estratégia serve para qual
    tipo de pergunta, usa-se o que foi medido.

    `criterio`:
      - "acuracia": a estratégia mais precisa na categoria.
      - "custo": entre as empatadas no topo, a mais barata em tokens. É o
        critério mais útil na prática — se duas estratégias acertam o mesmo,
        não há razão para pagar pela mais cara.
    """
    from relatorio import agregar, melhor_da_categoria

    agregados = agregar(execucao)
    if not agregados:
        return dict(TABELA_PADRAO)

    tabela: dict[str, str] = {}
    for categoria in execucao.categorias():
        if criterio == "custo":
            # Mesmo desempate dos relatórios: entre as empatadas no topo,
            # a mais barata em tokens.
            escolhida = melhor_da_categoria(agregados, categoria)
        else:
            escolhida = max(
                agregados.values(), key=lambda ag: ag.acuracia_categoria(categoria)
            )
        tabela[categoria] = escolhida.estrategia

    tabela.setdefault("geral", ESTRATEGIA_PADRAO)
    return tabela


def salvar_tabela(tabela: dict[str, str], caminho: str | Path = CAMINHO_TABELA) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(tabela, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return caminho


def carregar_tabela(caminho: str | Path = CAMINHO_TABELA) -> dict[str, str]:
    """Carrega a tabela aprendida; cai na padrão se ainda não existir."""
    caminho = Path(caminho)
    if not caminho.exists():
        return dict(TABELA_PADRAO)
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(TABELA_PADRAO)
