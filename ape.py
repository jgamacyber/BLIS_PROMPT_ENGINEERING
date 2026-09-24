"""
APE — Automatic Prompt Engineer.

Implementação do Algoritmo 1 de Zhou et al. (2023), *Large Language Models are
Human-Level Prompt Engineers* (ICLR 2023).

A tese: a instrução é um **programa**, e escrevê-la à mão é otimização manual de
um programa em linguagem natural. O APE automatiza isso tratando o problema como
busca de caixa-preta guiada por um LLM.

    ρ* = argmax_ρ f(ρ) = argmax_ρ E[f(ρ, Q, A)]

Algoritmo 1:

    1. Usar o LLM para propor instruções candidatas U = {ρ1, ..., ρm}
    2. Repetir até convergir:
       3.   Escolher um subconjunto aleatório de treino
       4.   Para cada ρ em U: avaliar o score no subconjunto
       7.   Filtrar os top-k% por score
       8.   Reamostrar variações semanticamente próximas das melhores
    9. Devolver a instrução de maior score

Duas peças vêm do LLM: **proposta** e **reamostragem**. A **pontuação** é
execução real — acurácia de execução (`f_exec`), a mesma métrica do artigo.

Geração em modo forward: o LLM vê pares de entrada/saída e completa a frase
"A instrução era ___". É o template da Figura 2 do artigo, adaptado ao português.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import SETTINGS, Settings, get_client
from dataset import Pergunta
from prompts import Exemplo
from runner import executar_estrategia_em_subconjunto

CAMINHO_INSTRUCAO = Path("resultados/instrucao_ape.txt")


# Template de geração forward (Figura 2, topo, do artigo).
TEMPLATE_PROPOSTA = """Dei a um amigo uma instrução e alguns exemplos de entrada.
O amigo leu a instrução e escreveu uma saída para cada uma das entradas.
Aqui estão os pares de entrada e saída:

{demonstracoes}

A instrução dada ao amigo era:"""

# Reamostragem por similaridade semântica (Seção 3.3 do artigo).
TEMPLATE_REAMOSTRAGEM = """Gere uma variação da instrução abaixo, mantendo o mesmo \
significado mas mudando a formulação.

A variação deve ser uma instrução clara, aplicável a qualquer pergunta, e não \
pode mencionar exemplos específicos.

Instrução: {instrucao}

Responda APENAS com a nova instrução, sem aspas e sem comentários."""


@dataclass
class Candidata:
    """Uma instrução candidata com seu score."""

    instrucao: str
    score: float = 0.0
    origem: str = "proposta"  # "proposta" | "reamostragem" | "semente"
    rodada: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Candidata {self.score:.1%} {self.instrucao[:50]!r}>"


@dataclass
class ResultadoAPE:
    """Saída completa da busca, com o histórico para auditoria."""

    melhor: Candidata | None = None
    historico: list[Candidata] = field(default_factory=list)
    rodadas: int = 0
    chamadas_llm: int = 0

    def imprimir(self, limite: int = 10) -> None:
        if not self.melhor:
            print("Nenhuma instrução candidata foi avaliada.")
            return

        print(f"\nRanking das candidatas ({len(self.historico)} avaliadas, "
              f"{self.rodadas} rodada(s)):")
        print("-" * 78)
        ordenadas = sorted(self.historico, key=lambda c: c.score, reverse=True)
        for i, c in enumerate(ordenadas[:limite], start=1):
            marca = " <-- melhor" if c is self.melhor else ""
            print(f"  {i:>2}. {c.score:>6.1%}  [r{c.rodada}/{c.origem[:5]}]{marca}")
            print(f"      {c.instrucao[:110]}")
        print("-" * 78)
        print(f"\nMelhor instrução ({self.melhor.score:.1%}):")
        print(f"  {self.melhor.instrucao}")


# --------------------------------------------------------------------------- #
# Proposta
# --------------------------------------------------------------------------- #


def _limpar_instrucao(texto: str) -> str:
    """Remove aspas, numeração e preâmbulos que o modelo costuma acrescentar."""
    t = (texto or "").strip()
    t = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", t)
    t = re.sub(r'^["\'“”«]\s*|\s*["\'“”»]$', "", t).strip()
    # Corta preâmbulos do tipo "A instrução era: ..."
    t = re.sub(
        r"^(a\s+)?instru(ç|c)ão\s*(dada|era|seria)?\s*[:\-]\s*",
        "", t, flags=re.IGNORECASE,
    ).strip()
    return t.strip(' "\'')


def propor_instrucoes(
    exemplos: list[Exemplo],
    n: int = 6,
    settings: Settings | None = None,
    n_demonstracoes: int = 5,
    verboso: bool = True,
) -> list[Candidata]:
    """
    Passo 1: o LLM propõe instruções candidatas a partir de demonstrações.

    Cada chamada usa um subconjunto diferente de demonstrações e temperatura
    alta, para que as candidatas sejam realmente diversas — uma busca só é útil
    se o espaço amostrado for variado.
    """
    settings = settings or SETTINGS
    client = get_client(settings)
    rng = random.Random(42)

    candidatas: list[Candidata] = []
    vistas: set[str] = set()

    for i in range(n):
        amostra = rng.sample(exemplos, min(n_demonstracoes, len(exemplos)))
        demonstracoes = "\n\n".join(
            f"Entrada: {e.pergunta}\nSaída: {e.resposta}" for e in amostra
        )

        try:
            resposta = client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": TEMPLATE_PROPOSTA.format(
                            demonstracoes=demonstracoes
                        ),
                    }
                ],
                temperature=settings.ape_temperatura,
                max_tokens=150,
            )
            instrucao = _limpar_instrucao(resposta.choices[0].message.content or "")
        except Exception as erro:  # noqa: BLE001
            print(f"  [ape] falha ao propor candidata {i + 1}: {erro}")
            continue

        chave = instrucao.lower()
        if not instrucao or chave in vistas:
            continue
        vistas.add(chave)
        candidatas.append(Candidata(instrucao=instrucao, origem="proposta"))

        if verboso:
            print(f"  candidata {len(candidatas)}: {instrucao[:90]}")

    return candidatas


def reamostrar(
    melhores: list[Candidata],
    n_por_candidata: int = 1,
    rodada: int = 1,
    settings: Settings | None = None,
    verboso: bool = True,
) -> list[Candidata]:
    """
    Passo 8: busca Monte Carlo — gera variações das melhores candidatas.

    É o que torna a busca iterativa em vez de um sorteio único: em vez de
    amostrar de novo do zero, explora a vizinhança semântica do que já funcionou.
    """
    settings = settings or SETTINGS
    client = get_client(settings)
    novas: list[Candidata] = []

    for candidata in melhores:
        for _ in range(n_por_candidata):
            try:
                resposta = client.chat.completions.create(
                    model=settings.chat_model,
                    messages=[
                        {
                            "role": "user",
                            "content": TEMPLATE_REAMOSTRAGEM.format(
                                instrucao=candidata.instrucao
                            ),
                        }
                    ],
                    temperature=settings.ape_temperatura,
                    max_tokens=150,
                )
                variacao = _limpar_instrucao(
                    resposta.choices[0].message.content or ""
                )
            except Exception as erro:  # noqa: BLE001
                print(f"  [ape] falha ao reamostrar: {erro}")
                continue

            if variacao and variacao.lower() != candidata.instrucao.lower():
                novas.append(
                    Candidata(
                        instrucao=variacao, origem="reamostragem", rodada=rodada
                    )
                )
                if verboso:
                    print(f"  variação: {variacao[:90]}")

    return novas


# --------------------------------------------------------------------------- #
# Pontuação
# --------------------------------------------------------------------------- #


def pontuar(
    candidata: Candidata,
    perguntas: list[Pergunta],
    exemplos: list[Exemplo],
    settings: Settings | None = None,
) -> float:
    """
    Função de score `f_exec`: acurácia de execução da instrução.

    Executa a instrução de verdade sobre um subconjunto e mede quantas respostas
    batem com o gabarito. É a mesma métrica do artigo — cara, porém honesta.
    """
    return executar_estrategia_em_subconjunto(
        perguntas, "ape", exemplos, settings, instrucao_ape=candidata.instrucao
    )


# --------------------------------------------------------------------------- #
# Busca completa
# --------------------------------------------------------------------------- #


def buscar(
    perguntas: list[Pergunta],
    exemplos: list[Exemplo],
    settings: Settings | None = None,
    n_candidatas: int | None = None,
    rodadas: int | None = None,
    top_k: int | None = None,
    tamanho_subconjunto: int = 8,
    incluir_semente: bool = True,
    verboso: bool = True,
) -> ResultadoAPE:
    """
    Algoritmo 1 completo: propor, pontuar, filtrar, reamostrar, repetir.

    `tamanho_subconjunto` controla o custo: pontuar cada candidata sobre TODAS
    as perguntas fica caro rápido. O artigo também amostra um subconjunto de
    treino a cada iteração (linha 3 do Algoritmo 1).
    """
    settings = settings or SETTINGS
    n_candidatas = n_candidatas or settings.ape_candidatos
    rodadas = rodadas or settings.ape_rodadas
    top_k = top_k or settings.ape_top_k

    resultado = ResultadoAPE(rodadas=rodadas)
    rng = random.Random(7)

    if verboso:
        print(f"\n[1/{rodadas + 1}] Propondo {n_candidatas} instruções candidatas")

    candidatas = propor_instrucoes(exemplos, n_candidatas, settings, verboso=verboso)
    resultado.chamadas_llm += n_candidatas

    # Sementes escritas à mão: a referência humana que o artigo compara.
    if incluir_semente:
        candidatas.append(
            Candidata(
                instrucao="Responda à pergunta de forma correta e o mais concisa "
                          "possível.",
                origem="semente",
            )
        )
        candidatas.append(
            Candidata(
                instrucao="Resolva a tarefa passo a passo, conferindo cada etapa, "
                          "e então dê a resposta final.",
                origem="semente",
            )
        )

    if not candidatas:
        print("  [ape] nenhuma candidata foi gerada.")
        return resultado

    for rodada in range(rodadas + 1):
        # Linha 3: subconjunto aleatório de treino, diferente a cada rodada.
        subconjunto = (
            rng.sample(perguntas, min(tamanho_subconjunto, len(perguntas)))
            if len(perguntas) > tamanho_subconjunto
            else list(perguntas)
        )

        if verboso:
            print(
                f"\n[{rodada + 1}/{rodadas + 1}] Pontuando {len(candidatas)} "
                f"candidata(s) em {len(subconjunto)} pergunta(s)"
            )

        for candidata in candidatas:
            candidata.score = pontuar(candidata, subconjunto, exemplos, settings)
            resultado.chamadas_llm += len(subconjunto)
            resultado.historico.append(candidata)
            if verboso:
                print(f"  {candidata.score:>6.1%}  {candidata.instrucao[:80]}")

        todas = sorted(resultado.historico, key=lambda c: c.score, reverse=True)
        resultado.melhor = todas[0]

        if rodada == rodadas:
            break

        # Linhas 7-8: filtra as melhores e reamostra na vizinhança.
        melhores = todas[:top_k]
        if verboso:
            print(f"\n  Reamostrando a partir das {len(melhores)} melhores")
        candidatas = reamostrar(melhores, 1, rodada + 1, settings, verboso)
        resultado.chamadas_llm += len(melhores)

        if not candidatas:
            if verboso:
                print("  nenhuma variação nova; encerrando a busca")
            break

    return resultado


# --------------------------------------------------------------------------- #
# Persistência
# --------------------------------------------------------------------------- #


def salvar_instrucao(instrucao: str, caminho: str | Path = CAMINHO_INSTRUCAO) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(instrucao.strip() + "\n", encoding="utf-8")
    return caminho


def carregar_instrucao(caminho: str | Path = CAMINHO_INSTRUCAO) -> str:
    caminho = Path(caminho)
    if not caminho.exists():
        return ""
    return caminho.read_text(encoding="utf-8").strip()
