"""
Carregamento do conjunto de perguntas e das demonstrações few-shot.

Duas coleções separadas, e isso é proposital: as demonstrações NUNCA podem
conter as perguntas avaliadas. Misturar as duas é vazamento de teste — o
few-shot passaria a ser consulta a gabarito, não aprendizado em contexto.
`validar()` checa exatamente isso.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from avaliacao import normalizar
from prompts import Exemplo

CAMINHO_PERGUNTAS = Path("data/perguntas.json")
CAMINHO_EXEMPLOS = Path("data/exemplos_fewshot.json")


@dataclass
class Pergunta:
    """Um item de avaliação, com gabarito e forma de verificação."""

    id: str
    categoria: str
    pergunta: str
    esperado: object
    verificacao: str = "exato"
    metadados: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #


def carregar_perguntas(caminho: str | Path = CAMINHO_PERGUNTAS) -> list[Pergunta]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Conjunto de perguntas não encontrado: {caminho}")

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    perguntas = []
    for i, d in enumerate(dados):
        faltando = {"pergunta", "esperado"} - set(d)
        if faltando:
            raise ValueError(f"Item {i} sem os campos obrigatórios: {faltando}")
        perguntas.append(
            Pergunta(
                id=d.get("id", f"item_{i:03d}"),
                categoria=d.get("categoria", "geral"),
                pergunta=d["pergunta"],
                esperado=d["esperado"],
                verificacao=d.get("verificacao", "exato"),
                metadados=d.get("metadados", {}),
            )
        )
    return perguntas


def carregar_exemplos(caminho: str | Path = CAMINHO_EXEMPLOS) -> list[Exemplo]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Exemplos few-shot não encontrados: {caminho}")

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return [
        Exemplo(
            pergunta=d["pergunta"],
            resposta=str(d["resposta"]),
            raciocinio=d.get("raciocinio", ""),
            categoria=d.get("categoria", "geral"),
        )
        for d in dados
    ]


# --------------------------------------------------------------------------- #
# Seleção
# --------------------------------------------------------------------------- #


def exemplos_para(
    categoria: str, exemplos: list[Exemplo], k: int = 3
) -> list[Exemplo]:
    """
    Escolhe as K demonstrações para uma pergunta.

    Prioriza exemplos da MESMA categoria — demonstrar aritmética antes de uma
    pergunta de classificação ensina o formato errado. Se faltarem exemplos da
    categoria, completa com outros para atingir K.
    """
    mesma = [e for e in exemplos if e.categoria == categoria]
    outras = [e for e in exemplos if e.categoria != categoria]
    return (mesma + outras)[:k]


def filtrar(perguntas: list[Pergunta], categorias: list[str] | None = None,
            ids: list[str] | None = None) -> list[Pergunta]:
    """Filtra o conjunto por categoria e/ou por id."""
    saida = perguntas
    if categorias:
        alvo = {c.lower() for c in categorias}
        saida = [p for p in saida if p.categoria.lower() in alvo]
    if ids:
        alvo_ids = set(ids)
        saida = [p for p in saida if p.id in alvo_ids]
    return saida


def categorias(perguntas: list[Pergunta]) -> list[str]:
    """Categorias presentes, em ordem de primeira aparição."""
    vistas: list[str] = []
    for p in perguntas:
        if p.categoria not in vistas:
            vistas.append(p.categoria)
    return vistas


# --------------------------------------------------------------------------- #
# Validação
# --------------------------------------------------------------------------- #


def validar(
    perguntas: list[Pergunta], exemplos: list[Exemplo]
) -> list[str]:
    """
    Procura problemas no conjunto. Devolve a lista de avisos (vazia = tudo bem).

    O mais importante é o vazamento: uma pergunta avaliada que também aparece
    como demonstração few-shot invalidaria a comparação inteira.
    """
    avisos: list[str] = []

    vistos: set[str] = set()
    for p in perguntas:
        if p.id in vistos:
            avisos.append(f"id duplicado: {p.id}")
        vistos.add(p.id)

    textos_exemplos = {normalizar(e.pergunta) for e in exemplos}
    for p in perguntas:
        if normalizar(p.pergunta) in textos_exemplos:
            avisos.append(
                f"VAZAMENTO: a pergunta '{p.id}' também é uma demonstração few-shot"
            )

    from avaliacao import VERIFICADORES

    for p in perguntas:
        if p.verificacao not in VERIFICADORES:
            avisos.append(
                f"'{p.id}' usa verificação desconhecida: '{p.verificacao}'"
            )
        if p.verificacao == "json" and not isinstance(p.esperado, list):
            avisos.append(
                f"'{p.id}' usa verificação json mas 'esperado' não é lista de chaves"
            )

    cats_perguntas = {p.categoria for p in perguntas}
    cats_exemplos = {e.categoria for e in exemplos}
    for c in sorted(cats_perguntas - cats_exemplos):
        avisos.append(
            f"a categoria '{c}' não tem demonstração few-shot correspondente"
        )

    sem_raciocinio = [e.pergunta[:40] for e in exemplos if not e.raciocinio]
    if sem_raciocinio:
        avisos.append(
            f"{len(sem_raciocinio)} exemplo(s) sem 'raciocinio': o CoT few-shot "
            f"degrada para few-shot comum neles"
        )

    return avisos


def resumo(perguntas: list[Pergunta], exemplos: list[Exemplo]) -> str:
    contagem: dict[str, int] = {}
    for p in perguntas:
        contagem[p.categoria] = contagem.get(p.categoria, 0) + 1
    detalhe = " | ".join(f"{c}={n}" for c, n in sorted(contagem.items()))
    return (
        f"{len(perguntas)} perguntas em {len(contagem)} categorias "
        f"({detalhe}) | {len(exemplos)} demonstrações few-shot"
    )
