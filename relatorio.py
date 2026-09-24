"""
Relatórios comparativos a partir de uma execução.

Três visões, em ordem de utilidade:

1. Geral — acurácia, custo em tokens e latência por estratégia.
2. Por categoria — onde cada estratégia ganha e onde perde. É a visão que
   importa: a média geral esconde que o CoT tende a ajudar em raciocínio e a
   não fazer diferença (ou atrapalhar) em classificação.
3. Divergências — perguntas em que as estratégias discordam entre si. São os
   itens que vale ler à mão.

Exporta em Markdown, CSV e HTML.
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass, field
from pathlib import Path

from runner import Execucao, Resultado

DIR_RESULTADOS = Path("resultados")


@dataclass
class Agregado:
    """Métricas de uma estratégia."""

    estrategia: str
    acertos: int = 0
    total: int = 0
    tokens_prompt: int = 0
    tokens_resposta: int = 0
    latencia: float = 0.0
    erros: int = 0
    por_categoria: dict = field(default_factory=dict)

    @property
    def acuracia(self) -> float:
        return self.acertos / self.total if self.total else 0.0

    @property
    def tokens_totais(self) -> int:
        return self.tokens_prompt + self.tokens_resposta

    @property
    def tokens_por_item(self) -> float:
        return self.tokens_totais / self.total if self.total else 0.0

    @property
    def latencia_media(self) -> float:
        return self.latencia / self.total if self.total else 0.0

    def acuracia_categoria(self, categoria: str) -> float:
        dados = self.por_categoria.get(categoria)
        if not dados or not dados["total"]:
            return 0.0
        return dados["acertos"] / dados["total"]


def agregar(execucao: Execucao) -> dict[str, Agregado]:
    """Consolida os resultados por estratégia e por categoria."""
    agregados: dict[str, Agregado] = {}

    for r in execucao.resultados:
        ag = agregados.setdefault(r.estrategia, Agregado(estrategia=r.estrategia))
        ag.total += 1
        ag.acertos += int(r.acertou)
        ag.tokens_prompt += r.tokens_prompt
        ag.tokens_resposta += r.tokens_resposta
        ag.latencia += r.latencia
        ag.erros += int(bool(r.erro))

        cat = ag.por_categoria.setdefault(r.categoria, {"acertos": 0, "total": 0})
        cat["total"] += 1
        cat["acertos"] += int(r.acertou)

    return agregados


# --------------------------------------------------------------------------- #
# Impressão no terminal
# --------------------------------------------------------------------------- #


def imprimir_geral(execucao: Execucao) -> None:
    agregados = agregar(execucao)
    if not agregados:
        print("Nenhum resultado para relatar.")
        return

    ordenados = sorted(agregados.values(), key=lambda a: a.acuracia, reverse=True)
    melhor = ordenados[0].acuracia

    print(f"\nComparação geral  —  modelo {execucao.modelo}, "
          f"temperatura {execucao.temperatura}")
    print("-" * 86)
    print(f"{'Estratégia':<22} {'Acurácia':>10} {'Acertos':>9} "
          f"{'Tok/item':>10} {'Latência':>10} {'Erros':>7}")
    print("-" * 86)

    for ag in ordenados:
        marca = "  <-- melhor" if ag.acuracia == melhor else ""
        print(
            f"{ag.estrategia:<22} {ag.acuracia:>9.1%} "
            f"{ag.acertos:>4}/{ag.total:<4} {ag.tokens_por_item:>10.0f} "
            f"{ag.latencia_media:>9.2f}s {ag.erros:>7}{marca}"
        )
    print("-" * 86)


def imprimir_por_categoria(execucao: Execucao) -> None:
    agregados = agregar(execucao)
    categorias = execucao.categorias()
    if not agregados or not categorias:
        return

    largura = max(12, max(len(c) for c in categorias) + 2)
    print("\nAcurácia por categoria")
    print("-" * (24 + largura * len(categorias)))
    cabecalho = f"{'Estratégia':<22}" + "".join(f"{c:>{largura}}" for c in categorias)
    print(cabecalho)
    print("-" * (24 + largura * len(categorias)))

    ordenados = sorted(agregados.values(), key=lambda a: a.acuracia, reverse=True)
    for ag in ordenados:
        linha = f"{ag.estrategia:<22}"
        for c in categorias:
            linha += f"{ag.acuracia_categoria(c):>{largura}.0%}"
        print(linha)
    print("-" * (24 + largura * len(categorias)))

    # Destaca a melhor estratégia de cada categoria — é o insumo do roteamento.
    # O desempate é pelo MENOR custo em tokens, o mesmo critério de
    # `aprender_tabela(criterio="custo")`, para que o que se lê aqui seja
    # exatamente o que a tabela de roteamento vai conter.
    print("\nMelhor estratégia por categoria (empate resolvido pelo menor custo):")
    for c in categorias:
        melhor = melhor_da_categoria(agregados, c)
        empatadas = sum(
            1 for a in agregados.values()
            if a.acuracia_categoria(c) == melhor.acuracia_categoria(c)
        )
        nota = f"  (empate entre {empatadas})" if empatadas > 1 else ""
        print(
            f"  {c:<16} {melhor.estrategia:<22} "
            f"{melhor.acuracia_categoria(c):>6.0%}{nota}"
        )


def melhor_da_categoria(agregados: dict[str, Agregado], categoria: str) -> Agregado:
    """
    Melhor estratégia em uma categoria: maior acurácia, desempatando pelo menor
    custo em tokens.

    Se duas estratégias acertam o mesmo, não há razão para pagar pela mais cara.
    Usado tanto pelos relatórios quanto pelo aprendizado da tabela de roteamento,
    para que os dois nunca discordem.
    """
    melhor_acuracia = max(a.acuracia_categoria(categoria) for a in agregados.values())
    empatadas = [
        a for a in agregados.values()
        if a.acuracia_categoria(categoria) == melhor_acuracia
    ]
    return min(empatadas, key=lambda a: a.tokens_por_item)


def imprimir_divergencias(execucao: Execucao, limite: int = 8) -> None:
    """Perguntas em que as estratégias discordam — os itens que valem leitura."""
    por_pergunta: dict[str, list[Resultado]] = {}
    for r in execucao.resultados:
        por_pergunta.setdefault(r.pergunta_id, []).append(r)

    divergentes = []
    for pid, resultados in por_pergunta.items():
        acertos = sum(r.acertou for r in resultados)
        if 0 < acertos < len(resultados):
            divergentes.append((pid, acertos, len(resultados), resultados))

    if not divergentes:
        print("\nNenhuma divergência: todas as estratégias concordaram em tudo.")
        return

    # Ordena pelas mais equilibradas (mais informativas).
    divergentes.sort(key=lambda t: abs(t[1] / t[2] - 0.5))

    print(f"\nPerguntas com divergência entre estratégias ({len(divergentes)} de "
          f"{len(por_pergunta)}):")
    print("-" * 86)
    for pid, acertos, total, resultados in divergentes[:limite]:
        categoria = resultados[0].categoria
        print(f"\n  {pid} ({categoria}) — {acertos}/{total} estratégias acertaram")
        print(f"  esperado: {resultados[0].esperado[:60]}")
        certas = [r.estrategia for r in resultados if r.acertou]
        erradas = [r.estrategia for r in resultados if not r.acertou]
        print(f"    acertaram: {', '.join(certas)}")
        print(f"    erraram:   {', '.join(erradas)}")
        for r in resultados:
            if not r.acertou and r.resposta_extraida:
                print(f"      {r.estrategia} respondeu: {r.resposta_extraida[:60]}")
                break
    print("-" * 86)


def imprimir_custo_beneficio(execucao: Execucao) -> None:
    """
    Acurácia contra custo — a decisão prática.

    Uma estratégia que acerta 5 pontos a mais gastando o triplo de tokens pode
    não valer a pena em produção. O ponto de referência é o zero-shot.
    """
    agregados = agregar(execucao)
    base = agregados.get("zero_shot_instrucao") or agregados.get("zero_shot")
    if not base or not base.tokens_por_item:
        return

    print("\nCusto-benefício (referência: "
          f"{base.estrategia}, {base.acuracia:.0%}, "
          f"{base.tokens_por_item:.0f} tokens/item)")
    print("-" * 78)
    print(f"{'Estratégia':<22} {'Δ acurácia':>12} {'custo relativo':>16} "
          f"{'pontos/token extra':>20}")
    print("-" * 78)

    for ag in sorted(agregados.values(), key=lambda a: a.acuracia, reverse=True):
        delta = (ag.acuracia - base.acuracia) * 100
        razao = ag.tokens_por_item / base.tokens_por_item
        extra = ag.tokens_por_item - base.tokens_por_item

        if ag.estrategia == base.estrategia:
            eficiencia = "—  (referência)"
        elif extra <= 0:
            eficiencia = "mais barata" if delta >= 0 else "pior e mais barata"
        else:
            eficiencia = f"{delta / extra * 100:>8.2f} p.p./100 tok"

        print(
            f"{ag.estrategia:<22} {delta:>+11.1f} {razao:>15.1f}x {eficiencia:>20}"
        )
    print("-" * 78)


# --------------------------------------------------------------------------- #
# Exportação
# --------------------------------------------------------------------------- #


def exportar_csv(execucao: Execucao, caminho: str | Path) -> Path:
    """Uma linha por resultado — para análise em planilha."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with caminho.open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow([
            "pergunta_id", "categoria", "estrategia", "repeticao", "acertou",
            "esperado", "resposta_extraida", "tokens_prompt", "tokens_resposta",
            "latencia_s", "erro",
        ])
        for r in execucao.resultados:
            escritor.writerow([
                r.pergunta_id, r.categoria, r.estrategia, r.repeticao,
                "sim" if r.acertou else "nao", r.esperado,
                (r.resposta_extraida or "").replace("\n", " ")[:300],
                r.tokens_prompt, r.tokens_resposta,
                f"{r.latencia:.3f}".replace(".", ","), r.erro[:150],
            ])
    return caminho


def exportar_markdown(execucao: Execucao, caminho: str | Path) -> Path:
    """Relatório em Markdown, pronto para colar no repositório."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    agregados = agregar(execucao)
    categorias = execucao.categorias()
    ordenados = sorted(agregados.values(), key=lambda a: a.acuracia, reverse=True)

    linhas = [
        "# Comparação de estratégias de prompt",
        "",
        f"- **Modelo:** `{execucao.modelo}`",
        f"- **Temperatura:** {execucao.temperatura}",
        f"- **K few-shot:** {execucao.k_fewshot}",
        f"- **Repetições:** {execucao.repeticoes}",
        f"- **Execução:** {execucao.timestamp}",
        "",
        "## Geral",
        "",
        "| Estratégia | Acurácia | Acertos | Tokens/item | Latência média |",
        "|---|---:|---:|---:|---:|",
    ]
    for ag in ordenados:
        linhas.append(
            f"| `{ag.estrategia}` | {ag.acuracia:.1%} | {ag.acertos}/{ag.total} "
            f"| {ag.tokens_por_item:.0f} | {ag.latencia_media:.2f}s |"
        )

    linhas += ["", "## Por categoria", "",
               "| Estratégia | " + " | ".join(categorias) + " |",
               "|---|" + "---:|" * len(categorias)]
    for ag in ordenados:
        celulas = " | ".join(f"{ag.acuracia_categoria(c):.0%}" for c in categorias)
        linhas.append(f"| `{ag.estrategia}` | {celulas} |")

    linhas += ["", "## Melhor estratégia por categoria", "",
               "| Categoria | Estratégia | Acurácia |", "|---|---|---:|"]
    for c in categorias:
        melhor = melhor_da_categoria(agregados, c)
        linhas.append(
            f"| {c} | `{melhor.estrategia}` | {melhor.acuracia_categoria(c):.0%} |"
        )

    if execucao.instrucao_ape:
        linhas += ["", "## Instrução descoberta pelo APE", "",
                   "```", execucao.instrucao_ape, "```"]

    linhas += ["", "---", "",
               "Gerado por `python main.py relatorio --markdown`."]

    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho


def exportar_html(execucao: Execucao, caminho: str | Path) -> Path:
    """Página autocontida com as tabelas, para abrir no navegador."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    agregados = agregar(execucao)
    categorias = execucao.categorias()
    ordenados = sorted(agregados.values(), key=lambda a: a.acuracia, reverse=True)

    def barra(valor: float) -> str:
        pct = round(valor * 100)
        return (
            f'<div class="barra"><div class="preenche" style="width:{pct}%"></div>'
            f'<span>{pct}%</span></div>'
        )

    linhas_geral = "".join(
        f"<tr><td><code>{html.escape(a.estrategia)}</code></td>"
        f"<td>{barra(a.acuracia)}</td>"
        f"<td class='num'>{a.acertos}/{a.total}</td>"
        f"<td class='num'>{a.tokens_por_item:.0f}</td>"
        f"<td class='num'>{a.latencia_media:.2f}s</td></tr>"
        for a in ordenados
    )

    cabecalho_cat = "".join(f"<th>{html.escape(c)}</th>" for c in categorias)
    linhas_cat = "".join(
        f"<tr><td><code>{html.escape(a.estrategia)}</code></td>"
        + "".join(
            f"<td>{barra(a.acuracia_categoria(c))}</td>" for c in categorias
        )
        + "</tr>"
        for a in ordenados
    )

    documento = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comparação de estratégias de prompt</title>
<style>
  :root {{
    --fundo: #ffffff; --texto: #1a1c1f; --suave: #5b6169;
    --linha: #e3e6ea; --destaque: #2563eb; --barra-fundo: #eef1f5;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --fundo: #14161a; --texto: #e8eaed; --suave: #9aa1ab;
      --linha: #2a2e35; --destaque: #5b8def; --barra-fundo: #22262d;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 16px; background: var(--fundo); color: var(--texto);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .envelope {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.02em; }}
  h2 {{ font-size: 1.15rem; margin: 36px 0 12px; }}
  .meta {{ color: var(--suave); font-size: 0.87rem; margin-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--linha); }}
  th {{ font-weight: 600; color: var(--suave); font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.04em; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  code {{ font-size: 0.86em; background: var(--barra-fundo);
          padding: 2px 5px; border-radius: 4px; }}
  .barra {{ position: relative; background: var(--barra-fundo);
            border-radius: 4px; height: 20px; min-width: 90px; }}
  .preenche {{ background: var(--destaque); height: 100%; border-radius: 4px; }}
  .barra span {{ position: absolute; inset: 0; display: flex; align-items: center;
                 justify-content: center; font-size: 0.76rem; font-weight: 600;
                 font-variant-numeric: tabular-nums; }}
  .rolagem {{ overflow-x: auto; }}
  footer {{ margin-top: 40px; color: var(--suave); font-size: 0.82rem; }}
</style>
</head>
<body>
<div class="envelope">
  <h1>Comparação de estratégias de prompt</h1>
  <p class="meta">
    Modelo <code>{html.escape(execucao.modelo)}</code> &middot;
    temperatura {execucao.temperatura} &middot;
    K={execucao.k_fewshot} &middot;
    {execucao.repeticoes} repetição(ões) &middot;
    {html.escape(execucao.timestamp)}
  </p>

  <h2>Geral</h2>
  <div class="rolagem">
  <table>
    <thead><tr><th>Estratégia</th><th>Acurácia</th><th>Acertos</th>
    <th>Tokens/item</th><th>Latência</th></tr></thead>
    <tbody>{linhas_geral}</tbody>
  </table>
  </div>

  <h2>Por categoria</h2>
  <div class="rolagem">
  <table>
    <thead><tr><th>Estratégia</th>{cabecalho_cat}</tr></thead>
    <tbody>{linhas_cat}</tbody>
  </table>
  </div>

  <footer>Gerado por <code>python main.py relatorio --html</code> —
  BLIS, módulo 02: Engenharia de Prompts.</footer>
</div>
</body>
</html>"""

    caminho.write_text(documento, encoding="utf-8")
    return caminho


def exportar_tudo(execucao: Execucao, diretorio: str | Path = DIR_RESULTADOS) -> list[Path]:
    diretorio = Path(diretorio)
    return [
        exportar_csv(execucao, diretorio / "comparacao.csv"),
        exportar_markdown(execucao, diretorio / "comparacao.md"),
        exportar_html(execucao, diretorio / "comparacao.html"),
    ]
