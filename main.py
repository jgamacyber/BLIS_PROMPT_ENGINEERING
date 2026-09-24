"""
BLIS — Módulo 02: Engenharia de Prompts
CLI para comparar estratégias de prompt, rotear e otimizar instruções.

Uso:
    python main.py listar                    # estratégias e conjunto (sem API)
    python main.py ver-prompt few_shot       # inspeciona um prompt (sem API)
    python main.py validar                   # checa vazamento no conjunto
    python main.py comparar                  # roda a matriz completa
    python main.py relatorio                 # relatórios da última execução
    python main.py rotear "sua pergunta"     # mostra a estratégia escolhida
    python main.py aprender-rotas            # deriva a tabela dos resultados
    python main.py ape                       # otimiza a instrução (APE)
    python main.py responder "pergunta"      # responde usando o roteador
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ape as mod_ape
import relatorio as mod_relatorio
from config import SETTINGS, resumo_config
from dataset import (
    carregar_exemplos,
    carregar_perguntas,
    categorias,
    exemplos_para,
    filtrar,
    resumo,
    validar,
)
from prompts import ESTRATEGIAS, PADRAO, obter, tamanho_aproximado, texto_do_prompt
from roteamento import (
    Roteador,
    aprender_tabela,
    carregar_tabela,
    salvar_tabela,
)
from runner import CAMINHO_RESULTADOS, Execucao, executar, executar_uma

LARGURA = 74


def banner(titulo: str) -> None:
    print("\n" + "=" * LARGURA)
    print(f"  {titulo}")
    print("=" * LARGURA)
    print(f"  {resumo_config()}")


def _carregar_dados(args) -> tuple[list, list]:
    perguntas = carregar_perguntas()
    exemplos = carregar_exemplos()
    perguntas = filtrar(
        perguntas,
        categorias=getattr(args, "categoria", None),
        ids=getattr(args, "ids", None),
    )
    if not perguntas:
        print("Nenhuma pergunta após os filtros.")
        sys.exit(1)
    return perguntas, exemplos


# --------------------------------------------------------------------------- #
# Comandos sem API
# --------------------------------------------------------------------------- #


def cmd_listar(args) -> None:
    banner("ESTRATÉGIAS E CONJUNTO DE AVALIAÇÃO")

    print("\nEstratégias disponíveis:")
    print("-" * LARGURA)
    for nome, e in ESTRATEGIAS.items():
        marca = "*" if nome in PADRAO else " "
        exemplos_marca = "usa exemplos" if e.usa_exemplos else ""
        print(f" {marca} {nome:<22} {e.descricao}")
        print(f"   {'':<22} {e.artigo}  {exemplos_marca}")
    print("-" * LARGURA)
    print(" * = incluída na comparação padrão")

    perguntas = carregar_perguntas()
    exemplos = carregar_exemplos()
    print(f"\nConjunto: {resumo(perguntas, exemplos)}")


def cmd_ver_prompt(args) -> None:
    """Renderiza um prompt sem enviá-lo. Custo zero."""
    banner(f"PROMPT: {args.estrategia}")

    perguntas, exemplos = _carregar_dados(args)
    pergunta = next((p for p in perguntas if p.id == args.pergunta_id), perguntas[0])

    estrategia = obter(args.estrategia)
    demos = (
        exemplos_para(pergunta.categoria, exemplos, SETTINGS.k_fewshot)
        if estrategia.usa_exemplos
        else []
    )
    extra = {"k": SETTINGS.k_fewshot} if estrategia.usa_exemplos else {}
    if args.estrategia == "ape":
        extra["instrucao"] = mod_ape.carregar_instrucao()

    mensagens = estrategia.montar(pergunta.pergunta, demos, **extra)

    print(f"\n  pergunta: {pergunta.id} ({pergunta.categoria})")
    print(f"  gabarito: {pergunta.esperado}")
    print(f"  tamanho estimado: ~{tamanho_aproximado(mensagens)} tokens")
    print("\n" + "-" * LARGURA)
    print(texto_do_prompt(mensagens))
    print("-" * LARGURA)


def cmd_validar(args) -> None:
    banner("VALIDAÇÃO DO CONJUNTO")
    perguntas = carregar_perguntas()
    exemplos = carregar_exemplos()

    print(f"\n{resumo(perguntas, exemplos)}")
    avisos = validar(perguntas, exemplos)

    if not avisos:
        print("\nNenhum problema encontrado.")
        print("Em especial: nenhuma pergunta avaliada aparece como demonstração "
              "few-shot (sem vazamento).")
        return

    print(f"\n{len(avisos)} aviso(s):")
    for a in avisos:
        gravidade = "GRAVE" if "VAZAMENTO" in a else "aviso"
        print(f"  [{gravidade}] {a}")

    if any("VAZAMENTO" in a for a in avisos):
        sys.exit(1)


def cmd_rotear(args) -> None:
    banner("ROTEAMENTO DE PROMPT")
    tabela = carregar_tabela()
    roteador = Roteador(tabela, metodo=args.metodo)

    print("\nTabela de roteamento ativa:")
    print(roteador.resumo())

    print("\nDecisão:")
    roteador.rotear(args.pergunta).imprimir()


# --------------------------------------------------------------------------- #
# Comandos que usam a API
# --------------------------------------------------------------------------- #


def cmd_comparar(args) -> None:
    banner("COMPARAÇÃO DE ESTRATÉGIAS")
    perguntas, exemplos = _carregar_dados(args)

    estrategias = args.estrategias or list(PADRAO)
    instrucao_ape = ""
    if "ape" in estrategias:
        instrucao_ape = mod_ape.carregar_instrucao()
        if not instrucao_ape:
            print("  [aviso] 'ape' pedida mas nenhuma instrução otimizada existe.")
            print("          Rode `python main.py ape` antes. Removendo do conjunto.")
            estrategias = [e for e in estrategias if e != "ape"]

    avisos = validar(perguntas, exemplos)
    graves = [a for a in avisos if "VAZAMENTO" in a]
    if graves:
        print("\n  [GRAVE] vazamento detectado no conjunto:")
        for a in graves:
            print(f"    {a}")
        sys.exit(1)

    total = len(perguntas) * len(estrategias) * args.repeticoes
    print(f"\n  {resumo(perguntas, exemplos)}")
    print(f"  {len(estrategias)} estratégias x {len(perguntas)} perguntas "
          f"x {args.repeticoes} repetição(ões) = {total} chamadas")

    if not args.sim and total > 120:
        resposta = input(f"\n  Isso são {total} chamadas. Continuar? [s/N] ")
        if resposta.strip().lower() not in {"s", "sim", "y"}:
            print("  Cancelado.")
            return

    from dataclasses import replace
    settings = replace(SETTINGS, repeticoes=args.repeticoes)

    execucao = executar(
        perguntas, estrategias, exemplos, settings,
        repeticoes=args.repeticoes, instrucao_ape=instrucao_ape,
    )
    execucao.salvar(CAMINHO_RESULTADOS)

    mod_relatorio.imprimir_geral(execucao)
    mod_relatorio.imprimir_por_categoria(execucao)
    mod_relatorio.imprimir_custo_beneficio(execucao)

    print(f"\nResultados salvos em {CAMINHO_RESULTADOS}")
    print("Gere os relatórios completos com: python main.py relatorio --tudo")


def cmd_relatorio(args) -> None:
    banner("RELATÓRIOS")
    execucao = Execucao.carregar(CAMINHO_RESULTADOS)
    print(f"  execução de {execucao.timestamp} | {len(execucao.resultados)} resultados")

    mod_relatorio.imprimir_geral(execucao)
    mod_relatorio.imprimir_por_categoria(execucao)
    mod_relatorio.imprimir_custo_beneficio(execucao)

    if args.divergencias or args.tudo:
        mod_relatorio.imprimir_divergencias(execucao)

    if args.tudo:
        caminhos = mod_relatorio.exportar_tudo(execucao)
        print("\nArquivos gerados:")
        for c in caminhos:
            print(f"  {c}")
    else:
        if args.csv:
            print(f"\n{mod_relatorio.exportar_csv(execucao, 'resultados/comparacao.csv')}")
        if args.markdown:
            print(f"\n{mod_relatorio.exportar_markdown(execucao, 'resultados/comparacao.md')}")
        if args.html:
            print(f"\n{mod_relatorio.exportar_html(execucao, 'resultados/comparacao.html')}")


def cmd_aprender_rotas(args) -> None:
    banner("APRENDIZADO DA TABELA DE ROTEAMENTO")
    execucao = Execucao.carregar(CAMINHO_RESULTADOS)

    tabela = aprender_tabela(execucao, criterio=args.criterio)
    caminho = salvar_tabela(tabela)

    print(f"\n  derivada de {len(execucao.resultados)} resultados "
          f"(critério: {args.criterio})")
    print("\nTabela aprendida:")
    for categoria, estrategia in sorted(tabela.items()):
        print(f"  {categoria:<16} -> {estrategia}")
    print(f"\nSalva em {caminho}")
    print("\nA partir de agora `python main.py rotear` e `responder` usam esta tabela.")


def cmd_ape(args) -> None:
    banner("APE — OTIMIZAÇÃO AUTOMÁTICA DA INSTRUÇÃO")
    perguntas, exemplos = _carregar_dados(args)

    n = args.candidatas or SETTINGS.ape_candidatos
    rodadas = args.rodadas if args.rodadas is not None else SETTINGS.ape_rodadas
    subconjunto = min(args.subconjunto, len(perguntas))
    estimativa = n + 2 + (n + 2) * subconjunto * (rodadas + 1)

    print(f"\n  {n} candidatas + 2 sementes | {rodadas} rodada(s) de reamostragem")
    print(f"  pontuação em {subconjunto} pergunta(s) por rodada")
    print(f"  estimativa: ~{estimativa} chamadas ao LLM")

    if not args.sim:
        resposta = input("\n  Continuar? [s/N] ")
        if resposta.strip().lower() not in {"s", "sim", "y"}:
            print("  Cancelado.")
            return

    resultado = mod_ape.buscar(
        perguntas, exemplos, SETTINGS,
        n_candidatas=n, rodadas=rodadas, tamanho_subconjunto=subconjunto,
    )
    resultado.imprimir()

    if resultado.melhor:
        caminho = mod_ape.salvar_instrucao(resultado.melhor.instrucao)
        print(f"\nInstrução salva em {caminho}")
        print("Use com: python main.py comparar --estrategias ape few_shot cot_few_shot")
        print(f"\nChamadas ao LLM efetuadas: {resultado.chamadas_llm}")


def cmd_responder(args) -> None:
    banner("RESPOSTA COM ROTEAMENTO")
    _, exemplos = _carregar_dados(args)

    roteador = Roteador(carregar_tabela(), metodo=args.metodo)
    rota = roteador.rotear(args.pergunta)

    print("\nRoteamento:")
    rota.imprimir()

    from dataset import Pergunta

    pergunta = Pergunta(
        id="adhoc", categoria=rota.categoria, pergunta=args.pergunta,
        esperado="", verificacao="exato",
    )
    instrucao_ape = mod_ape.carregar_instrucao() if rota.estrategia == "ape" else ""

    resultado = executar_uma(
        pergunta, rota.estrategia, exemplos, SETTINGS, instrucao_ape=instrucao_ape
    )

    if resultado.erro:
        print(f"\n[erro] {resultado.erro}")
        return

    print(f"\nSaída ({resultado.tokens_totais} tokens, {resultado.latencia:.2f}s):")
    print("-" * LARGURA)
    print(resultado.saida)
    print("-" * LARGURA)
    print(f"Resposta extraída: {resultado.resposta_extraida}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BLIS — Módulo 02: Engenharia de Prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def add_filtros(p):
        p.add_argument("--categoria", nargs="+", help="filtra por categoria")
        p.add_argument("--ids", nargs="+", help="filtra por id de pergunta")

    p = sub.add_parser("listar", help="estratégias e conjunto (sem API)")
    p.set_defaults(func=cmd_listar)

    p = sub.add_parser("ver-prompt", help="inspeciona um prompt montado (sem API)")
    p.add_argument("estrategia", choices=list(ESTRATEGIAS))
    p.add_argument("--pergunta-id", default="", help="id da pergunta a usar")
    add_filtros(p)
    p.set_defaults(func=cmd_ver_prompt)

    p = sub.add_parser("validar", help="checa vazamento e consistência (sem API)")
    p.set_defaults(func=cmd_validar)

    p = sub.add_parser("comparar", help="roda a matriz estratégia x pergunta")
    p.add_argument("--estrategias", nargs="+", choices=list(ESTRATEGIAS))
    p.add_argument("--repeticoes", type=int, default=1)
    p.add_argument("--sim", action="store_true", help="não pede confirmação")
    add_filtros(p)
    p.set_defaults(func=cmd_comparar)

    p = sub.add_parser("relatorio", help="relatórios da última execução")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--html", action="store_true")
    p.add_argument("--divergencias", action="store_true")
    p.add_argument("--tudo", action="store_true", help="exporta em todos os formatos")
    p.set_defaults(func=cmd_relatorio)

    p = sub.add_parser("rotear", help="mostra a estratégia escolhida (sem API)")
    p.add_argument("pergunta")
    p.add_argument("--metodo", default="heuristico",
                   choices=["heuristico", "llm", "hibrido"])
    p.set_defaults(func=cmd_rotear)

    p = sub.add_parser("aprender-rotas", help="deriva a tabela dos resultados medidos")
    p.add_argument("--criterio", default="custo", choices=["acuracia", "custo"])
    p.set_defaults(func=cmd_aprender_rotas)

    p = sub.add_parser("ape", help="otimiza a instrução automaticamente")
    p.add_argument("--candidatas", type=int, default=0)
    p.add_argument("--rodadas", type=int, default=None)
    p.add_argument("--subconjunto", type=int, default=8)
    p.add_argument("--sim", action="store_true")
    add_filtros(p)
    p.set_defaults(func=cmd_ape)

    p = sub.add_parser("responder", help="responde uma pergunta com roteamento")
    p.add_argument("pergunta")
    p.add_argument("--metodo", default="heuristico",
                   choices=["heuristico", "llm", "hibrido"])
    add_filtros(p)
    p.set_defaults(func=cmd_responder)

    return parser


def main() -> None:
    args = construir_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as erro:  # noqa: BLE001
        print(f"\n[erro] {erro}")
        sys.exit(1)


if __name__ == "__main__":
    main()
