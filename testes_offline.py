"""
Testes offline — validam a lógica sem chamar a API nem gastar créditos.

Cobrem montagem de prompts, extração e verificação de respostas, carregamento e
validação do conjunto, roteamento heurístico, agregação dos relatórios e a
limpeza de instruções do APE.

    python testes_offline.py
"""

from __future__ import annotations

import sys

from avaliacao import (
    extrair_resposta,
    normalizar,
    normalizar_numero,
    verificar,
    verificar_contem,
    verificar_exato,
    verificar_json,
    verificar_numerico,
)
from ape import Candidata, _limpar_instrucao
from dataset import Pergunta, carregar_exemplos, carregar_perguntas, exemplos_para, validar
from prompts import (
    ESTRATEGIAS,
    PADRAO,
    Exemplo,
    obter,
    tamanho_aproximado,
    texto_do_prompt,
)
from relatorio import agregar
from roteamento import Roteador, aprender_tabela, classificar_heuristico
from runner import Execucao, Resultado

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]   {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * 62)


EXEMPLOS = [
    Exemplo("Quanto é 2+2?", "4", "Dois mais dois são quatro.", "aritmetica"),
    Exemplo("Quanto é 3+3?", "6", "Três mais três são seis.", "aritmetica"),
    Exemplo("Sentimento: adorei", "positivo", "É um elogio claro.", "classificacao"),
    Exemplo("Sentimento: odiei", "negativo", "É uma crítica clara.", "classificacao"),
]


# --------------------------------------------------------------------------- #


def testar_prompts() -> None:
    secao("1. Montagem de prompts")

    for nome in PADRAO:
        estrategia = obter(nome)
        extra = {"k": 2} if estrategia.usa_exemplos else {}
        msgs = estrategia.montar("Quanto é 5+5?", EXEMPLOS, **extra)

        checar(bool(msgs), f"'{nome}' produz mensagens")
        checar(
            all(set(m) == {"role", "content"} for m in msgs),
            f"'{nome}' usa o formato role/content",
        )
        checar(
            all(m["role"] in {"system", "user", "assistant"} for m in msgs),
            f"'{nome}' usa papéis válidos",
        )
        checar(
            any("5+5" in m["content"] for m in msgs),
            f"'{nome}' inclui a pergunta no prompt",
        )

    # Zero-shot puro: nada além da pergunta.
    zs = obter("zero_shot").montar("Quanto é 5+5?", EXEMPLOS)
    checar(len(zs) == 1 and zs[0]["role"] == "user", "zero_shot envia só a pergunta")
    checar(zs[0]["content"] == "Quanto é 5+5?", "zero_shot não acrescenta instrução")

    # One-shot: exatamente uma demonstração.
    os_ = obter("one_shot").montar("Quanto é 5+5?", EXEMPLOS, k=1)
    corpo = os_[-1]["content"]
    checar(corpo.count("P: ") == 2, "one_shot traz 1 demonstração + a pergunta")

    # Few-shot com K=3.
    fs = obter("few_shot").montar("Quanto é 5+5?", EXEMPLOS, k=3)
    checar(fs[-1]["content"].count("P: ") == 4, "few_shot com K=3 traz 3 + 1")

    # CoT few-shot precisa conter o raciocínio das demonstrações.
    cot = obter("cot_few_shot").montar("Quanto é 5+5?", EXEMPLOS, k=2)
    checar(
        "Dois mais dois são quatro" in cot[-1]["content"],
        "cot_few_shot inclui a cadeia de raciocínio",
    )
    checar(
        "A resposta é 4" in cot[-1]["content"],
        "cot_few_shot põe a resposta final DEPOIS do raciocínio",
    )

    # Sem raciocínio, CoT degrada para few-shot em vez de quebrar.
    sem_rac = [Exemplo("Q?", "A", "", "geral")]
    degradado = obter("cot_few_shot").montar("Quanto é 5+5?", sem_rac, k=1)
    checar("P: Q?\nR: A" in degradado[-1]["content"],
           "cot_few_shot sem raciocínio degrada para par simples")

    # Custo relativo: few-shot deve ser mais caro que zero-shot.
    t_zs = tamanho_aproximado(obter("zero_shot_instrucao").montar("Q?", EXEMPLOS))
    t_fs = tamanho_aproximado(obter("few_shot").montar("Q?", EXEMPLOS, k=3))
    checar(t_fs > t_zs, "few_shot custa mais tokens que zero_shot")

    # APE usa a instrução fornecida.
    com_instrucao = obter("ape").montar("Q?", None, instrucao="INSTRUÇÃO XYZ")
    checar(
        "INSTRUÇÃO XYZ" in com_instrucao[0]["content"],
        "ape usa a instrução otimizada recebida",
    )

    texto = texto_do_prompt(fs)
    checar("[SYSTEM]" in texto and "[USER]" in texto, "renderização legível do prompt")

    try:
        obter("inexistente")
        checar(False, "estratégia desconhecida levanta erro")
    except ValueError:
        checar(True, "estratégia desconhecida levanta erro")


def testar_extracao() -> None:
    secao("2. Extração da resposta")

    checar(extrair_resposta("bla bla\nRESPOSTA: 42") == "42",
           "extrai do formato RESPOSTA:")
    checar(extrair_resposta("Portanto, a resposta é 42") == "42",
           "extrai de 'a resposta é'")
    checar(extrair_resposta("R: 42") == "42", "extrai do formato R:")
    checar(extrair_resposta("só uma linha solta") == "só uma linha solta",
           "sem marcador, usa a última linha")
    checar(extrair_resposta("") == "", "saída vazia devolve vazio")
    checar(extrair_resposta("RESPOSTA: **42**") == "42", "remove negrito markdown")

    # A última ocorrência é a que vale: o modelo pode citar o formato antes.
    multi = "Vou usar RESPOSTA: rascunho\nAgora sim\nRESPOSTA: final"
    checar(extrair_resposta(multi) == "final", "usa a ÚLTIMA ocorrência do marcador")

    longo = "Passo 1: somar.\nPasso 2: conferir.\nRESPOSTA: 11"
    checar(extrair_resposta(longo) == "11", "ignora o raciocínio e pega a resposta")


def testar_numeros() -> None:
    secao("3. Normalização numérica")

    casos = [
        ("42", 42.0), ("R$ 1.234,56", 1234.56), ("1,234.56", 1234.56),
        ("11 maçãs", 11.0), ("-5", -5.0), ("3,5", 3.5), ("1.500", 1500.0),
        ("R$ 347,90", 347.90), ("100%", 100.0), ("225", 225.0),
    ]
    for texto, esperado in casos:
        obtido = normalizar_numero(texto)
        checar(
            obtido is not None and abs(obtido - esperado) < 1e-6,
            f"'{texto}' -> {esperado} (obtido {obtido})",
        )

    checar(normalizar_numero("sem número") is None, "texto sem número devolve None")
    checar(normalizar_numero(None) is None, "None não quebra")

    checar(verificar_numerico("R$ 347,90", "347,90"), "compara valores formatados")
    checar(verificar_numerico("a resposta é 11 bolas", "11"), "ignora unidades")
    checar(not verificar_numerico("12", "11"), "rejeita número diferente")


def testar_verificadores() -> None:
    secao("4. Verificadores")

    checar(verificar_exato("Positivo.", "positivo"), "exato ignora caixa e pontuação")
    checar(verificar_exato("nao", ["nao", "não"]), "exato aceita lista de alternativas")
    checar(not verificar_exato("negativo", "positivo"), "exato rejeita divergente")

    checar(verificar_contem("A capital é Canberra.", "canberra"),
           "contem encontra a substring")
    checar(verificar_contem("Vetrix Logística", "vetrix logistica"),
           "contem ignora acentos")

    checar(verificar_json('{"nome": "x", "cargo": "y"}', ["nome", "cargo"]),
           "json valida as chaves obrigatórias")
    checar(verificar_json('bla {"a": 1, "b": 2} bla', ["a", "b"]),
           "json encontra o objeto embutido em texto")
    checar(not verificar_json('{"nome": "x"}', ["nome", "cargo"]),
           "json rejeita chave faltando")
    checar(not verificar_json("não é json", ["a"]), "json rejeita texto solto")

    acertou, extraida = verificar("RESPOSTA: 11", "11", "numerico")
    checar(acertou and extraida == "11", "verificar() integra extração e checagem")

    acertou, _ = verificar("Não sei responder.", "(não sei|nao sei)", "regex")
    checar(acertou, "regex detecta abstenção")

    try:
        verificar("x", "y", "inexistente")
        checar(False, "tipo de verificação inválido levanta erro")
    except ValueError:
        checar(True, "tipo de verificação inválido levanta erro")

    checar(normalizar("  Ação,  RÁPIDA!  ") == "acao, rapida",
           "normalização remove acentos, caixa e pontuação de borda")


def testar_dataset() -> None:
    secao("5. Conjunto de avaliação")

    perguntas = carregar_perguntas()
    exemplos = carregar_exemplos()

    checar(len(perguntas) >= 20, f"conjunto tem {len(perguntas)} perguntas")
    checar(len(exemplos) >= 5, f"há {len(exemplos)} demonstrações few-shot")
    checar(
        len({p.id for p in perguntas}) == len(perguntas),
        "todos os ids de pergunta são únicos",
    )

    avisos = validar(perguntas, exemplos)
    vazamentos = [a for a in avisos if "VAZAMENTO" in a]
    checar(not vazamentos, "nenhum vazamento entre avaliação e demonstrações")
    checar(not avisos, f"conjunto sem avisos (encontrados: {len(avisos)})")

    # O detector de vazamento precisa realmente detectar.
    plantado = perguntas + [
        Pergunta("plant", "aritmetica", exemplos[0].pergunta, "x", "exato")
    ]
    checar(
        any("VAZAMENTO" in a for a in validar(plantado, exemplos)),
        "o detector encontra um vazamento plantado",
    )

    demos = exemplos_para("aritmetica", exemplos, k=3)
    checar(len(demos) == 3, "seleciona K demonstrações")
    checar(
        demos[0].categoria == "aritmetica",
        "prioriza demonstrações da mesma categoria",
    )

    demos_raras = exemplos_para("categoria_inexistente", exemplos, k=3)
    checar(len(demos_raras) == 3, "completa com outras categorias quando falta")


def testar_roteamento() -> None:
    secao("6. Roteamento heurístico")

    casos = [
        ("Quantas bolas ele tem agora?", "aritmetica"),
        ("Classifique o sentimento do texto: adorei", "classificacao"),
        ("Extraia apenas o valor total: R$ 347,90", "extracao"),
        ("Responda APENAS com um objeto JSON com as chaves a e b", "extracao"),
        ("Todos os bloxes são frims. Necessariamente algum é gral?", "logica"),
        ("Qual é a capital da Austrália?", "factual"),
        ("Qual a cor favorita do autor do artigo?", "abstencao"),
    ]
    for pergunta, esperada in casos:
        categoria, _, _ = classificar_heuristico(pergunta)
        checar(categoria == esperada, f"'{pergunta[:38]}...' -> {esperada}")

    # Verbo de tarefa deve vencer a presença de números.
    cat, _, _ = classificar_heuristico("Extraia o valor total de R$ 347,90 do texto")
    checar(cat == "extracao", "verbo 'extraia' pesa mais que a menção a valores")

    cat, conf, _ = classificar_heuristico("blablabla")
    checar(cat == "geral" and conf == 0.0, "texto sem sinal cai em 'geral'")

    roteador = Roteador()
    rota = roteador.rotear("Quantas maçãs sobraram depois de usar 20?")
    checar(rota.categoria == "aritmetica", "roteador identifica a categoria")
    checar(bool(rota.estrategia), "roteador escolhe uma estratégia")
    checar(rota.estrategia in ESTRATEGIAS, "a estratégia escolhida existe")
    checar(bool(rota.evidencias), "roteador registra as evidências usadas")

    # Acurácia do roteador sobre o conjunto real.
    perguntas = carregar_perguntas()
    acertos = sum(
        classificar_heuristico(p.pergunta)[0] == p.categoria for p in perguntas
    )
    taxa = acertos / len(perguntas)
    checar(taxa >= 0.85, f"acurácia do roteador no conjunto: {taxa:.1%}")

    # Tabela customizada é respeitada.
    custom = Roteador({"aritmetica": "persona"})
    checar(
        custom.rotear("Quantas maçãs sobraram?").estrategia == "persona",
        "tabela customizada é respeitada",
    )
    checar(
        custom.rotear("blablabla").estrategia == "zero_shot_instrucao",
        "categoria ausente da tabela cai no padrão",
    )


def testar_relatorio() -> None:
    secao("7. Agregação dos relatórios")

    execucao = Execucao(modelo="teste", resultados=[
        Resultado("q1", "aritmetica", "few_shot", acertou=True,
                  tokens_prompt=100, tokens_resposta=20, latencia=1.0),
        Resultado("q2", "aritmetica", "few_shot", acertou=False,
                  tokens_prompt=100, tokens_resposta=20, latencia=1.0),
        Resultado("q3", "classificacao", "few_shot", acertou=True,
                  tokens_prompt=100, tokens_resposta=20, latencia=1.0),
        Resultado("q1", "aritmetica", "zero_shot", acertou=False,
                  tokens_prompt=20, tokens_resposta=10, latencia=0.5),
        Resultado("q2", "aritmetica", "zero_shot", acertou=False,
                  tokens_prompt=20, tokens_resposta=10, latencia=0.5),
        Resultado("q3", "classificacao", "zero_shot", acertou=True,
                  tokens_prompt=20, tokens_resposta=10, latencia=0.5),
    ])

    ag = agregar(execucao)
    checar(len(ag) == 2, "agrega por estratégia")
    checar(abs(ag["few_shot"].acuracia - 2 / 3) < 1e-9, "acurácia geral correta")
    checar(abs(ag["zero_shot"].acuracia - 1 / 3) < 1e-9, "acurácia da segunda correta")
    checar(ag["few_shot"].tokens_totais == 360, "soma os tokens")
    checar(abs(ag["few_shot"].tokens_por_item - 120) < 1e-9, "tokens por item")
    checar(abs(ag["few_shot"].latencia_media - 1.0) < 1e-9, "latência média")

    checar(
        abs(ag["few_shot"].acuracia_categoria("aritmetica") - 0.5) < 1e-9,
        "acurácia por categoria correta",
    )
    checar(
        ag["zero_shot"].acuracia_categoria("classificacao") == 1.0,
        "zero_shot acerta 100% em classificação neste exemplo",
    )
    checar(
        ag["few_shot"].acuracia_categoria("inexistente") == 0.0,
        "categoria ausente devolve 0 em vez de quebrar",
    )

    checar(execucao.estrategias() == ["few_shot", "zero_shot"],
           "lista as estratégias na ordem de aparição")
    checar(execucao.categorias() == ["aritmetica", "classificacao"],
           "lista as categorias na ordem de aparição")

    # Tabela aprendida: few_shot vence em aritmética; em classificação há empate
    # e o critério de custo deve escolher a mais barata (zero_shot).
    tabela = aprender_tabela(execucao, criterio="custo")
    checar(tabela["aritmetica"] == "few_shot",
           "aprende que few_shot vence em aritmética")
    checar(tabela["classificacao"] == "zero_shot",
           "no empate, o critério de custo escolhe a mais barata")

    por_acuracia = aprender_tabela(execucao, criterio="acuracia")
    checar("classificacao" in por_acuracia, "critério por acurácia também funciona")


def testar_ape() -> None:
    secao("8. APE — utilidades")

    casos = [
        ('"Escreva a resposta."', "Escreva a resposta."),
        ("1. Resolva a tarefa.", "Resolva a tarefa."),
        ("- Some os valores.", "Some os valores."),
        ("A instrução era: Classifique o texto.", "Classifique o texto."),
        ("Instrução: Extraia o dado.", "Extraia o dado."),
        ("  Responda corretamente.  ", "Responda corretamente."),
    ]
    for bruto, esperado in casos:
        obtido = _limpar_instrucao(bruto)
        checar(obtido == esperado, f"limpa {bruto[:34]!r} -> {esperado!r}")

    checar(_limpar_instrucao("") == "", "string vazia não quebra")
    checar(_limpar_instrucao(None) == "", "None não quebra")

    c = Candidata(instrucao="Teste", score=0.75, origem="proposta")
    checar(c.score == 0.75 and c.origem == "proposta", "Candidata guarda os campos")


def main() -> int:
    print("=" * 62)
    print("  TESTES OFFLINE — sem API, sem custo")
    print("=" * 62)

    testar_prompts()
    testar_extracao()
    testar_numeros()
    testar_verificadores()
    testar_dataset()
    testar_roteamento()
    testar_relatorio()
    testar_ape()

    print("\n" + "=" * 62)
    if FALHAS:
        print(f"  {len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"    - {f}")
        print("=" * 62)
        return 1

    print("  Todos os testes passaram.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
