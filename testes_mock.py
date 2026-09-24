"""
Testes com LLM simulado — validam os caminhos que chamam a API, sem gastar nada.

Substitui o cliente da OpenRouter por um dublê que devolve respostas
controladas, e verifica que as chamadas são montadas no formato certo, que as
respostas são interpretadas corretamente, e que respostas malformadas não
derrubam o experimento.

    python testes_mock.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import ape as mod_ape
import avaliacao as mod_avaliacao
import config
import roteamento as mod_roteamento
import runner as mod_runner
from dataset import Pergunta
from prompts import Exemplo

FALHAS: list[str] = []
CHAMADAS: list[dict] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]   {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * 62)


# --------------------------------------------------------------------------- #
# Dublê do cliente
# --------------------------------------------------------------------------- #


class ChatFalso:
    def __init__(self, respostas: list[str]) -> None:
        self.respostas = respostas
        self.i = 0

    def create(self, **kwargs):
        CHAMADAS.append(kwargs)
        texto = self.respostas[min(self.i, len(self.respostas) - 1)]
        self.i += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=texto))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=35),
        )


class ClienteFalso:
    def __init__(self, respostas: list[str]) -> None:
        self.chat = SimpleNamespace(completions=ChatFalso(respostas))


def instalar(respostas: list[str]) -> ClienteFalso:
    """Substitui get_client em todos os módulos que o importaram."""
    CHAMADAS.clear()
    cliente = ClienteFalso(respostas)
    falso = lambda settings=None: cliente  # noqa: E731

    for modulo in (config, mod_runner, mod_ape, mod_roteamento, mod_avaliacao):
        if hasattr(modulo, "get_client"):
            modulo.get_client = falso
    return cliente


class ClienteQuebrado:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: (_ for _ in ()).throw(RuntimeError("falha de rede"))
            )
        )


class SettingsFalsas:
    api_key = "sk-teste"
    chat_model = "modelo/teste"
    temperature = 0.0
    max_tokens = 700
    k_fewshot = 2
    repeticoes = 1
    ape_candidatos = 3
    ape_temperatura = 0.9
    ape_rodadas = 1
    ape_top_k = 2
    tem_chave = True


S = SettingsFalsas()

EXEMPLOS = [
    Exemplo("Quanto é 2+2?", "4", "Dois mais dois são quatro.", "aritmetica"),
    Exemplo("Quanto é 3+3?", "6", "Três mais três são seis.", "aritmetica"),
    Exemplo("Sentimento: adorei", "positivo", "É um elogio.", "classificacao"),
]

PERGUNTAS = [
    Pergunta("q1", "aritmetica", "Quanto é 5+5?", "10", "numerico"),
    Pergunta("q2", "aritmetica", "Quanto é 7+7?", "14", "numerico"),
    Pergunta("q3", "classificacao", "Sentimento: odiei", "negativo", "exato"),
]


# --------------------------------------------------------------------------- #


def testar_execucao_unitaria() -> None:
    secao("1. Execução de uma combinação")
    instalar(["O cálculo é 5+5.\nRESPOSTA: 10"])

    r = mod_runner.executar_uma(PERGUNTAS[0], "few_shot", EXEMPLOS, S, k=2)

    checar(r.acertou, "resposta correta é marcada como acerto")
    checar(r.resposta_extraida == "10", "extrai a resposta do formato pedido")
    checar(r.tokens_prompt == 120 and r.tokens_resposta == 35, "registra os tokens")
    checar(r.latencia > 0, "registra a latência")
    checar(not r.erro, "sem erro em execução normal")
    checar(r.estrategia == "few_shot", "registra a estratégia usada")
    checar(r.categoria == "aritmetica", "registra a categoria")
    checar(r.tokens_estimados_prompt > 0, "estima o tamanho do prompt")

    chamada = CHAMADAS[0]
    checar(chamada["model"] == "modelo/teste", "usa o modelo configurado")
    checar(chamada["temperature"] == 0.0, "usa a temperatura configurada")
    checar(len(chamada["messages"]) == 2, "envia system + user")
    checar(
        "Quanto é 2+2?" in chamada["messages"][-1]["content"],
        "few_shot inclui as demonstrações no prompt",
    )

    # Resposta errada
    instalar(["RESPOSTA: 11"])
    errada = mod_runner.executar_uma(PERGUNTAS[0], "few_shot", EXEMPLOS, S, k=2)
    checar(not errada.acertou, "resposta incorreta é marcada como erro")

    # Falha de rede não derruba: vira um Resultado com erro
    mod_runner.get_client = lambda settings=None: ClienteQuebrado()
    falhou = mod_runner.executar_uma(PERGUNTAS[0], "zero_shot", EXEMPLOS, S)
    checar(bool(falhou.erro), "falha de rede é registrada no campo erro")
    checar(not falhou.acertou, "resultado com erro não conta como acerto")
    checar(falhou.latencia >= 0, "latência é registrada mesmo na falha")


def testar_matriz() -> None:
    secao("2. Matriz estratégia x pergunta")
    instalar(["RESPOSTA: 10", "RESPOSTA: 14", "RESPOSTA: negativo"])

    execucao = mod_runner.executar(
        PERGUNTAS, ["zero_shot", "few_shot"], EXEMPLOS, S, verboso=False
    )

    checar(len(execucao.resultados) == 6, "roda 2 estratégias x 3 perguntas")
    checar(len(CHAMADAS) == 6, "faz uma chamada por célula da matriz")
    checar(execucao.estrategias() == ["zero_shot", "few_shot"],
           "preserva a ordem das estratégias")
    checar(execucao.modelo == "modelo/teste", "registra o modelo na execução")
    checar(bool(execucao.timestamp), "registra o timestamp")

    # Repetições
    instalar(["RESPOSTA: 10"])
    rep = mod_runner.executar(
        PERGUNTAS[:1], ["zero_shot"], EXEMPLOS, S, repeticoes=3, verboso=False
    )
    checar(len(rep.resultados) == 3, "repeticoes=3 gera 3 resultados")
    checar([r.repeticao for r in rep.resultados] == [0, 1, 2],
           "numera as repetições")

    # Persistência (ida e volta)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "exec.json"
        execucao.salvar(caminho)
        recarregada = mod_runner.Execucao.carregar(caminho)
        checar(
            len(recarregada.resultados) == len(execucao.resultados),
            "salva e recarrega sem perder resultados",
        )
        checar(recarregada.modelo == execucao.modelo, "preserva os metadados")
        checar(
            recarregada.resultados[0].acertou == execucao.resultados[0].acertou,
            "preserva os acertos",
        )


def testar_subconjunto() -> None:
    secao("3. Acurácia em subconjunto (score do APE)")
    instalar(["RESPOSTA: 10", "RESPOSTA: 14", "RESPOSTA: negativo"])

    score = mod_runner.executar_estrategia_em_subconjunto(
        PERGUNTAS, "ape", EXEMPLOS, S, instrucao_ape="Instrução X"
    )
    checar(abs(score - 1.0) < 1e-9, "acerta tudo -> score 1.0")
    checar(
        "Instrução X" in CHAMADAS[0]["messages"][0]["content"],
        "a instrução do APE chega ao prompt",
    )

    instalar(["RESPOSTA: 99"])
    zero = mod_runner.executar_estrategia_em_subconjunto(PERGUNTAS, "zero_shot", EXEMPLOS, S)
    checar(zero == 0.0, "erra tudo -> score 0.0")

    instalar(["RESPOSTA: 10", "RESPOSTA: 0", "RESPOSTA: 0"])
    parcial = mod_runner.executar_estrategia_em_subconjunto(
        PERGUNTAS, "zero_shot", EXEMPLOS, S
    )
    checar(abs(parcial - 1 / 3) < 1e-9, "1 de 3 -> score 0.333")

    checar(
        mod_runner.executar_estrategia_em_subconjunto([], "zero_shot", EXEMPLOS, S) == 0.0,
        "subconjunto vazio devolve 0 sem quebrar",
    )


def testar_ape() -> None:
    secao("4. APE — proposta, reamostragem e busca")

    instalar([
        "Resolva o problema com cuidado.",
        "Calcule a resposta passo a passo.",
        "Resolva o problema com cuidado.",  # duplicata: deve ser descartada
    ])
    candidatas = mod_ape.propor_instrucoes(EXEMPLOS, n=3, settings=S, verboso=False)

    checar(len(candidatas) == 2, "descarta candidatas duplicadas")
    checar(len(CHAMADAS) == 3, "faz uma chamada por candidata pedida")
    checar(CHAMADAS[0]["temperature"] == 0.9,
           "usa temperatura alta para diversificar as propostas")
    checar(
        "pares de entrada e saída" in CHAMADAS[0]["messages"][0]["content"],
        "usa o template de geração forward do artigo",
    )
    checar(
        all(c.origem == "proposta" for c in candidatas),
        "marca a origem das candidatas",
    )

    instalar(["Analise o enunciado e responda com precisão."])
    variacoes = mod_ape.reamostrar(candidatas[:1], 1, 1, S, verboso=False)
    checar(len(variacoes) == 1, "reamostragem gera uma variação")
    checar(variacoes[0].origem == "reamostragem", "marca a origem como reamostragem")
    checar(variacoes[0].rodada == 1, "registra a rodada")

    # Variação idêntica à original deve ser descartada.
    instalar([candidatas[0].instrucao])
    iguais = mod_ape.reamostrar(candidatas[:1], 1, 1, S, verboso=False)
    checar(len(iguais) == 0, "variação idêntica à original é descartada")

    # Busca completa
    instalar(
        ["Instrução A", "Instrução B", "Instrução C"]     # propostas
        + ["RESPOSTA: 10"] * 60                           # pontuações
    )
    resultado = mod_ape.buscar(
        PERGUNTAS, EXEMPLOS, S, n_candidatas=3, rodadas=0,
        tamanho_subconjunto=2, verboso=False,
    )
    checar(resultado.melhor is not None, "a busca devolve uma melhor instrução")
    checar(len(resultado.historico) >= 3, "registra o histórico de candidatas")
    checar(resultado.chamadas_llm > 0, "contabiliza as chamadas ao LLM")
    checar(
        all(0.0 <= c.score <= 1.0 for c in resultado.historico),
        "todos os scores ficam em [0,1]",
    )
    checar(
        resultado.melhor.score == max(c.score for c in resultado.historico),
        "a melhor é realmente a de maior score",
    )
    checar(
        any(c.origem == "semente" for c in resultado.historico),
        "inclui as sementes escritas à mão como referência",
    )

    # Falha na proposta não derruba a busca
    mod_ape.get_client = lambda settings=None: ClienteQuebrado()
    vazias = mod_ape.propor_instrucoes(EXEMPLOS, n=2, settings=S, verboso=False)
    checar(vazias == [], "falha ao propor devolve lista vazia sem quebrar")


def testar_roteador_llm() -> None:
    secao("5. Roteador por LLM")

    instalar(['{"categoria": "aritmetica", "confianca": 0.95}'])
    categoria, confianca = mod_roteamento.classificar_llm("Quanto é 5+5?", S)
    checar(categoria == "aritmetica", "interpreta a categoria do JSON")
    checar(abs(confianca - 0.95) < 1e-9, "interpreta a confiança do JSON")
    checar(CHAMADAS[0]["temperature"] == 0.0, "classifica com temperatura 0")

    # Categoria inventada deve cair na heurística
    instalar(['{"categoria": "categoria_inventada", "confianca": 0.9}'])
    cat, _ = mod_roteamento.classificar_llm("Quantas maçãs sobraram?", S)
    checar(cat == "aritmetica", "categoria inválida cai na heurística")

    instalar(["isso não é json"])
    cat, _ = mod_roteamento.classificar_llm("Classifique o sentimento: adorei", S)
    checar(cat == "classificacao", "resposta ilegível cai na heurística")

    mod_roteamento.get_client = lambda settings=None: ClienteQuebrado()
    cat, _ = mod_roteamento.classificar_llm("Quantas maçãs sobraram?", S)
    checar(cat == "aritmetica", "falha de rede cai na heurística")

    # Modo híbrido: escala ao LLM quando a heurística está insegura
    # A heurística é confiante nesta pergunta (confiança 1.0); para exercitar a
    # escalada de propósito, o limiar precisa ser maior que 1.0.
    instalar(['{"categoria": "factual", "confianca": 0.8}'])
    roteador = mod_roteamento.Roteador(metodo="hibrido", limiar_escalada=1.1)
    rota = roteador.rotear("Quantas maçãs sobraram?")
    checar("escalou" in rota.metodo, "modo híbrido escala quando a confiança é baixa")
    checar(rota.categoria == "factual", "usa a categoria devolvida pelo LLM")

    # Com limiar 0, nunca escala — e não gasta chamada.
    instalar(['{"categoria": "factual", "confianca": 0.8}'])
    sem_escalar = mod_roteamento.Roteador(metodo="hibrido", limiar_escalada=0.0)
    rota2 = sem_escalar.rotear("Quantas maçãs sobraram?")
    checar(rota2.categoria == "aritmetica", "confiança alta não escala")
    checar(len(CHAMADAS) == 0, "sem escalada, não gasta chamada ao LLM")


def testar_juiz() -> None:
    secao("6. Juiz por LLM")

    instalar(['{"correta": true, "motivo": "mesma informação"}'])
    j = mod_avaliacao.julgar_com_llm("P?", "Canberra", "Canberra", S)
    checar(j.correta, "juiz aprova resposta equivalente")
    checar(j.motivo == "mesma informação", "captura o motivo")

    instalar(['{"correta": false, "motivo": "cidade errada"}'])
    j = mod_avaliacao.julgar_com_llm("P?", "Sydney", "Canberra", S)
    checar(not j.correta, "juiz reprova resposta divergente")

    instalar(["sem json aqui"])
    j = mod_avaliacao.julgar_com_llm("P?", "x", "y", S)
    checar(not j.correta, "resposta ilegível é tratada como incorreta")

    config.get_client = lambda settings=None: ClienteQuebrado()
    mod_avaliacao.get_client = config.get_client
    j = mod_avaliacao.julgar_com_llm("P?", "x", "y", S)
    checar(not j.correta and "falha" in j.motivo, "falha de rede não quebra o juiz")


def main() -> int:
    print("=" * 62)
    print("  TESTES COM LLM SIMULADO — sem API, sem custo")
    print("=" * 62)

    testar_execucao_unitaria()
    testar_matriz()
    testar_subconjunto()
    testar_ape()
    testar_roteador_llm()
    testar_juiz()

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
