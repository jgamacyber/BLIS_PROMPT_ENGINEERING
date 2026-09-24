"""
Execução do experimento: a matriz estratégia × pergunta.

Para cada combinação, monta o prompt, chama o modelo, extrai a resposta,
verifica contra o gabarito e registra tudo — incluindo custo em tokens e
latência, porque uma estratégia que acerta mais gastando cinco vezes mais
contexto nem sempre compensa.

Os resultados são salvos em JSON para que os relatórios possam ser regerados
sem repetir as chamadas.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from avaliacao import julgar_com_llm, verificar
from config import SETTINGS, Settings, get_client
from dataset import Pergunta, exemplos_para
from prompts import Exemplo, obter, tamanho_aproximado

CAMINHO_RESULTADOS = Path("resultados/execucao.json")


@dataclass
class Resultado:
    """Uma célula da matriz: uma estratégia respondendo uma pergunta."""

    pergunta_id: str
    categoria: str
    estrategia: str
    repeticao: int = 0
    saida: str = ""
    resposta_extraida: str = ""
    esperado: str = ""
    acertou: bool = False
    tokens_prompt: int = 0
    tokens_resposta: int = 0
    tokens_estimados_prompt: int = 0
    latencia: float = 0.0
    erro: str = ""

    @property
    def tokens_totais(self) -> int:
        return self.tokens_prompt + self.tokens_resposta


@dataclass
class Execucao:
    """O experimento inteiro, serializável."""

    resultados: list[Resultado] = field(default_factory=list)
    modelo: str = ""
    temperatura: float = 0.0
    k_fewshot: int = 3
    repeticoes: int = 1
    instrucao_ape: str = ""
    timestamp: str = ""

    def salvar(self, caminho: str | Path = CAMINHO_RESULTADOS) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "modelo": self.modelo,
            "temperatura": self.temperatura,
            "k_fewshot": self.k_fewshot,
            "repeticoes": self.repeticoes,
            "instrucao_ape": self.instrucao_ape,
            "timestamp": self.timestamp,
            "resultados": [asdict(r) for r in self.resultados],
        }
        caminho.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def carregar(cls, caminho: str | Path = CAMINHO_RESULTADOS) -> "Execucao":
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Resultados não encontrados em {caminho}. "
                f"Rode primeiro: python main.py comparar"
            )
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return cls(
            resultados=[Resultado(**r) for r in dados["resultados"]],
            modelo=dados.get("modelo", ""),
            temperatura=dados.get("temperatura", 0.0),
            k_fewshot=dados.get("k_fewshot", 3),
            repeticoes=dados.get("repeticoes", 1),
            instrucao_ape=dados.get("instrucao_ape", ""),
            timestamp=dados.get("timestamp", ""),
        )

    def estrategias(self) -> list[str]:
        vistas: list[str] = []
        for r in self.resultados:
            if r.estrategia not in vistas:
                vistas.append(r.estrategia)
        return vistas

    def categorias(self) -> list[str]:
        vistas: list[str] = []
        for r in self.resultados:
            if r.categoria not in vistas:
                vistas.append(r.categoria)
        return vistas


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #


def executar_uma(
    pergunta: Pergunta,
    nome_estrategia: str,
    exemplos: list[Exemplo],
    settings: Settings | None = None,
    repeticao: int = 0,
    k: int = 3,
    instrucao_ape: str = "",
) -> Resultado:
    """Executa uma única combinação estratégia × pergunta."""
    settings = settings or SETTINGS
    estrategia = obter(nome_estrategia)

    extra: dict = {}
    if estrategia.usa_exemplos:
        extra["k"] = k
    if nome_estrategia == "ape" and instrucao_ape:
        extra["instrucao"] = instrucao_ape

    demos = exemplos_para(pergunta.categoria, exemplos, k) if estrategia.usa_exemplos else []
    mensagens = estrategia.montar(pergunta.pergunta, demos, **extra)

    resultado = Resultado(
        pergunta_id=pergunta.id,
        categoria=pergunta.categoria,
        estrategia=nome_estrategia,
        repeticao=repeticao,
        esperado=str(pergunta.esperado),
        tokens_estimados_prompt=tamanho_aproximado(mensagens),
    )

    inicio = time.perf_counter()
    try:
        client = get_client(settings)
        resposta = client.chat.completions.create(
            model=settings.chat_model,
            messages=mensagens,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        resultado.saida = (resposta.choices[0].message.content or "").strip()
        uso = getattr(resposta, "usage", None)
        resultado.tokens_prompt = getattr(uso, "prompt_tokens", 0) or 0
        resultado.tokens_resposta = getattr(uso, "completion_tokens", 0) or 0
    except Exception as erro:  # noqa: BLE001
        resultado.erro = str(erro)
        resultado.latencia = time.perf_counter() - inicio
        return resultado

    resultado.latencia = time.perf_counter() - inicio

    if pergunta.verificacao == "llm":
        julgamento = julgar_com_llm(
            pergunta.pergunta, resultado.saida, str(pergunta.esperado), settings
        )
        resultado.acertou = julgamento.correta
        resultado.resposta_extraida = resultado.saida[:200]
    else:
        resultado.acertou, resultado.resposta_extraida = verificar(
            resultado.saida, pergunta.esperado, pergunta.verificacao
        )

    return resultado


def executar(
    perguntas: list[Pergunta],
    estrategias: list[str],
    exemplos: list[Exemplo],
    settings: Settings | None = None,
    repeticoes: int = 1,
    instrucao_ape: str = "",
    verboso: bool = True,
) -> Execucao:
    """
    Roda a matriz completa: cada estratégia sobre cada pergunta, N vezes.

    Com `repeticoes > 1` e temperatura > 0, é possível medir a VARIÂNCIA de cada
    estratégia — informação que uma execução única esconde e que muda a leitura
    de diferenças pequenas.
    """
    settings = settings or SETTINGS
    k = settings.k_fewshot

    execucao = Execucao(
        modelo=settings.chat_model,
        temperatura=settings.temperature,
        k_fewshot=k,
        repeticoes=repeticoes,
        instrucao_ape=instrucao_ape,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    total = len(perguntas) * len(estrategias) * repeticoes
    feitos = 0

    for nome in estrategias:
        acertos_estrategia = 0
        n_estrategia = 0

        if verboso:
            print(f"\n--- {nome} ---")

        for pergunta in perguntas:
            for rep in range(repeticoes):
                resultado = executar_uma(
                    pergunta, nome, exemplos, settings, rep, k, instrucao_ape
                )
                execucao.resultados.append(resultado)
                feitos += 1
                n_estrategia += 1
                acertos_estrategia += int(resultado.acertou)

                if verboso:
                    if resultado.erro:
                        marca = "ERRO"
                    else:
                        marca = " OK " if resultado.acertou else "  X "
                    previa = (resultado.resposta_extraida or resultado.erro)[:34]
                    print(
                        f"  [{marca}] {pergunta.id:<9} {previa:<34} "
                        f"({feitos}/{total})"
                    )

        if verboso and n_estrategia:
            taxa = acertos_estrategia / n_estrategia
            print(f"  → {acertos_estrategia}/{n_estrategia} = {taxa:.1%}")

    return execucao


def executar_estrategia_em_subconjunto(
    perguntas: list[Pergunta],
    nome_estrategia: str,
    exemplos: list[Exemplo],
    settings: Settings | None = None,
    instrucao_ape: str = "",
) -> float:
    """
    Acurácia de uma estratégia sobre um subconjunto.

    É a função de pontuação usada pelo APE (acurácia de execução, `f_exec` no
    artigo) e pelo aprendizado da tabela de roteamento.
    """
    if not perguntas:
        return 0.0
    settings = settings or SETTINGS

    acertos = 0
    for pergunta in perguntas:
        resultado = executar_uma(
            pergunta,
            nome_estrategia,
            exemplos,
            settings,
            k=settings.k_fewshot,
            instrucao_ape=instrucao_ape,
        )
        acertos += int(resultado.acertou)
    return acertos / len(perguntas)
