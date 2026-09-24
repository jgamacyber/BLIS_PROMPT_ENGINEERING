"""
As estratégias de prompt comparadas neste módulo.

Cada estratégia é uma função que recebe uma pergunta (e, quando aplicável,
exemplos de demonstração) e devolve a lista de `messages` no formato da API.
Nada é enviado aqui — a montagem do prompt é separada da execução, o que
permite inspecionar qualquer prompt sem gastar um único token
(`python main.py ver-prompt`).

Fundamentação:

- Brown et al. (2020), *Language Models are Few-Shot Learners*, define os três
  regimes que comparamos. Zero-shot: apenas uma instrução em linguagem natural
  descrevendo a tarefa, sem demonstração. One-shot: a descrição mais UMA
  demonstração. Few-shot: a descrição mais K demonstrações, com K entre 10 e 100
  no artigo, limitado pela janela de contexto. Em nenhum deles há atualização de
  pesos — é tudo aprendizado em contexto (in-context learning).

- Wei et al. (2022), *Chain-of-Thought Prompting*, mostra que incluir a cadeia de
  raciocínio nas demonstrações — triplas ⟨entrada, cadeia de pensamento, saída⟩ —
  melhora muito tarefas de raciocínio. Com PaLM 540B, o GSM8K sobe de 18% para
  57%. ATENÇÃO: é uma habilidade emergente de escala. Em modelos pequenos o
  artigo observou cadeias "fluentes porém ilógicas", com desempenho PIOR que o
  prompting padrão.

- Zhou et al. (2023), *APE*, trata a instrução como um programa a ser otimizado.
  A estratégia `ape` usa a melhor instrução descoberta por `ape.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Mensagens = list[dict[str, str]]


@dataclass
class Exemplo:
    """Uma demonstração de entrada/saída, opcionalmente com cadeia de raciocínio."""

    pergunta: str
    resposta: str
    raciocinio: str = ""
    categoria: str = ""

    def como_par(self) -> str:
        """Formato padrão de demonstração: pergunta e resposta direta."""
        return f"P: {self.pergunta}\nR: {self.resposta}"

    def como_cot(self) -> str:
        """Formato chain-of-thought: a cadeia vem ANTES da resposta final."""
        if not self.raciocinio:
            return self.como_par()
        return f"P: {self.pergunta}\nR: {self.raciocinio} A resposta é {self.resposta}."


# --------------------------------------------------------------------------- #
# Textos reaproveitados
# --------------------------------------------------------------------------- #

# Instrução mínima da tarefa. No regime zero-shot do GPT-3, é só isto que o
# modelo recebe além da pergunta.
INSTRUCAO_TAREFA = (
    "Responda à pergunta de forma correta e o mais concisa possível."
)

# Pedido de formato de saída. Sem isso, a resposta vem embrulhada em texto e a
# verificação automática fica ruidosa — o que é, por si só, um achado do
# experimento: parte do ganho de uma estratégia vem de disciplinar o FORMATO,
# não de melhorar o raciocínio.
FORMATO_FINAL = (
    "Termine sua resposta com uma última linha exatamente no formato:\n"
    "RESPOSTA: <resposta final, sem explicação>"
)


# --------------------------------------------------------------------------- #
# Estratégias
# --------------------------------------------------------------------------- #


def zero_shot(pergunta: str, exemplos: list[Exemplo] | None = None) -> Mensagens:
    """
    Zero-shot puro: só a pergunta, sem instrução e sem demonstração.

    É o piso da comparação. Serve para medir quanto cada camada seguinte
    (instrução, exemplos, persona, formato) realmente acrescenta.
    """
    return [{"role": "user", "content": pergunta}]


def zero_shot_instrucao(
    pergunta: str, exemplos: list[Exemplo] | None = None
) -> Mensagens:
    """
    Zero-shot do GPT-3: uma instrução em linguagem natural + a pergunta.

    Nenhuma demonstração. É o regime que Brown et al. descrevem como "o mais
    conveniente, mais robusto, e também o mais desafiador" — às vezes injusto,
    porque sem exemplos pode não ficar claro que FORMATO se espera.
    """
    return [
        {"role": "system", "content": f"{INSTRUCAO_TAREFA}\n\n{FORMATO_FINAL}"},
        {"role": "user", "content": f"P: {pergunta}"},
    ]


def _com_demonstracoes(
    pergunta: str,
    exemplos: list[Exemplo],
    k: int,
    usar_cot: bool = False,
) -> Mensagens:
    """Monta o bloco de demonstrações seguido da pergunta em aberto."""
    selecionados = exemplos[:k]
    formatador = (lambda e: e.como_cot()) if usar_cot else (lambda e: e.como_par())
    blocos = "\n\n".join(formatador(e) for e in selecionados)

    if usar_cot:
        instrucao = (
            "Responda às perguntas no mesmo formato dos exemplos: primeiro o "
            "raciocínio passo a passo, depois a resposta final."
        )
    else:
        instrucao = INSTRUCAO_TAREFA

    conteudo = f"{blocos}\n\nP: {pergunta}\nR:" if blocos else f"P: {pergunta}\nR:"
    return [
        {"role": "system", "content": f"{instrucao}\n\n{FORMATO_FINAL}"},
        {"role": "user", "content": conteudo},
    ]


def one_shot(
    pergunta: str, exemplos: list[Exemplo] | None = None, k: int = 1
) -> Mensagens:
    """
    One-shot: instrução + exatamente UMA demonstração.

    O parâmetro `k` é aceito por uniformidade de assinatura com as demais
    estratégias que usam exemplos, mas é ignorado de propósito: one-shot é
    definido por K=1 (Brown et al., 2020).
    """
    return _com_demonstracoes(pergunta, exemplos or [], k=1)


def few_shot(pergunta: str, exemplos: list[Exemplo] | None = None, k: int = 3) -> Mensagens:
    """Few-shot: instrução + K demonstrações, sem raciocínio explícito."""
    return _com_demonstracoes(pergunta, exemplos or [], k=k)


def persona(pergunta: str, exemplos: list[Exemplo] | None = None) -> Mensagens:
    """
    Prompt com persona: atribui um papel ao modelo antes da tarefa.

    É a técnica mais popular entre praticantes e a menos sustentada pelos
    artigos deste módulo — nenhum dos quatro a avalia. Está aqui justamente
    para ser MEDIDA em vez de assumida. Em tarefas de resposta objetiva, é comum
    que a persona não ajude, ou ajude apenas por carregar junto um pedido
    implícito de rigor.
    """
    return [
        {
            "role": "system",
            "content": (
                "Você é um especialista meticuloso, com formação em matemática e "
                "análise de dados. Você confere cada passo antes de concluir e "
                "prefere admitir incerteza a arriscar um palpite.\n\n"
                f"{INSTRUCAO_TAREFA}\n\n{FORMATO_FINAL}"
            ),
        },
        {"role": "user", "content": f"P: {pergunta}"},
    ]


def estruturada(pergunta: str, exemplos: list[Exemplo] | None = None) -> Mensagens:
    """
    Instrução estruturada: papel, tarefa, regras, formato — com delimitadores.

    Não vem de um artigo específico; é a forma que consolidou na prática de
    engenharia de prompts. A hipótese testada aqui é que a estrutura reduz a
    ambiguidade que Brown et al. apontam como a fraqueza do zero-shot: o modelo
    deixa de precisar adivinhar o formato esperado.

    Os delimitadores (### e <pergunta>) separam instrução de dado — a mesma
    ideia que, em segurança de LLM, reduz a superfície de prompt injection.
    """
    sistema = """### PAPEL
Você resolve tarefas de perguntas e respostas com precisão.

### TAREFA
Responder à pergunta delimitada por <pergunta></pergunta>.

### REGRAS
1. Analise a pergunta antes de responder.
2. Para cálculos, mostre as contas.
3. Se a pergunta for ambígua ou não puder ser respondida, diga isso \
explicitamente em vez de adivinhar.
4. Não invente dados que não estejam na pergunta.
5. Seja conciso: sem preâmbulo, sem repetir a pergunta.

### FORMATO DE SAÍDA
Raciocínio: <breve, no máximo 3 frases>
RESPOSTA: <resposta final, sem explicação>"""

    return [
        {"role": "system", "content": sistema},
        {"role": "user", "content": f"<pergunta>{pergunta}</pergunta>"},
    ]


def cot_zero_shot(pergunta: str, exemplos: list[Exemplo] | None = None) -> Mensagens:
    """
    Chain-of-thought sem demonstrações: basta pedir o passo a passo.

    Variante zero-shot do CoT (Kojima et al., 2022, citada pelo APE). É muito
    mais barata que o CoT few-shot, porque não gasta contexto com exemplos.

    O APE (Zhou et al., 2023) mostra que o gatilho pode ser otimizado
    automaticamente — a frase clássica "vamos pensar passo a passo" não é a
    melhor possível. Use `python main.py ape` para procurar uma melhor.
    """
    return [
        {
            "role": "system",
            "content": (
                "Responda à pergunta raciocinando passo a passo antes de concluir.\n\n"
                f"{FORMATO_FINAL}"
            ),
        },
        {"role": "user", "content": f"P: {pergunta}\nR: Vamos pensar passo a passo."},
    ]


def cot_few_shot(
    pergunta: str, exemplos: list[Exemplo] | None = None, k: int = 3
) -> Mensagens:
    """
    Chain-of-thought com demonstrações (Wei et al., 2022).

    Cada demonstração é uma tripla ⟨entrada, cadeia de pensamento, saída⟩. É a
    forma original do artigo, que usa 8 exemplares escritos à mão.

    Só funciona se os exemplos tiverem o campo `raciocinio` preenchido; sem ele,
    degrada para few-shot comum.
    """
    return _com_demonstracoes(pergunta, exemplos or [], k=k, usar_cot=True)


def ape(
    pergunta: str,
    exemplos: list[Exemplo] | None = None,
    instrucao: str = "",
) -> Mensagens:
    """
    Estratégia com instrução descoberta automaticamente pelo APE.

    A instrução vem de `ape.py`, que a otimiza por busca: propõe candidatas a
    partir de demonstrações, pontua cada uma pela acurácia de execução e mantém
    as melhores.
    """
    if not instrucao:
        instrucao = INSTRUCAO_TAREFA
    return [
        {"role": "system", "content": f"{instrucao}\n\n{FORMATO_FINAL}"},
        {"role": "user", "content": f"P: {pergunta}"},
    ]


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


@dataclass
class Estrategia:
    """Metadados de uma estratégia, para os relatórios."""

    nome: str
    funcao: Callable[..., Mensagens]
    descricao: str
    usa_exemplos: bool = False
    artigo: str = ""
    parametros: dict = field(default_factory=dict)

    def montar(
        self, pergunta: str, exemplos: list[Exemplo] | None = None, **extra
    ) -> Mensagens:
        kwargs = {**self.parametros, **extra}
        if not self.usa_exemplos:
            kwargs.pop("k", None)
        return self.funcao(pergunta, exemplos, **kwargs)


ESTRATEGIAS: dict[str, Estrategia] = {
    "zero_shot": Estrategia(
        nome="zero_shot",
        funcao=zero_shot,
        descricao="Só a pergunta, sem instrução nem exemplos (piso da comparação)",
        artigo="Brown et al. (2020)",
    ),
    "zero_shot_instrucao": Estrategia(
        nome="zero_shot_instrucao",
        funcao=zero_shot_instrucao,
        descricao="Instrução em linguagem natural, sem demonstrações",
        artigo="Brown et al. (2020)",
    ),
    "one_shot": Estrategia(
        nome="one_shot",
        funcao=one_shot,
        descricao="Instrução + 1 demonstração",
        usa_exemplos=True,
        artigo="Brown et al. (2020)",
    ),
    "few_shot": Estrategia(
        nome="few_shot",
        funcao=few_shot,
        descricao="Instrução + K demonstrações (in-context learning)",
        usa_exemplos=True,
        artigo="Brown et al. (2020)",
    ),
    "persona": Estrategia(
        nome="persona",
        funcao=persona,
        descricao="Papel atribuído ao modelo antes da tarefa",
        artigo="— (prática comum, não avaliada pelos artigos)",
    ),
    "estruturada": Estrategia(
        nome="estruturada",
        funcao=estruturada,
        descricao="Papel, tarefa, regras e formato, com delimitadores",
        artigo="— (prática consolidada)",
    ),
    "cot_zero_shot": Estrategia(
        nome="cot_zero_shot",
        funcao=cot_zero_shot,
        descricao="Pede raciocínio passo a passo, sem demonstrações",
        artigo="Kojima et al. (2022); otimizável por APE",
    ),
    "cot_few_shot": Estrategia(
        nome="cot_few_shot",
        funcao=cot_few_shot,
        descricao="Demonstrações com cadeia de raciocínio ⟨entrada, cadeia, saída⟩",
        usa_exemplos=True,
        artigo="Wei et al. (2022)",
    ),
    "ape": Estrategia(
        nome="ape",
        funcao=ape,
        descricao="Instrução descoberta automaticamente por busca",
        artigo="Zhou et al. (2023)",
    ),
}

# Estratégias padrão da comparação. `ape` fica de fora porque depende de uma
# instrução previamente otimizada (rode `python main.py ape` antes).
PADRAO = [
    "zero_shot",
    "zero_shot_instrucao",
    "one_shot",
    "few_shot",
    "persona",
    "estruturada",
    "cot_zero_shot",
    "cot_few_shot",
]


def obter(nome: str) -> Estrategia:
    if nome not in ESTRATEGIAS:
        raise ValueError(
            f"Estratégia '{nome}' desconhecida. "
            f"Disponíveis: {', '.join(ESTRATEGIAS)}"
        )
    return ESTRATEGIAS[nome]


def texto_do_prompt(mensagens: Mensagens) -> str:
    """Renderiza as mensagens como texto legível, para inspeção e depuração."""
    partes = []
    for m in mensagens:
        partes.append(f"[{m['role'].upper()}]\n{m['content']}")
    return "\n\n".join(partes)


def tamanho_aproximado(mensagens: Mensagens) -> int:
    """
    Estimativa grosseira de tokens: ~4 caracteres por token.

    Serve para comparar o CUSTO relativo das estratégias. Few-shot e CoT few-shot
    gastam muito mais contexto que zero-shot — um ganho de acurácia precisa
    justificar esse custo.
    """
    return sum(len(m["content"]) for m in mensagens) // 4
