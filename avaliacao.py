"""
Extração e verificação das respostas.

A verificação é **determinística** por padrão — comparação exata, numérica,
regex ou validação de JSON. Não usa LLM como juiz na métrica principal, por
três motivos: é de graça, é reprodutível, e um juiz LLM introduz o próprio viés
naquilo que estamos tentando medir.

Há um juiz por LLM opcional (`julgar_com_llm`) para perguntas abertas, onde não
existe gabarito exato. Ele é marcado como tal nos relatórios.

O ponto delicado é a EXTRAÇÃO. Uma estratégia pode acertar o raciocínio e ainda
assim "errar" a métrica por embrulhar a resposta em texto. Por isso todas as
estratégias (menos o zero-shot puro, de propósito) pedem uma última linha
`RESPOSTA: <x>`, e o extrator tem fallbacks em cascata para ser justo com quem
não obedeceu ao formato.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from config import SETTINGS, get_client


# --------------------------------------------------------------------------- #
# Normalização
# --------------------------------------------------------------------------- #


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos, sem pontuação de borda, espaços colapsados."""
    texto = remover_acentos(str(texto).lower().strip())
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip(" .,;:!?\"'`*()[]{}")


def normalizar_numero(texto: str) -> float | None:
    """
    Extrai um número de um texto, tolerando formatos pt-BR e en-US.

    'R$ 1.234,56' -> 1234.56 ; '1,234.56' -> 1234.56 ; '11 maçãs' -> 11.0
    """
    if texto is None:
        return None
    t = str(texto)
    # Remove símbolos de moeda, percentuais e espaços não separadores.
    t = re.sub(r"[R$€£%\s ]", "", t)

    match = re.search(r"-?\d[\d.,]*", t)
    if not match:
        return None
    bruto = match.group(0)

    tem_ponto, tem_virgula = "." in bruto, "," in bruto
    if tem_ponto and tem_virgula:
        # O separador decimal é o que aparece por último.
        if bruto.rfind(",") > bruto.rfind("."):
            bruto = bruto.replace(".", "").replace(",", ".")
        else:
            bruto = bruto.replace(",", "")
    elif tem_virgula:
        # Vírgula isolada: decimal se separar 1-2 dígitos finais, senão milhar.
        partes = bruto.split(",")
        bruto = (
            bruto.replace(",", ".")
            if len(partes[-1]) <= 2 and len(partes) == 2
            else bruto.replace(",", "")
        )
    elif tem_ponto:
        partes = bruto.split(".")
        # '1.234' com 3 dígitos após o ponto é separador de milhar.
        if len(partes) > 2 or (len(partes) == 2 and len(partes[-1]) == 3):
            bruto = bruto.replace(".", "")

    try:
        return float(bruto)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Extração da resposta final
# --------------------------------------------------------------------------- #

_PADROES_RESPOSTA = [
    re.compile(r"RESPOSTA\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\ba\s+resposta\s+(?:final\s+)?(?:é|e)\s*:?\s*(.+?)\s*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^R\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
]


def extrair_resposta(saida: str) -> str:
    """
    Extrai a resposta final da saída do modelo, em cascata:

    1. Linha `RESPOSTA: x` (o formato que pedimos)
    2. "a resposta é x"
    3. Linha `R: x`
    4. Última linha não vazia

    Sempre devolve a ÚLTIMA ocorrência, porque o modelo pode citar o formato no
    meio do raciocínio antes de concluir.
    """
    if not saida:
        return ""
    texto = saida.strip()

    for padrao in _PADROES_RESPOSTA:
        achados = padrao.findall(texto)
        if achados:
            return achados[-1].strip().strip("*` ")

    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    return linhas[-1].strip("*` ") if linhas else ""


# --------------------------------------------------------------------------- #
# Verificadores
# --------------------------------------------------------------------------- #


def verificar_exato(resposta: str, esperado) -> bool:
    """Igualdade após normalização. Aceita lista de alternativas aceitáveis."""
    alternativas = esperado if isinstance(esperado, list) else [esperado]
    alvo = normalizar(resposta)
    return any(normalizar(a) == alvo for a in alternativas)


def verificar_contem(resposta: str, esperado) -> bool:
    """O esperado aparece na resposta. Com lista, basta uma alternativa."""
    alternativas = esperado if isinstance(esperado, list) else [esperado]
    alvo = normalizar(resposta)
    return any(normalizar(a) in alvo for a in alternativas)


def verificar_numerico(resposta: str, esperado, tolerancia: float = 1e-6) -> bool:
    """Compara valores numéricos, ignorando formatação e unidades."""
    obtido = normalizar_numero(resposta)
    alvo = normalizar_numero(esperado)
    if obtido is None or alvo is None:
        return False
    return abs(obtido - alvo) <= max(tolerancia, abs(alvo) * 1e-9)


def verificar_regex(resposta: str, esperado) -> bool:
    """O esperado é um padrão regex que deve casar com a resposta."""
    try:
        return bool(re.search(str(esperado), resposta, re.IGNORECASE | re.DOTALL))
    except re.error:
        return False


def verificar_json(resposta: str, esperado) -> bool:
    """
    Valida que a resposta contém um JSON com as chaves esperadas.

    `esperado` é a lista de chaves obrigatórias. Verifica estrutura, não valores
    — o objetivo é medir aderência ao FORMATO pedido.
    """
    match = re.search(r"\{.*\}", resposta, re.DOTALL)
    if not match:
        return False
    try:
        dados = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False
    if not isinstance(dados, dict):
        return False

    chaves = esperado if isinstance(esperado, list) else [esperado]
    return all(str(c) in dados for c in chaves)


VERIFICADORES = {
    "exato": verificar_exato,
    "contem": verificar_contem,
    "numerico": verificar_numerico,
    "regex": verificar_regex,
    "json": verificar_json,
}


def verificar(saida: str, esperado, tipo: str = "exato") -> tuple[bool, str]:
    """
    Verifica a saída bruta do modelo contra o gabarito.

    Devolve (acertou, resposta_extraída). Para os tipos `json` e `regex` a
    verificação usa a saída INTEIRA, porque o que se avalia é a estrutura do
    texto produzido, não uma resposta final isolada.
    """
    if tipo not in VERIFICADORES:
        raise ValueError(
            f"Tipo de verificação '{tipo}' desconhecido. "
            f"Disponíveis: {', '.join(VERIFICADORES)}"
        )

    if tipo in {"json", "regex"}:
        return VERIFICADORES[tipo](saida or "", esperado), (saida or "").strip()

    extraida = extrair_resposta(saida)
    acertou = VERIFICADORES[tipo](extraida, esperado)

    # Segunda chance: o modelo pode ter acertado no corpo mas errado o formato
    # da última linha. Não queremos punir a estratégia pelo extrator.
    if not acertou and saida:
        acertou = VERIFICADORES[tipo](saida, esperado) if tipo != "exato" else acertou
    return acertou, extraida


# --------------------------------------------------------------------------- #
# Juiz por LLM (opcional, para perguntas abertas)
# --------------------------------------------------------------------------- #

PROMPT_JUIZ = """Avalie se a RESPOSTA do modelo está correta, comparando-a com o \
GABARITO.

PERGUNTA: {pergunta}

GABARITO: {gabarito}

RESPOSTA DO MODELO: {resposta}

A resposta está correta se transmite a mesma informação do gabarito, mesmo com \
palavras diferentes. Diferenças de formatação, ordem ou verbosidade não tornam \
a resposta incorreta. Informação factual divergente torna.

Responda APENAS com um JSON: {{"correta": true/false, "motivo": "uma frase"}}"""


@dataclass
class Julgamento:
    correta: bool
    motivo: str = ""


def julgar_com_llm(
    pergunta: str, resposta: str, gabarito: str, settings=None
) -> Julgamento:
    """
    Juiz por LLM, para perguntas abertas sem gabarito exato.

    Usado apenas em itens marcados com `verificacao: "llm"`. Custa uma chamada
    por item avaliado e introduz o viés do próprio modelo na métrica — por isso
    os relatórios separam esses itens dos verificados deterministicamente.
    """
    settings = settings or SETTINGS
    try:
        client = get_client(settings)
        resposta_api = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_JUIZ.format(
                        pergunta=pergunta, gabarito=gabarito, resposta=resposta
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=150,
        )
        texto = resposta_api.choices[0].message.content or ""
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            dados = json.loads(match.group(0))
            return Julgamento(
                correta=bool(dados.get("correta", False)),
                motivo=str(dados.get("motivo", "")),
            )
    except Exception as erro:  # noqa: BLE001
        return Julgamento(False, f"falha ao julgar: {erro}")
    return Julgamento(False, "resposta do juiz não interpretável")
