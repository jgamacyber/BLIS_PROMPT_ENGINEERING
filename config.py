"""
Configuração central — cliente OpenRouter e parâmetros do experimento.

Todos os valores podem ser sobrescritos pelo .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    # --- Credenciais e modelo ---
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    chat_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "openai/gpt-4o-mini")
    )

    # --- Parâmetros de geração ---
    # Temperatura 0 é o padrão para comparar estratégias: queremos medir o efeito
    # do PROMPT, não a variação da amostragem. Para medir variância, use
    # --repeticoes com temperatura > 0.
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "700")))

    # --- Experimento ---
    # Quantos exemplos usar no few-shot. O GPT-3 usa K entre 10 e 100; aqui o
    # padrão é 3 porque o conjunto de demonstração é pequeno e o custo importa.
    k_fewshot: int = field(default_factory=lambda: int(os.getenv("K_FEWSHOT", "3")))
    repeticoes: int = field(default_factory=lambda: int(os.getenv("REPETICOES", "1")))

    # --- APE (Automatic Prompt Engineer) ---
    ape_candidatos: int = field(
        default_factory=lambda: int(os.getenv("APE_CANDIDATOS", "6"))
    )
    ape_temperatura: float = field(
        default_factory=lambda: float(os.getenv("APE_TEMPERATURA", "0.9"))
    )
    ape_rodadas: int = field(default_factory=lambda: int(os.getenv("APE_RODADAS", "2")))
    ape_top_k: int = field(default_factory=lambda: int(os.getenv("APE_TOP_K", "2")))

    @property
    def tem_chave(self) -> bool:
        return bool(self.api_key)


SETTINGS = Settings()


def get_client(settings: Settings | None = None) -> OpenAI:
    """
    Cliente OpenAI apontando para a OpenRouter.

    Levanta um erro claro se a chave não estiver configurada, em vez de falhar
    com um 401 opaco no meio do experimento.
    """
    settings = settings or SETTINGS
    if not settings.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha a chave. Os testes offline (`python testes_offline.py`) e "
            "a inspeção de prompts (`python main.py ver-prompt`) rodam sem chave."
        )
    return OpenAI(
        api_key=settings.api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/jgamacyber",
            "X-Title": "BLIS-Prompt-Engineering",
        },
    )


def resumo_config(settings: Settings | None = None) -> str:
    s = settings or SETTINGS
    return (
        f"modelo={s.chat_model} | temp={s.temperature} | k_fewshot={s.k_fewshot} "
        f"| repetições={s.repeticoes} | api_key={'OK' if s.tem_chave else 'AUSENTE'}"
    )
