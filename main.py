"""Local-only Groq budget demo. The in-memory ledger is not durable accounting."""

import asyncio
import os
import traceback
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from groq import AsyncGroq
from pydantic import BaseModel, Field


BUDGET_MICROS = 100_000  # $0.10
MAX_COMPLETION_TOKENS = 256
MAX_MESSAGE_CHARS = 6_000
SYSTEM_MESSAGE = "Reply helpfully and concisely. This is a local budget-ledger demo."
TOKEN_OVERHEAD_UPPER_BOUND = 64


def env_price(name: str, default: str) -> Decimal:
    try:
        value = Decimal(os.getenv(name, default))
    except InvalidOperation as exc:
        raise RuntimeError(f"{name} must be a decimal price per million tokens") from exc
    if value < 0:
        raise RuntimeError(f"{name} must not be negative")
    return value


INPUT_USD_PER_MILLION = env_price("GROQ_INPUT_USD_PER_MILLION", "1.00")
OUTPUT_USD_PER_MILLION = env_price("GROQ_OUTPUT_USD_PER_MILLION", "1.00")
MODEL = "openai/gpt-oss-20b"


def cost_micros(prompt_tokens: int, completion_tokens: int) -> int:
    """Round upward to whole microdollars so accounting never rounds down."""
    micros = (
        Decimal(prompt_tokens) * INPUT_USD_PER_MILLION
        + Decimal(completion_tokens) * OUTPUT_USD_PER_MILLION
    )
    return int(micros.to_integral_value(rounding=ROUND_CEILING))


def reservation_for(message: str) -> int:
    # A token cannot contain more information than its UTF-8 bytes. Counting every
    # byte as a token, plus fixed framing overhead, is intentionally conservative.
    prompt_upper_bound = (
        len(message.encode("utf-8"))
        + len(SYSTEM_MESSAGE.encode("utf-8"))
        + TOKEN_OVERHEAD_UPPER_BOUND
    )
    return cost_micros(prompt_upper_bound, MAX_COMPLETION_TOKENS)


@dataclass
class Account:
    spent_micros: int = 0
    reserved_micros: int = 0


class BudgetExceeded(Exception):
    pass


class BudgetLedger:
    """Process-local ledger guarded by one lock for atomic check-and-reserve."""

    def __init__(self, budget_micros: int = BUDGET_MICROS):
        self.budget_micros = budget_micros
        self._accounts: dict[str, Account] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, session_id: str, amount_micros: int) -> None:
        async with self._lock:
            account = self._accounts.setdefault(session_id, Account())
            if account.spent_micros + account.reserved_micros + amount_micros > self.budget_micros:
                raise BudgetExceeded
            account.reserved_micros += amount_micros

    async def settle(self, session_id: str, reserved_micros: int, charged_micros: int) -> None:
        async with self._lock:
            account = self._accounts[session_id]
            account.reserved_micros -= reserved_micros
            account.spent_micros += charged_micros

    async def snapshot(self, session_id: str) -> dict[str, int]:
        async with self._lock:
            account = self._accounts.setdefault(session_id, Account())
            return {
                "spent_micros": account.spent_micros,
                "reserved_micros": account.reserved_micros,
                "remaining_micros": max(
                    0,
                    self.budget_micros - account.spent_micros - account.reserved_micros,
                ),
                "budget_micros": self.budget_micros,
            }


def trustworthy_usage(response: Any, reserved_micros: int) -> int | None:
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if (
        isinstance(prompt, bool)
        or isinstance(completion, bool)
        or not isinstance(prompt, int)
        or not isinstance(completion, int)
        or prompt < 0
        or completion < 0
    ):
        return None
    actual = cost_micros(prompt, completion)
    return actual if actual <= reserved_micros else None


async def run_chat(
    *,
    ledger: BudgetLedger,
    client: Any,
    session_id: str,
    message: str,
    reservation_micros: int,
) -> dict[str, Any]:
    await ledger.reserve(session_id, reservation_micros)
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": message},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            stream=False,
        )
    except Exception as e:
        print(f"Groq error [{type(e).__name__}]: {e!r}", flush=True)
        traceback.print_exc()
        # The provider may have accepted/billed the request before an error reached
        # us. Keep the full reservation charged because actual usage is unknown.
        await ledger.settle(session_id, reservation_micros, reservation_micros)
        raise

    actual = trustworthy_usage(response, reservation_micros)
    usage_estimated = actual is None
    charged = reservation_micros if usage_estimated else actual
    await ledger.settle(session_id, reservation_micros, charged)
    snapshot = await ledger.snapshot(session_id)
    content = response.choices[0].message.content or ""
    return {"reply": content, "usage_estimated": usage_estimated, **snapshot}


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


app = FastAPI(title="Ten-cent Groq demo")
ledger = BudgetLedger()


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/spend/{session_id}")
async def spend(session_id: str) -> dict[str, int]:
    if not 8 <= len(session_id) <= 128 or not all(c.isalnum() or c in "_-" for c in session_id):
        raise HTTPException(status_code=422, detail="Invalid session ID")
    return await ledger.snapshot(session_id)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Set GROQ_API_KEY before sending a message")

    reservation = reservation_for(request.message)
    try:
        return await run_chat(
            ledger=ledger,
            client=AsyncGroq(api_key=api_key),
            session_id=request.session_id,
            message=request.message,
            reservation_micros=reservation,
        )
    except BudgetExceeded:
        snapshot = await ledger.snapshot(request.session_id)
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Request blocked: its worst-case cost exceeds the session budget.",
                **snapshot,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Groq did not return a trustworthy result; the reservation remains charged.",
        ) from exc
