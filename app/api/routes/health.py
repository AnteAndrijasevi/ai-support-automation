from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness/readiness check")
async def health() -> dict[str, str]:
    return {"status": "ok"}
