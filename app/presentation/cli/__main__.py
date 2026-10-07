"""Admin command-line interface.

uv run rag-admin api-key create --name dev
uv run rag-admin api-key list
uv run rag-admin api-key revoke <id>
"""

import argparse
import asyncio
import sys
from uuid import UUID

from app.bootstrap.container import Container
from app.core.config import get_settings


async def _create(container: Container, name: str) -> None:
    issued = await container.issue_api_key().execute(name)
    print(f"Created API key '{issued.api_key.name}' (id {issued.api_key.id})")
    print()
    print(f"    {issued.raw_key}")
    print()
    print("Store it now: it is shown only once and cannot be recovered.")


async def _list(container: Container) -> None:
    keys = await container.list_api_keys().execute()
    if not keys:
        print("No API keys.")
        return
    print(f"{'ID':36}  {'NAME':20}  {'PREFIX':12}  {'LAST USED':25}  STATUS")
    for key in keys:
        last_used = key.last_used_at.isoformat(timespec="seconds") if key.last_used_at else "never"
        status = "active" if key.is_active else "revoked"
        print(f"{key.id!s:36}  {key.name:20}  {key.display_prefix:12}  {last_used:25}  {status}")


async def _revoke(container: Container, api_key_id: UUID) -> None:
    key = await container.revoke_api_key().execute(api_key_id)
    print(f"Revoked API key '{key.name}' ({key.id})")


async def _run(args: argparse.Namespace) -> None:
    container = Container(get_settings())
    try:
        match args.action:
            case "create":
                await _create(container, args.name)
            case "list":
                await _list(container)
            case "revoke":
                await _revoke(container, args.id)
    finally:
        await container.aclose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag-admin", description="Simple RAG administration")
    resources = parser.add_subparsers(dest="resource", required=True)

    api_key = resources.add_parser("api-key", help="Manage API keys")
    actions = api_key.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create", help="Issue a new API key")
    create.add_argument("--name", required=True, help="Who or what the key is for")
    actions.add_parser("list", help="List API keys")
    revoke = actions.add_parser("revoke", help="Revoke an API key")
    revoke.add_argument("id", type=UUID)

    args = parser.parse_args(argv)
    try:
        asyncio.run(_run(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
