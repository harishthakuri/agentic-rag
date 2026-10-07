from app.infrastructure.persistence.database import Database


class DatabaseHealthCheck:
    name = "database"

    def __init__(self, database: Database) -> None:
        self._database = database

    async def check(self) -> None:
        await self._database.ping()
