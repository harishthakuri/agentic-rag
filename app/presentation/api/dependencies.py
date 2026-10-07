"""FastAPI dependencies: resolve use cases from the composition root."""

from typing import Annotated

from fastapi import Depends, Request

from app.application.use_cases.system.check_readiness import CheckReadiness
from app.bootstrap.container import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_check_readiness(container: ContainerDep) -> CheckReadiness:
    return container.check_readiness()


CheckReadinessDep = Annotated[CheckReadiness, Depends(get_check_readiness)]
