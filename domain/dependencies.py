from fastapi import Request
from domain.container import DependencyContainer

def get_container(request: Request) -> DependencyContainer:
    return request.app.state.container