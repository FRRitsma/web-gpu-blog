from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    input_size: tuple[int, int] = (224, 224)
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    ...


settings = Settings()
