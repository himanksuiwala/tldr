import typer
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

def main(name: str):
    print(f"Hello, {name}!")
    
if __name__ == "__main__":
    typer.run(main)