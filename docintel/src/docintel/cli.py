"""Command-line interface for DocIntel."""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from docintel.config import settings
from docintel.document_processor import DocumentProcessor
from docintel.vector_store import VectorStore
from docintel.rag_engine import RAGEngine
from docintel.models import QueryRequest

app = typer.Typer(help="DocIntel - Document Intelligence Platform")
console = Console()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@app.command()
def upload(
    file_path: Path = typer.Argument(..., help="Path to the document to upload"),
):
    """Upload and index a document."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Processing document:[/cyan] {file_path.name}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing document...", total=None)

            # Initialize services
            processor = DocumentProcessor()
            vector_store = VectorStore()

            # Process document
            with open(file_path, "rb") as f:
                document, chunks = processor.process_document(f, file_path.name)

            progress.update(task, description="Indexing chunks...")

            # Index chunks
            indexed_count = vector_store.index_chunks(chunks)

            progress.update(task, description="Complete!")

        console.print(f"\n[green]Success![/green]")
        console.print(f"Document ID: {document.id}")
        console.print(f"Chunks indexed: {indexed_count}")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask about your documents"),
    max_results: int = typer.Option(5, "--max-results", "-n", help="Maximum number of chunks to retrieve"),
):
    """Query your indexed documents."""
    console.print(f"\n[cyan]Question:[/cyan] {question}\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching documents...", total=None)

            # Initialize services
            vector_store = VectorStore()
            rag_engine = RAGEngine(vector_store=vector_store)

            # Execute query
            request = QueryRequest(query=question, max_results=max_results)
            response = rag_engine.query(request)

            progress.update(task, description="Complete!")

        # Display answer
        console.print(Panel(response.answer, title="[bold green]Answer[/bold green]", border_style="green"))

        # Display sources
        if response.sources:
            console.print("\n[cyan]Sources:[/cyan]")
            for i, source in enumerate(response.sources, 1):
                console.print(f"\n[bold]{i}.[/bold] [dim](Relevance: {source['relevance_score']:.2f})[/dim]")
                console.print(f"   {source['content_preview']}")

        console.print(f"\n[dim]Processing time: {response.processing_time:.2f}s[/dim]")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats():
    """Show system statistics."""
    try:
        vector_store = VectorStore()
        chunk_count = vector_store.count_documents()

        table = Table(title="DocIntel Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Chunks", str(chunk_count))
        table.add_row("Collection", settings.qdrant_collection_name)
        table.add_row("Embedding Model", settings.embedding_model)
        table.add_row("LLM Provider", settings.llm_provider)
        table.add_row("LLM Model", settings.llm_model)
        table.add_row("Vector DB", f"{settings.qdrant_host}:{settings.qdrant_port}")

        console.print("\n")
        console.print(table)
        console.print("\n")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host", help="API host"),
    port: int = typer.Option(settings.api_port, "--port", help="API port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the API server."""
    import uvicorn

    console.print(f"\n[cyan]Starting DocIntel API server...[/cyan]")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Docs: http://{host}:{port}/docs\n")

    uvicorn.run(
        "docintel.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset the vector database (delete all documents)."""
    if not confirm:
        proceed = typer.confirm("⚠️  This will delete ALL documents. Are you sure?", abort=True)

    try:
        vector_store = VectorStore()
        vector_store.reset_collection()
        console.print("[green]Collection reset successfully[/green]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def health():
    """Check system health."""
    try:
        vector_store = VectorStore()
        is_healthy = vector_store.health_check()

        if is_healthy:
            console.print("[green]✓[/green] System is healthy")
            console.print(f"  Vector DB: Connected ({settings.qdrant_host}:{settings.qdrant_port})")
            console.print(f"  LLM Provider: {settings.llm_provider}")
        else:
            console.print("[red]✗[/red] System is unhealthy")
            console.print(f"  Vector DB: Disconnected")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
