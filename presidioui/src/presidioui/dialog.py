from dataclasses import dataclass
from typing import Protocol, Sequence
import json
import structlog

import click
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown


@dataclass
class DialogState:
    """State object passed between dialog steps."""

    first_name: str | None = None
    last_name: str | None = None


class DialogProtocol(Protocol):
    """Protocol defining dialog interface for both CLI and websocket."""

    def input(self, message: str) -> str | None:
        """Send message to user and get response."""
        ...

    def display(self, message: str) -> None:
        """Display message to user without expecting response."""
        ...

    def display_markdown(self, markdown: str) -> None:
        """Display markdown formatted text to user.

        Parameters
        ----------
        markdown : str
            Markdown formatted text to display
        """
        ...

    def display_table(
        self, title: str, columns: Sequence[str], rows: Sequence[dict]
    ) -> None:
        """Display tabular data to user.

        Parameters
        ----------
        title : str
            Title of the table
        columns : Sequence[str]
            Column headers
        rows : Sequence[dict]
            List of dictionaries containing row data
        """
        ...

    def log(self, event: str, *, is_markdown: bool = False, **kwargs) -> None:
        """Log a message for debugging/monitoring purposes.

        Parameters
        ----------
        event : str
            Name of the event being logged
        is_markdown : bool, optional
            Whether to render the event as markdown, by default False
        **kwargs : dict
            Additional key-value pairs to log
        """
        ...

    def set_as_loading(self) -> None:
        """Set the dialog in a loading state."""
        ...


class CLIDialog:
    """CLI implementation of dialog protocol."""

    def input(self, message: str) -> str:
        return click.prompt(message, type=str)

    def display(self, message: str) -> None:
        click.echo(message)

    def display_markdown(self, markdown: str) -> None:
        rprint(Markdown(markdown))

    def display_table(
        self, title: str, columns: Sequence[str], rows: Sequence[dict]
    ) -> None:
        table = Table(title=title)
        for col in columns:
            table.add_column(col)

        for row in rows:
            table.add_row(*[str(row.get(col, "")) for col in columns])

        rprint(table)

    def log(self, event: str, *, is_markdown: bool = False, **kwargs) -> None:
        if kwargs:
            content = []
            for key, value in kwargs.items():
                value_str = str(value)
                if is_markdown:
                    value_str = Markdown(value_str).markup
                content.append(f"[bold]{key}[/bold]\n{value_str}")

            content_str = "\n".join(content)
            rprint(
                Panel.fit(
                    f"[bold]{event}[/bold]\n\n{content_str}",
                    border_style="orange3",
                )
            )
        else:
            if is_markdown:
                rprint(Panel.fit(Markdown(event).markup, border_style="orange3"))
            else:
                rprint(Panel.fit(event, border_style="orange3"))

    def set_as_loading(self) -> None:
        # No-op for CLI - we don't show loading states
        pass


class WebsocketDialog:
    def __init__(self, ws):
        self.ws = ws
        self.logger = structlog.get_logger()

    def input(self, message: str) -> str:
        # Send message to client
        self.ws.send(json.dumps({"data": {"type": "text", "content": message}}))
        # Wait for response
        response = json.loads(self.ws.receive())
        return response["data"]

    def display(self, message: str) -> None:
        # Send message to client without expecting response
        self.ws.send(json.dumps({"data": {"type": "text", "content": message}}))

    def display_markdown(self, markdown: str) -> None:
        # Send markdown as structured message for client-side rendering
        self.ws.send(json.dumps({"data": {"type": "markdown", "content": markdown}}))

    def display_table(
        self, title: str, columns: Sequence[str], rows: Sequence[dict]
    ) -> None:
        # Send table data as structured message for client-side rendering
        table_data = {"type": "table", "title": title, "columns": columns, "rows": rows}
        self.ws.send(json.dumps({"data": table_data}))

    def log(self, event: str, *, is_markdown: bool = False, **kwargs) -> None:
        self.logger.info(event, **kwargs)

    def set_as_loading(self) -> None:
        # Send loading state message to client
        self.ws.send(json.dumps({"data": {"type": "loading", "content": True}}))


def name_dialog(dialog: DialogProtocol) -> DialogState:
    """
    Dialog that collects user's name.

    Parameters
    ----------
    dialog : DialogProtocol
        Dialog implementation (CLI or websocket)

    Returns
    -------
    DialogState
        Final dialog state with collected data
    """
    state = DialogState()

    # Get first name
    state.first_name = dialog.input("What is your first name?")

    # Get last name
    state.last_name = dialog.input("What is your last name?")

    # Print full name and end dialog
    full_name = f"{state.first_name} {state.last_name}"
    dialog.display(f"Your full name is: {full_name}")

    return state


@click.command()
def cli_dialog():
    """CLI entry point for name dialog."""
    dialog = CLIDialog()
    name_dialog(dialog)


if __name__ == "__main__":
    cli_dialog()
