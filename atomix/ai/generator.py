"""Natural language to simulation setup generator."""

from pathlib import Path
from typing import Any

from ase import Atoms

from atomix.ai.prompts import SYSTEM_PROMPT


class NLGenerator:
    """Generate simulation inputs from natural language descriptions.

    Parameters
    ----------
    provider : str
        LLM provider ('anthropic', 'openai', 'ollama').
    model : str
        Model identifier.
    api_key : str | None
        API key (if not set via environment).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client: Any = None

    def _init_client(self) -> None:
        """Initialize LLM client."""
        if self.provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        elif self.provider == "openai":
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        atoms: Atoms | None = None,
    ) -> dict[str, Any]:
        """Generate simulation setup from natural language.

        Parameters
        ----------
        prompt : str
            Natural language description of desired simulation.
        context : dict | None
            Additional context (existing parameters, etc.).
        atoms : Atoms | None
            Current structure for context.

        Returns
        -------
        dict[str, Any]
            Generated simulation parameters and files.
        """
        raise NotImplementedError

    def _build_messages(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        atoms: Atoms | None,
    ) -> list[dict[str, str]]:
        """Build message list for LLM."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        user_content = prompt
        if atoms is not None:
            user_content += f"\n\nCurrent structure: {len(atoms)} atoms, {atoms.get_chemical_formula()}"
        if context:
            user_content += f"\n\nContext: {context}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def load_docs(self, doc_path: Path | str) -> str:
        """Load local documentation for RAG context."""
        path = Path(doc_path)
        if path.is_dir():
            docs = []
            for f in path.glob("*.md"):
                docs.append(f.read_text())
            return "\n\n".join(docs)
        return path.read_text()
