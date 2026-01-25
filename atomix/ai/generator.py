"""Natural language to simulation setup generator."""

import json
import re
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
        self._docs: str = ""

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
        if self._client is None:
            self._init_client()

        messages = self._build_messages(prompt, context, atoms)
        response_text = self._call_llm(messages)
        return self._parse_response(response_text)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call LLM API and return response text."""
        if self.provider == "anthropic":
            # Anthropic uses system as separate parameter
            system_content = messages[0]["content"]
            user_messages = messages[1:]

            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_content,
                messages=user_messages,
            )
            return response.content[0].text

        elif self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM response to extract JSON."""
        # Try to extract JSON from response
        # Handle case where LLM wraps JSON in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"Could not find JSON in response: {response_text[:200]}...")

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        # Validate required fields
        if "incar" not in result:
            raise ValueError("Response missing required 'incar' field")

        # Set defaults for optional fields
        result.setdefault("kpoints", {"type": "automatic", "grid": [1, 1, 1]})
        result.setdefault("potcar", {})
        result.setdefault("structure", {"action": "use_provided"})
        result.setdefault("constraints", {})
        result.setdefault("warnings", [])
        result.setdefault("calc_type", "static")

        return result

    def _build_messages(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        atoms: Atoms | None,
    ) -> list[dict[str, str]]:
        """Build message list for LLM."""
        # Build system prompt with optional docs
        system_content = SYSTEM_PROMPT
        if self._docs:
            system_content += f"\n\n## Reference Documentation\n\n{self._docs}"

        messages = [{"role": "system", "content": system_content}]

        user_content = prompt
        if atoms is not None:
            formula = atoms.get_chemical_formula()
            cell = atoms.get_cell()
            pbc = atoms.get_pbc()
            user_content += f"\n\nCurrent structure: {len(atoms)} atoms, formula: {formula}"
            user_content += f"\nCell: {cell[0]:.2f} x {cell[1]:.2f} x {cell[2]:.2f} Å"
            user_content += f"\nPeriodic: {pbc}"
        if context:
            user_content += f"\n\nContext: {context}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def load_docs(self, doc_path: Path | str) -> None:
        """Load local documentation for RAG context.

        Parameters
        ----------
        doc_path : Path | str
            Path to markdown file or directory containing .md files.
        """
        path = Path(doc_path)
        if path.is_dir():
            docs = []
            for f in sorted(path.glob("*.md")):
                docs.append(f.read_text())
            self._docs = "\n\n".join(docs)
        else:
            self._docs = path.read_text()
