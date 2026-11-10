"""
Display formatters for Venice AI models
"""

import json
from datetime import datetime
from typing import Any

from rich.panel import Panel
from rich.table import Table


class ModelFormatter:
    """Handles model display formatting"""

    @staticmethod
    def format_price(pricing: Any, currency: str = "both") -> str:
        """Format pricing information"""
        if not pricing:
            return "N/A"

        usd = f"${pricing.usd:.2f}" if hasattr(pricing, "usd") else "N/A"
        diem = f"Ð{pricing.diem:.2f}" if getattr(pricing, "diem", None) is not None else "N/A"

        if currency == "usd":
            return usd
        elif currency == "diem":
            return diem
        else:  # both
            return f"{usd} / {diem}"

    @staticmethod
    def format_gen_price(pricing: Any, currency: str = "both") -> str:
        """Format generation pricing (for images)"""
        if not pricing:
            return "N/A"

        usd = f"${pricing.usd:.3f}" if hasattr(pricing, "usd") else "N/A"
        diem = f"Ð{pricing.diem:.3f}" if getattr(pricing, "diem", None) is not None else "N/A"

        if currency == "usd":
            return usd
        elif currency == "diem":
            return diem
        else:  # both
            return f"{usd} / {diem}"

    @staticmethod
    def format_capabilities(capabilities: Any) -> str:
        """Format capabilities as icons"""
        if not capabilities:
            return ""

        icons = []
        if getattr(capabilities, "supportsFunctionCalling", False):
            icons.append("🔧")
        if getattr(capabilities, "supportsVision", False):
            icons.append("👁️")
        if getattr(capabilities, "supportsReasoning", False):
            icons.append("🧠")
        if getattr(capabilities, "supportsWebSearch", False):
            icons.append("🌐")
        if getattr(capabilities, "optimizedForCode", False):
            icons.append("💻")
        if getattr(capabilities, "supportsResponseSchema", False):
            icons.append("📝")

        return " ".join(icons)

    @staticmethod
    def format_context(tokens: int | None) -> str:
        """Format context window size"""
        if not tokens:
            return "N/A"

        if tokens >= 100000:
            return f"{int(tokens // 1000)}k"
        else:
            return f"{int(tokens):,}"

    @staticmethod
    def get_capability_legend() -> str:
        """Get the capability icon legend"""
        return "🔑 Legend: 🔧 Functions | 👁️ Vision | 🧠 Reasoning | 🌐 Web | 💻 Code | 📝 Schema"

    @classmethod
    def format_text_table(cls, models: list[Any], currency: str = "both") -> Table:
        """Format text models in table"""
        table = Table(
            title=f"TEXT MODELS ({len(models)} available)",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            padding=(0, 1),
        )

        table.add_column("Model Name\nID: model-id", style="cyan", no_wrap=False)
        table.add_column("Context", style="green")
        table.add_column(
            f"Pricing ({currency.upper() if currency != 'both' else 'USD / DIEM'})\nper 1M tokens",
            style="blue",
        )
        table.add_column("Features", style="magenta")

        for model in sorted(models, key=lambda x: x.id):
            # Format name with ID and traits
            name_parts = []
            if hasattr(model.model_spec, "name") and model.model_spec.name:
                name_parts.append(model.model_spec.name)
            else:
                name_parts.append(model.id)

            name_parts.append(f"[dim]{model.id}[/dim]")

            if hasattr(model.model_spec, "traits") and model.model_spec.traits:
                traits_str = ", ".join(model.model_spec.traits)
                name_parts.append(f"🏷️  [italic]{traits_str}[/italic]")

            name_col = "\n".join(name_parts)

            # Context
            context = cls.format_context(getattr(model.model_spec, "availableContextTokens", None))

            # Pricing
            pricing_lines = []
            if hasattr(model.model_spec, "pricing") and model.model_spec.pricing:
                if hasattr(model.model_spec.pricing, "input") and model.model_spec.pricing.input:
                    pricing_lines.append(
                        f"🔹 In: {cls.format_price(model.model_spec.pricing.input, currency)}"
                    )
                if hasattr(model.model_spec.pricing, "output") and model.model_spec.pricing.output:
                    pricing_lines.append(
                        f"🔸 Out: {cls.format_price(model.model_spec.pricing.output, currency)}"
                    )
            pricing_col = "\n".join(pricing_lines) if pricing_lines else "N/A"

            # Capabilities
            caps = ""
            if hasattr(model.model_spec, "capabilities"):
                caps = cls.format_capabilities(model.model_spec.capabilities)

            table.add_row(name_col, context, pricing_col, caps)

        return table

    @classmethod
    def format_image_table(cls, models: list[Any], currency: str = "both") -> Table:
        """Format image models in table"""
        table = Table(
            title=f"IMAGE MODELS ({len(models)} available)",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            padding=(0, 1),
        )

        table.add_column("Model Name\nID: model-id", style="cyan", no_wrap=False)
        table.add_column("Steps", style="green")
        table.add_column(
            f"Generation Price\n({currency.upper() if currency != 'both' else 'USD / DIEM'})",
            style="blue",
        )
        table.add_column("Status", style="yellow")

        for model in sorted(models, key=lambda x: x.id):
            # Name
            name_parts = []
            if hasattr(model.model_spec, "name") and model.model_spec.name:
                name_parts.append(model.model_spec.name)
            else:
                name_parts.append(model.id)

            name_parts.append(f"[dim]{model.id}[/dim]")

            if hasattr(model.model_spec, "traits") and model.model_spec.traits:
                traits_str = ", ".join(model.model_spec.traits)
                name_parts.append(f"🏷️  [italic]{traits_str}[/italic]")

            name_col = "\n".join(name_parts)

            # Steps
            steps = "N/A"
            if hasattr(model.model_spec, "constraints") and model.model_spec.constraints:
                constraints = model.model_spec.constraints
                if hasattr(constraints, "steps"):
                    step_config = constraints.steps
                    if hasattr(step_config, "default") and hasattr(step_config, "max"):
                        steps = f"{step_config.default} (max {step_config.max})"
                    elif hasattr(step_config, "default"):
                        steps = str(step_config.default)

            # Generation price
            gen_price = "N/A"
            if (
                hasattr(model.model_spec, "pricing")
                and model.model_spec.pricing
                and (
                    hasattr(model.model_spec.pricing, "generation")
                    and model.model_spec.pricing.generation
                )
            ):
                gen_price = cls.format_gen_price(model.model_spec.pricing.generation, currency)

            # Status
            status = "✅ Online"
            if hasattr(model.model_spec, "offline") and model.model_spec.offline:
                status = "🔴 Offline"
            elif hasattr(model.model_spec, "beta") and model.model_spec.beta:
                status = "🔶 Beta"

            table.add_row(name_col, steps, gen_price, status)

        return table

    @classmethod
    def format_tts_table(cls, models: list[Any], currency: str = "both") -> Table:
        """Format TTS models in table"""
        table = Table(
            title=f"TTS MODELS ({len(models)} available)",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Model Name\nID: model-id", style="cyan", no_wrap=False)
        table.add_column("Voices", style="green")
        table.add_column(
            f"Pricing ({currency.upper() if currency != 'both' else 'USD / DIEM'})\nper 1M characters",
            style="blue",
        )

        for model in sorted(models, key=lambda x: x.id):
            # Name
            name_parts = []
            if hasattr(model.model_spec, "name") and model.model_spec.name:
                name_parts.append(model.model_spec.name)
            else:
                name_parts.append(model.id)
            name_parts.append(f"[dim]{model.id}[/dim]")
            name_col = "\n".join(name_parts)

            # Voices
            voices = "N/A"
            if hasattr(model.model_spec, "voices") and model.model_spec.voices:
                voices = str(len(model.model_spec.voices))

            # Pricing
            price = "N/A"
            if (
                hasattr(model.model_spec, "pricing")
                and model.model_spec.pricing
                and hasattr(model.model_spec.pricing, "input")
                and model.model_spec.pricing.input
            ):
                price = cls.format_price(model.model_spec.pricing.input, currency)

            table.add_row(name_col, voices, price)

        return table

    @classmethod
    def format_embedding_table(cls, models: list[Any], currency: str = "both") -> Table:
        """Format embedding models in table"""
        table = Table(
            title=f"EMBEDDING MODELS ({len(models)} available)",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Model Name\nID: model-id", style="cyan", no_wrap=False)
        table.add_column("Context", style="green")
        table.add_column(
            f"Pricing ({currency.upper() if currency != 'both' else 'USD / DIEM'})\nper 1M tokens",
            style="blue",
        )

        for model in sorted(models, key=lambda x: x.id):
            # Name
            name_parts = []
            if hasattr(model.model_spec, "name") and model.model_spec.name:
                name_parts.append(model.model_spec.name)
            else:
                name_parts.append(model.id)
            name_parts.append(f"[dim]{model.id}[/dim]")
            name_col = "\n".join(name_parts)

            # Context
            context = "N/A"
            if hasattr(model.model_spec, "availableContextTokens"):
                context = cls.format_context(model.model_spec.availableContextTokens)

            # Pricing
            pricing_lines = []
            if hasattr(model.model_spec, "pricing") and model.model_spec.pricing:
                if hasattr(model.model_spec.pricing, "input") and model.model_spec.pricing.input:
                    pricing_lines.append(
                        f"In: {cls.format_price(model.model_spec.pricing.input, currency)}"
                    )
                if hasattr(model.model_spec.pricing, "output") and model.model_spec.pricing.output:
                    pricing_lines.append(
                        f"Out: {cls.format_price(model.model_spec.pricing.output, currency)}"
                    )
            price_col = "\n".join(pricing_lines) if pricing_lines else "N/A"

            table.add_row(name_col, context, price_col)

        return table

    @classmethod
    def format_upscale_table(cls, models: list[Any], currency: str = "both") -> Table:
        """Format upscale models in table"""
        table = Table(
            title=f"UPSCALE MODELS ({len(models)} available)",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            padding=(0, 1),
        )

        table.add_column("Model Name\nID: model-id", style="cyan", no_wrap=False)
        table.add_column(
            f"Upscaling Prices ({currency.upper() if currency != 'both' else 'USD / DIEM'})",
            style="blue",
        )

        for model in sorted(models, key=lambda x: x.id):
            # Name
            name_parts = []
            if hasattr(model.model_spec, "name") and model.model_spec.name:
                name_parts.append(model.model_spec.name)
            else:
                name_parts.append(model.id)
            name_parts.append(f"[dim]{model.id}[/dim]")
            name_col = "\n".join(name_parts)

            # Pricing - upscale models have dict-like upscale object with "2x"/"4x" keys
            pricing_info = []
            if hasattr(model.model_spec, "pricing") and model.model_spec.pricing:
                pricing = model.model_spec.pricing

                # Show generation price if available
                if hasattr(pricing, "generation") and pricing.generation:
                    pricing_info.append(
                        f"Gen: {cls.format_gen_price(pricing.generation, currency)}"
                    )

                # Handle upscale pricing with dict-style access
                if hasattr(pricing, "upscale") and pricing.upscale:
                    upscale = pricing.upscale
                    # Access dict keys directly (Python objects from API use dict-like access)
                    try:
                        # Try as dictionary first
                        if isinstance(upscale, dict):
                            if "2x" in upscale:
                                pricing_info.append(
                                    f"2x: {cls.format_gen_price(upscale['2x'], currency)}"
                                )
                            if "4x" in upscale:
                                pricing_info.append(
                                    f"4x: {cls.format_gen_price(upscale['4x'], currency)}"
                                )
                        # Try as object with __getitem__
                        elif hasattr(upscale, "__getitem__"):
                            try:
                                price_2x = upscale["2x"]
                                pricing_info.append(
                                    f"2x: {cls.format_gen_price(price_2x, currency)}"
                                )
                            except (KeyError, TypeError):
                                pass
                            try:
                                price_4x = upscale["4x"]
                                pricing_info.append(
                                    f"4x: {cls.format_gen_price(price_4x, currency)}"
                                )
                            except (KeyError, TypeError):
                                pass
                    except Exception as e:
                        # Pricing parse failed across all known shapes — leave column empty.
                        import logging

                        logging.getLogger(__name__).debug("pricing format failed: %s", e)

            price_col = " | ".join(pricing_info) if pricing_info else "N/A"
            table.add_row(name_col, price_col)

        return table

    @classmethod
    def format_inpaint_table(cls, models: list[Any]) -> Table:
        """Format inpaint models in table"""
        table = Table(
            title=f"INPAINT MODELS ({len(models)} available)",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            padding=(0, 1),
        )

        table.add_column("Model Name\nID: model-id", style="cyan", no_wrap=False)
        table.add_column("Status", style="yellow")
        table.add_column("Beta", style="yellow")

        for model in sorted(models, key=lambda x: x.id):
            # Name
            name_parts = []
            if hasattr(model.model_spec, "name") and model.model_spec.name:
                name_parts.append(model.model_spec.name)
            else:
                name_parts.append(model.id)
            name_parts.append(f"[dim]{model.id}[/dim]")
            name_col = "\n".join(name_parts)

            # Status
            status = "✅ Online"
            if hasattr(model.model_spec, "offline") and model.model_spec.offline:
                status = "🔴 Offline"

            # Beta
            beta_status = "🔶 Beta" if getattr(model.model_spec, "beta", False) else ""

            table.add_row(name_col, status, beta_status)

        return table

    @classmethod
    def format_verbose_model(cls, model: Any, currency: str = "both") -> Panel:
        """Format detailed model view"""
        content = []

        # Header
        name = (
            getattr(model.model_spec, "name", model.id)
            if hasattr(model.model_spec, "name")
            else model.id
        )
        content.append(f"🤖 [bold cyan]{name}[/bold cyan]")
        content.append(f"ID: [dim]{model.id}[/dim]")
        content.append(f"Type: [yellow]{model.type.upper()}[/yellow]")

        if hasattr(model.model_spec, "traits") and model.model_spec.traits:
            traits = ", ".join(model.model_spec.traits)
            content.append(f"🏷️  Traits: [italic]{traits}[/italic]")

        content.append("─" * 69)

        # Capabilities (for text models)
        if model.type == "text" and hasattr(model.model_spec, "capabilities"):
            content.append("📊 [bold]CAPABILITIES[/bold]")
            caps = model.model_spec.capabilities

            cap_items = [
                ("Function Calling", getattr(caps, "supportsFunctionCalling", False)),
                ("Reasoning", getattr(caps, "supportsReasoning", False)),
                ("Response Schema", getattr(caps, "supportsResponseSchema", False)),
                ("Web Search", getattr(caps, "supportsWebSearch", False)),
                ("LogProbs", getattr(caps, "supportsLogProbs", False)),
                ("Vision", getattr(caps, "supportsVision", False)),
                ("Optimized for Code", getattr(caps, "optimizedForCode", False)),
            ]

            cap_line = "  "
            for i, (name, value) in enumerate(cap_items):
                check = "✓" if value else "✗"
                cap_line += f"{check} {name:20}"
                if (i + 1) % 3 == 0:
                    content.append(cap_line)
                    cap_line = "  "

            if cap_line.strip() and cap_line != "  ":
                content.append(cap_line)

            content.append("─" * 69)

        # Pricing
        if hasattr(model.model_spec, "pricing"):
            content.append("💰 [bold]PRICING[/bold]")
            pricing = model.model_spec.pricing

            if hasattr(pricing, "input") and pricing.input:
                content.append(
                    f"  Input (per 1M tokens):  {cls.format_price(pricing.input, currency)}"
                )
            if hasattr(pricing, "output") and pricing.output:
                content.append(
                    f"  Output (per 1M tokens): {cls.format_price(pricing.output, currency)}"
                )
            if hasattr(pricing, "generation") and pricing.generation:
                content.append(
                    f"  Generation:             {cls.format_gen_price(pricing.generation, currency)}"
                )
            if hasattr(pricing, "upscale") and pricing.upscale:
                upscale = pricing.upscale
                if hasattr(upscale, "2x"):
                    content.append(
                        f"  Upscale 2x:             {cls.format_gen_price(getattr(upscale, '2x'), currency)}"
                    )
                if hasattr(upscale, "4x"):
                    content.append(
                        f"  Upscale 4x:             {cls.format_gen_price(getattr(upscale, '4x'), currency)}"
                    )

            content.append("─" * 69)

        # Specifications
        content.append("📐 [bold]SPECIFICATIONS[/bold]")

        if (
            hasattr(model.model_spec, "availableContextTokens")
            and model.model_spec.availableContextTokens is not None
        ):
            context = cls.format_context(model.model_spec.availableContextTokens)
            content.append(
                f"  Context Window:    {model.model_spec.availableContextTokens:,} tokens ({context})"
            )

        if hasattr(model.model_spec, "capabilities") and hasattr(
            model.model_spec.capabilities, "quantization"
        ):
            content.append(f"  Quantization:      {model.model_spec.capabilities.quantization}")

        if hasattr(model.model_spec, "constraints"):
            constraints = model.model_spec.constraints
            if hasattr(constraints, "temperature"):
                temp = constraints.temperature
                if hasattr(temp, "default"):
                    content.append(f"  Temperature:       {temp.default} (default)")
            if hasattr(constraints, "top_p"):
                top_p = constraints.top_p
                if hasattr(top_p, "default"):
                    content.append(f"  Top P:             {top_p.default} (default)")
            if hasattr(constraints, "steps"):
                steps = constraints.steps
                if hasattr(steps, "default"):
                    step_text = f"{steps.default}"
                    if hasattr(steps, "max"):
                        step_text += f" (max {steps.max})"
                    content.append(f"  Steps:             {step_text}")

        content.append("─" * 69)

        # Source and metadata
        if hasattr(model.model_spec, "modelSource"):
            content.append(f"🔗 Source: {model.model_spec.modelSource}")

        if hasattr(model, "created"):
            created_date = datetime.fromtimestamp(model.created).strftime("%Y-%m-%d")
            content.append(f"📅 Created: {created_date}")

        # Status
        status_parts = []
        if hasattr(model.model_spec, "offline"):
            status_parts.append("🔴 Offline" if model.model_spec.offline else "✅ Online")
        if hasattr(model.model_spec, "beta") and model.model_spec.beta:
            status_parts.append("🔶 Beta")
        if status_parts:
            content.append(f"Status: {' | '.join(status_parts)}")

        return Panel("\n".join(content), border_style="cyan", padding=(1, 2))

    @staticmethod
    def format_json(models: list[Any]) -> str:
        """Format models as JSON"""
        models_dict = []
        for model in models:
            model_dict = {
                "id": model.id,
                "type": model.type,
                "name": getattr(model.model_spec, "name", None)
                if hasattr(model.model_spec, "name")
                else None,
                "traits": getattr(model.model_spec, "traits", [])
                if hasattr(model.model_spec, "traits")
                else [],
            }

            # Add capabilities for text models
            if hasattr(model.model_spec, "capabilities"):
                caps = model.model_spec.capabilities
                model_dict["capabilities"] = {
                    "function_calling": getattr(caps, "supportsFunctionCalling", False),
                    "vision": getattr(caps, "supportsVision", False),
                    "reasoning": getattr(caps, "supportsReasoning", False),
                    "web_search": getattr(caps, "supportsWebSearch", False),
                    "code_optimized": getattr(caps, "optimizedForCode", False),
                    "response_schema": getattr(caps, "supportsResponseSchema", False),
                }

            # Add pricing
            if hasattr(model.model_spec, "pricing"):
                pricing = model.model_spec.pricing
                model_dict["pricing"] = {}
                if hasattr(pricing, "input") and pricing.input:
                    model_dict["pricing"]["input_usd"] = pricing.input.usd
                    model_dict["pricing"]["input_diem"] = pricing.input.diem
                if hasattr(pricing, "output") and pricing.output:
                    model_dict["pricing"]["output_usd"] = pricing.output.usd
                    model_dict["pricing"]["output_diem"] = pricing.output.diem

            # Add context
            if hasattr(model.model_spec, "availableContextTokens"):
                model_dict["context_tokens"] = model.model_spec.availableContextTokens

            models_dict.append(model_dict)

        return json.dumps(models_dict, indent=2)
