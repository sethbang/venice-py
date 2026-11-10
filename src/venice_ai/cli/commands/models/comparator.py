"""
Model comparison functionality for Venice AI models
"""

from typing import Any

from rich.table import Table


class ModelComparator:
    """Handles model comparison"""

    @staticmethod
    def compare_models(models: list[Any], currency: str = "both") -> Table | None:
        """Create a comparison table for multiple models"""
        if not models:
            return None

        # Create comparison table
        table = Table(title="MODEL COMPARISON", show_header=True, header_style="bold cyan")

        # Add attribute column
        table.add_column("Attribute", style="cyan", no_wrap=False)

        # Add a column for each model
        for model in models:
            model_name = (
                getattr(model.model_spec, "name", model.id)
                if hasattr(model.model_spec, "name")
                else model.id
            )
            table.add_column(model_name, style="green")

        # Helper to get values for all models
        def add_row(attr_name: str, value_getter):
            values = [value_getter(m) for m in models]
            table.add_row(attr_name, *values)

        # Basic info
        add_row("ID", lambda m: m.id)
        add_row("Type", lambda m: m.type.upper())

        # Capabilities (if text models)
        if all(m.type == "text" for m in models):
            add_row(
                "Function Calling",
                lambda m: (
                    "✓"
                    if hasattr(m.model_spec, "capabilities")
                    and getattr(m.model_spec.capabilities, "supportsFunctionCalling", False)
                    else "✗"
                ),
            )
            add_row(
                "Vision",
                lambda m: (
                    "✓"
                    if hasattr(m.model_spec, "capabilities")
                    and getattr(m.model_spec.capabilities, "supportsVision", False)
                    else "✗"
                ),
            )
            add_row(
                "Reasoning",
                lambda m: (
                    "✓"
                    if hasattr(m.model_spec, "capabilities")
                    and getattr(m.model_spec.capabilities, "supportsReasoning", False)
                    else "✗"
                ),
            )
            add_row(
                "Web Search",
                lambda m: (
                    "✓"
                    if hasattr(m.model_spec, "capabilities")
                    and getattr(m.model_spec.capabilities, "supportsWebSearch", False)
                    else "✗"
                ),
            )
            add_row(
                "Code Optimized",
                lambda m: (
                    "✓"
                    if hasattr(m.model_spec, "capabilities")
                    and getattr(m.model_spec.capabilities, "optimizedForCode", False)
                    else "✗"
                ),
            )

        # Context window
        if any(hasattr(m.model_spec, "availableContextTokens") for m in models):
            add_row(
                "Context Window",
                lambda m: (
                    f"{m.model_spec.availableContextTokens:,}"
                    if hasattr(m.model_spec, "availableContextTokens")
                    else "N/A"
                ),
            )

        # Pricing
        if currency in ["both", "usd"]:
            add_row(
                "Input Price (USD)",
                lambda m: (
                    f"${m.model_spec.pricing.input.usd:.2f}"
                    if hasattr(m.model_spec, "pricing")
                    and hasattr(m.model_spec.pricing, "input")
                    and m.model_spec.pricing.input
                    else "N/A"
                ),
            )
            add_row(
                "Output Price (USD)",
                lambda m: (
                    f"${m.model_spec.pricing.output.usd:.2f}"
                    if hasattr(m.model_spec, "pricing")
                    and hasattr(m.model_spec.pricing, "output")
                    and m.model_spec.pricing.output
                    else "N/A"
                ),
            )

        if currency in ["both", "diem"]:
            add_row(
                "Input Price (DIEM)",
                lambda m: (
                    f"Ð{m.model_spec.pricing.input.diem:.2f}"
                    if hasattr(m.model_spec, "pricing")
                    and hasattr(m.model_spec.pricing, "input")
                    and m.model_spec.pricing.input
                    and m.model_spec.pricing.input.diem is not None
                    else "N/A"
                ),
            )
            add_row(
                "Output Price (DIEM)",
                lambda m: (
                    f"Ð{m.model_spec.pricing.output.diem:.2f}"
                    if hasattr(m.model_spec, "pricing")
                    and hasattr(m.model_spec.pricing, "output")
                    and m.model_spec.pricing.output
                    and m.model_spec.pricing.output.diem is not None
                    else "N/A"
                ),
            )

        # Traits
        add_row(
            "Default Trait",
            lambda m: (
                ", ".join(m.model_spec.traits)
                if hasattr(m.model_spec, "traits") and m.model_spec.traits
                else "None"
            ),
        )

        # Quantization
        if any(
            hasattr(m.model_spec, "capabilities")
            and hasattr(m.model_spec.capabilities, "quantization")
            for m in models
        ):
            add_row(
                "Quantization",
                lambda m: (
                    m.model_spec.capabilities.quantization
                    if hasattr(m.model_spec, "capabilities")
                    and hasattr(m.model_spec.capabilities, "quantization")
                    else "N/A"
                ),
            )

        # Status
        add_row(
            "Status",
            lambda m: (
                "🔴 Offline"
                if getattr(m.model_spec, "offline", False)
                else "🔶 Beta"
                if getattr(m.model_spec, "beta", False)
                else "✅ Online"
            ),
        )

        return table

    @staticmethod
    def find_model_by_id(models: list[Any], model_id: str) -> Any:
        """Find a model by its ID"""
        for model in models:
            if model.id == model_id:
                return model
        return None
