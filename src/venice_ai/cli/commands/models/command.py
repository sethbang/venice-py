"""
Main command handler for models listing
"""

from typing import get_args

import click
from rich.panel import Panel

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.resources.models import ModelListType

from ...config import get_client_kwargs
from ...utils import console, print_error, print_info
from .comparator import ModelComparator
from .filters import FilterOptions, ModelFilter
from .formatters import ModelFormatter
from .sorter import ModelSorter

#: ``ModelListType`` members that are not concrete model types: ``"chat"`` is
#: an SDK-side alias for ``"text"``, and ``"all"``/``"code"`` are cross-cutting
#: views. Everything else names a type the catalog can be filtered by.
ALIAS_MODEL_TYPES = frozenset({"chat", "all", "code"})


def concrete_model_types() -> list[ModelListType]:
    """Every real model type, derived from ``ModelListType``.

    Hand-listing these silently stops covering new types: the ``--type`` filter
    can only narrow what was already fetched, so a missing type reads as
    "No models match the specified filters" rather than as an error. That is
    exactly how ``--type decision`` came to return nothing while the model was
    live in the catalog.
    """
    return [t for t in get_args(ModelListType.__value__) if t not in ALIAS_MODEL_TYPES]


async def list_models(
    ctx: click.Context,
    model_type: list[str] | None = None,
    verbose: bool = False,
    output_json: bool = False,
    currency: str = "both",
    # Capability filters
    function_calling: bool = False,
    vision: bool = False,
    reasoning: bool = False,
    web_search: bool = False,
    code: bool = False,
    response_schema: bool = False,
    # Trait filters
    traits: list[str] | None = None,
    # Price filters
    max_input: float | None = None,
    max_output: float | None = None,
    max_gen: float | None = None,
    budget: float | None = None,
    # Status filters
    beta: bool | None = None,
    online: bool | None = None,
    # Search and detail
    search: str | None = None,
    detail_id: str | None = None,
    compare_ids: str | None = None,
    # Display options
    sort: str = "name",
    no_legend: bool = False,
) -> None:
    """List and filter available AI models"""

    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            # Fetch all models
            print_info("Fetching available models...")

            all_models = []
            seen_ids = set()
            # Fetch every real model type so --type video|asr|music returns
            # results instead of silently empty.
            model_types_to_fetch: list[ModelListType] = concrete_model_types()

            for mtype in model_types_to_fetch:
                try:
                    response = await client.models.list(type=mtype)
                    if response.data:
                        # Deduplicate by ID to avoid showing same model multiple times
                        for model in response.data:
                            if model.id not in seen_ids:
                                all_models.append(model)
                                seen_ids.add(model.id)
                except Exception as e:
                    # Some model types might not be available — log and skip.
                    import logging

                    logging.getLogger(__name__).debug("models.list(type=%s) failed: %s", mtype, e)

            if not all_models:
                print_info("No models found")
                return

            # Handle comparison mode
            if compare_ids:
                model_ids = [mid.strip() for mid in compare_ids.split(",")]
                models_to_compare = []
                for mid in model_ids:
                    model = ModelComparator.find_model_by_id(all_models, mid)
                    if model:
                        models_to_compare.append(model)
                    else:
                        print_error(f"Model not found: {mid}")

                if len(models_to_compare) < 2:
                    print_error("Need at least 2 models to compare")
                    raise SystemExit(1)

                comparison_table = ModelComparator.compare_models(models_to_compare, currency)
                if comparison_table:
                    console.print(comparison_table)
                return

            # Handle detail mode
            if detail_id:
                model = ModelComparator.find_model_by_id(all_models, detail_id)
                if not model:
                    print_error(f"Model not found: {detail_id}")
                    raise SystemExit(1)

                panel = ModelFormatter.format_verbose_model(model, currency)
                console.print(panel)
                return

            # Build filter options
            capabilities = []
            if function_calling:
                capabilities.append("function-calling")
            if vision:
                capabilities.append("vision")
            if reasoning:
                capabilities.append("reasoning")
            if web_search:
                capabilities.append("web-search")
            if code:
                capabilities.append("code")
            if response_schema:
                capabilities.append("response-schema")

            filter_opts = FilterOptions(
                types=list(model_type) if model_type else None,
                capabilities=capabilities if capabilities else None,
                traits=list(traits) if traits else None,
                max_input_price=max_input,
                max_output_price=max_output,
                max_gen_price=max_gen,
                budget=budget,
                beta=beta,
                online=online,
                search_query=search,
            )

            # Apply filters
            filtered_models = ModelFilter.apply_all_filters(all_models, filter_opts)

            if not filtered_models:
                print_info("No models match the specified filters")
                return

            # Sort models
            sorted_models = ModelSorter.sort_models(filtered_models, sort)

            # JSON output
            if output_json:
                json_output = ModelFormatter.format_json(sorted_models)
                console.print(json_output)
                return

            # Group models by type
            models_by_type: dict[str, list] = {}
            for model in sorted_models:
                model_type_key = model.type or "unknown"
                if model_type_key not in models_by_type:
                    models_by_type[model_type_key] = []
                models_by_type[model_type_key].append(model)

            # Display header
            console.print()
            console.print(Panel("[bold cyan]📋 VENICE AI MODELS[/bold cyan]", border_style="cyan"))
            console.print()

            # Display models by type
            for model_type_key in sorted(models_by_type.keys()):
                type_models = models_by_type[model_type_key]

                if verbose:
                    # Verbose mode: show detailed panels for each model
                    console.print(
                        f"\n[bold cyan]{model_type_key.upper()} MODELS - DETAILED VIEW[/bold cyan]\n"
                    )
                    for model in type_models:
                        panel = ModelFormatter.format_verbose_model(model, currency)
                        console.print(panel)
                        console.print()
                else:
                    # Compact mode: use formatters
                    if model_type_key == "text":
                        table = ModelFormatter.format_text_table(type_models, currency)
                    elif model_type_key == "image":
                        table = ModelFormatter.format_image_table(type_models, currency)
                    elif model_type_key == "tts":
                        table = ModelFormatter.format_tts_table(type_models, currency)
                    elif model_type_key == "embedding":
                        table = ModelFormatter.format_embedding_table(type_models, currency)
                    elif model_type_key == "upscale":
                        table = ModelFormatter.format_upscale_table(type_models, currency)
                    elif model_type_key == "inpaint":
                        table = ModelFormatter.format_inpaint_table(type_models)
                    else:
                        # Fallback to text table for unknown types
                        table = ModelFormatter.format_text_table(type_models, currency)

                    console.print(table)
                    console.print()

            # Show legend unless disabled
            if not no_legend and any(m.type == "text" for m in sorted_models):
                console.print(ModelFormatter.get_capability_legend())
                console.print()

            # Show summary
            total = len(sorted_models)
            types_count = len(models_by_type)
            console.print(
                Panel(
                    f"💡 Tip: Use --verbose for detailed view, --filter options to narrow results\n"
                    f"📊 Total: [bold]{total}[/bold] models across [bold]{types_count}[/bold] types",
                    border_style="dim",
                )
            )

    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e
