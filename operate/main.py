"""
Self-Operating Computer
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv, set_key
from prompt_toolkit.shortcuts import radiolist_dialog

from operate.utils.style import ANSI_BRIGHT_MAGENTA, style
from operate.operate import main


PROVIDER_DEFAULTS = {
    "openai": "gpt-4-with-ocr",
    "ollama": "llava",
}

CONFIG_PATH = Path(".env")


def _persist_choice(provider: str, model: str) -> None:
    set_key(str(CONFIG_PATH), "MODEL_PROVIDER", provider)
    set_key(str(CONFIG_PATH), "MODEL_NAME", model)


def _load_saved_choice():
    load_dotenv()
    provider = os.getenv("MODEL_PROVIDER")
    model = os.getenv("MODEL_NAME")
    if provider not in PROVIDER_DEFAULTS:
        provider = None
    if not model:
        model = None
    elif provider and model:
        return provider, model
    if provider:
        return provider, PROVIDER_DEFAULTS[provider]
    return None, None


def _prompt_for_provider():
    selection = radiolist_dialog(
        title="Model Provider",
        text="Which provider would you like to use?",
        values=[
            ("openai", "OpenAI (default model: gpt-4-with-ocr)"),
            ("ollama", "Self-hosted Ollama server (default model: llava)"),
        ],
        style=style,
    ).run()

    if selection is None:
        raise SystemExit("Operation cancelled by user.")

    model = PROVIDER_DEFAULTS[selection]
    _persist_choice(selection, model)
    return selection, model


def main_entry():
    saved_provider, saved_model = _load_saved_choice()

    if saved_provider is None:
        saved_provider, saved_model = _prompt_for_provider()

    parser = argparse.ArgumentParser(
        description="Run the self-operating-computer with a specified model."
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_DEFAULTS.keys()),
        default=saved_provider,
        help="Select the model provider to use",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model to use",
        required=False,
        default=saved_model,
    )

    # Add a voice flag
    parser.add_argument(
        "--voice",
        help="Use voice input mode",
        action="store_true",
    )
    
    # Add a flag for verbose mode
    parser.add_argument(
        "--verbose",
        help="Run operate in verbose mode",
        action="store_true",
    )
    
    # Allow for direct input of prompt
    parser.add_argument(
        "--prompt",
        help="Directly input the objective prompt",
        type=str,
        required=False,
    )

    try:
        args = parser.parse_args()
        if args.provider != saved_provider or args.model != saved_model:
            _persist_choice(args.provider, args.model)
        main(
            args.model,
            terminal_prompt=args.prompt,
            voice_mode=args.voice,
            verbose_mode=args.verbose
        )
    except KeyboardInterrupt:
        print(f"\n{ANSI_BRIGHT_MAGENTA}Exiting...")


if __name__ == "__main__":
    main_entry()
