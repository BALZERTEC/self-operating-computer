import os
import sys

import anthropic
import google.generativeai as genai
from prompt_toolkit.shortcuts import input_dialog
from prompt_toolkit.shortcuts import radiolist_dialog
from dotenv import load_dotenv, set_key
from pathlib import Path
from ollama import Client
from openai import OpenAI

from operate.utils.style import style


class Config:
    """
    Configuration class for managing settings.

    Attributes:
        verbose (bool): Flag indicating whether verbose mode is enabled.
        openai_api_key (str): API key for OpenAI.
        google_api_key (str): API key for Google.
        ollama_host (str): url to ollama running remotely.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Put any initialization here
        return cls._instance

    def __init__(self):
        load_dotenv()
        self.verbose = False
        self.model_provider = os.getenv("MODEL_PROVIDER")
        self.model_name = os.getenv("MODEL_NAME")
        self.openai_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )
        self.google_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )
        self.ollama_host = (
            None  # instance variables are backups in case savint to a `.env` fails
        )
        self.anthropic_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )
        self.qwen_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )
        self.env_path = Path(".env")

    def initialize_openai(self):
        if self.verbose:
            print("[Config][initialize_openai]")

        if self.openai_api_key:
            if self.verbose:
                print("[Config][initialize_openai] using cached openai_api_key")
            api_key = self.openai_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_openai] no cached openai_api_key, try to get from env."
                )
            api_key = os.getenv("OPENAI_API_KEY")

        client = OpenAI(
            api_key=api_key,
        )
        client.api_key = api_key
        client.base_url = os.getenv("OPENAI_API_BASE_URL", client.base_url)
        return client

    def initialize_qwen(self):
        if self.verbose:
            print("[Config][initialize_qwen]")

        if self.qwen_api_key:
            if self.verbose:
                print("[Config][initialize_qwen] using cached qwen_api_key")
            api_key = self.qwen_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_qwen] no cached qwen_api_key, try to get from env."
                )
            api_key = os.getenv("QWEN_API_KEY")

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client.api_key = api_key
        client.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return client

    def initialize_google(self):
        if self.google_api_key:
            if self.verbose:
                print("[Config][initialize_google] using cached google_api_key")
            api_key = self.google_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_google] no cached google_api_key, try to get from env."
                )
            api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key, transport="rest")
        model = genai.GenerativeModel("gemini-pro-vision")

        return model

    def initialize_ollama(self):
        if self.ollama_host:
            if self.verbose:
                print("[Config][initialize_ollama] using cached ollama host")
        else:
            if self.verbose:
                print(
                    "[Config][initialize_ollama] no cached ollama host. Assuming ollama running locally."
                )
            self.ollama_host = os.getenv("OLLAMA_HOST", None)
        model = Client(host=self.ollama_host)
        return model

    def configure_ollama(self, preferred_model=None):
        load_dotenv()
        self.ollama_host = os.getenv("OLLAMA_HOST") or self.ollama_host
        if not self.ollama_host:
            host = input_dialog(
                title="Ollama Host", text="Please enter your Ollama server URL:"
            ).run()
            if host is None:
                sys.exit("Operation cancelled by user.")
            host = host.strip()
            if not host:
                sys.exit("Ollama host cannot be empty.")
            self.ollama_host = host
            set_key(str(self.env_path), "OLLAMA_HOST", self.ollama_host)
            load_dotenv()

        try:
            client = Client(host=self.ollama_host)
            models_response = client.list()
        except Exception as exc:
            sys.exit(
                f"Could not connect to Ollama at {self.ollama_host}. Please ensure the server is running. Error: {exc}"
            )

        if isinstance(models_response, dict):
            available_models = models_response.get("models", [])
        else:
            available_models = getattr(models_response, "models", [])

        model_names = [available_model.get("name") for available_model in available_models if available_model.get("name")]

        if not model_names:
            sys.exit(
                f"No models available on Ollama server at {self.ollama_host}. Pull a model and try again."
            )

        if preferred_model and preferred_model in model_names:
            chosen_model = preferred_model
        else:
            selection = radiolist_dialog(
                title="Ollama Model",
                text="Select an Ollama model to use:",
                values=[(model_name, model_name) for model_name in model_names],
                style=style,
            ).run()

            if selection is None:
                sys.exit("Operation cancelled by user.")

            chosen_model = selection

        return chosen_model

    def initialize_anthropic(self):
        if self.anthropic_api_key:
            api_key = self.anthropic_api_key
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)

    def validation(self, model, voice_mode, provider=None):
        """
        Validate the input parameters for the dialog operation.
        """
        provider = provider or self.model_provider

        if provider == "ollama":
            self._validate_ollama_settings(model)
        else:
            self.require_api_key(
                "OPENAI_API_KEY",
                "OpenAI API key",
                model == "gpt-4"
                or voice_mode
                or model == "gpt-4-with-som"
                or model == "gpt-4-with-ocr"
                or model == "gpt-4.1-with-ocr"
                or model == "o1-with-ocr",
            )
        self.require_api_key(
            "GOOGLE_API_KEY", "Google API key", model == "gemini-pro-vision"
        )
        self.require_api_key(
            "ANTHROPIC_API_KEY", "Anthropic API key", model == "claude-3"
        )
        self.require_api_key("QWEN_API_KEY", "Qwen API key", model == "qwen-vl")

    def require_api_key(self, key_name, key_description, is_required):
        key_exists = bool(os.environ.get(key_name))
        if self.verbose:
            print("[Config] require_api_key")
            print("[Config] key_name", key_name)
            print("[Config] key_description", key_description)
            print("[Config] key_exists", key_exists)
        if is_required and not key_exists:
            self.prompt_and_save_api_key(key_name, key_description)

    def prompt_and_save_api_key(self, key_name, key_description):
        key_value = input_dialog(
            title="API Key Required", text=f"Please enter your {key_description}:"
        ).run()

        if key_value is None:  # User pressed cancel or closed the dialog
            sys.exit("Operation cancelled by user.")

        if key_value:
            if key_name == "OPENAI_API_KEY":
                self.openai_api_key = key_value
            elif key_name == "GOOGLE_API_KEY":
                self.google_api_key = key_value
            elif key_name == "ANTHROPIC_API_KEY":
                self.anthropic_api_key = key_value
            elif key_name == "QWEN_API_KEY":
                self.qwen_api_key = key_value
            self.save_api_key_to_env(key_name, key_value)
            load_dotenv()  # Reload environment variables
            # Update the instance attribute with the new key

    @staticmethod
    def save_api_key_to_env(key_name, key_value):
        with open(".env", "a") as file:
            file.write(f"\n{key_name}='{key_value}'")

    def _validate_ollama_settings(self, model):
        self.ollama_host = os.getenv("OLLAMA_HOST") or self.ollama_host
        self.model_name = model or self.model_name

        if not self.ollama_host:
            sys.exit(
                "Ollama provider selected but no OLLAMA_HOST configured. Run the CLI with --provider ollama to set it up."
            )
        try:
            client = Client(host=self.ollama_host)
            models_response = client.list()
        except Exception as exc:
            sys.exit(
                f"Could not connect to Ollama at {self.ollama_host}. Please ensure the server is running. Error: {exc}"
            )

        if isinstance(models_response, dict):
            available_models = models_response.get("models", [])
        else:
            available_models = getattr(models_response, "models", [])

        model_names = [
            available_model.get("name")
            for available_model in available_models
            if available_model.get("name")
        ]

        if not model_names:
            sys.exit(
                f"No models available on Ollama server at {self.ollama_host}. Pull a model and try again."
            )

        if self.model_name and self.model_name not in model_names:
            sys.exit(
                f"Configured Ollama model '{self.model_name}' not available on {self.ollama_host}. Update your configuration or pull the model."
            )
