import mimetypes
import os
import sys
import json
import logging
from base64 import b64encode
from io import BytesIO
from typing import List, Optional, Union, Iterable

import boto3
import llm
from httpx import request
from PIL import Image

# Supported image formats for the Bedrock Embeddings API
BEDROCK_EMBEDDING_IMAGE_FORMATS = ["png", "jpeg", "gif", "webp"]

# Mapping from MIME types to Bedrock Embeddings-supported document formats
MIME_TYPE_TO_BEDROCK_EMBEDDING_DOCUMENT_FORMAT = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

# Maximum image size and dimensions
EMBEDDING_MAX_IMAGE_SIZE_MB = 25
EMBEDDING_MAX_IMAGE_DIMENSION = 2560  # Example value; adjust as needed

# Default Titan Embedding model IDs
DEFAULT_TITAN_TEXT_EMBED_V2 = "amazon.titan-embed-text-v2:0"
DEFAULT_TITAN_IMAGE_EMBED_G1 = "amazon.titan-embed-image-g1:0"

# Maximum tokens and embedding lengths
MAX_TOKENS_TEXT_EMBED = 8192
MAX_TOKENS_IMAGE_EMBED = 256
DEFAULT_EMBEDDING_LENGTH = 1024  # Can be set to 256, 384, or 1024


@llm.hookimpl
def register_models(register):
    """Register Amazon Titan Embedding models with llm. You can change aliases as desired."""
    register(
        BedrockTitanTextEmbeddingV2(DEFAULT_TITAN_TEXT_EMBED_V2),
        aliases=("bedrock-titan-text-embed-v2", "titan-text-embed-v2"),
    )
    register(
        BedrockTitanImageEmbeddingG1(DEFAULT_TITAN_IMAGE_EMBED_G1),
        aliases=("bedrock-titan-image-embed-g1", "titan-image-embed-g1"),
    )


class BedrockEmbeddingBase(llm.Model):
    """Base class for Amazon Bedrock Embedding models."""

    can_stream: bool = False  # Embeddings typically don't stream
    attachment_types = (
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "application/pdf",
        "text/csv",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/html",
        "text/plain",
        "text/markdown",
    )

    class Options(llm.Options):
        """Parameters that users can optionally override."""

        def __init__(
            self,
            bedrock_model_id: Optional[str] = None,
            bedrock_attach: Optional[List[str]] = None,
            attachment_type: Optional[str] = None,
            embedding_length: Optional[int] = DEFAULT_EMBEDDING_LENGTH,
        ):
            self.bedrock_model_id = bedrock_model_id
            self.bedrock_attach = bedrock_attach or []
            self.attachment_type = attachment_type
            self.embedding_length = embedding_length

            # Validate embedding_length
            if self.embedding_length not in [256, 384, 1024]:
                raise ValueError("embedding_length must be one of [256, 384, 1024]")

    def __init__(self, model_id: str):
        """
        :param model_id: The Bedrock modelId for invocation (e.g., amazon.titan-embed-text-v2:0).
        """
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime")

    @staticmethod
    def load_and_preprocess_image(file_path):
        """Load and preprocess the given image for Bedrock Embeddings API."""
        with open(file_path, "rb") as fp:
            img_bytes = fp.read()

        if len(img_bytes) > EMBEDDING_MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image size exceeds {EMBEDDING_MAX_IMAGE_SIZE_MB} MB limit.")

        with Image.open(BytesIO(img_bytes)) as img:
            img_format = img.format.lower()
            width, height = img.size

            # Resize if the image is larger than the maximum dimension
            if width > EMBEDDING_MAX_IMAGE_DIMENSION or height > EMBEDDING_MAX_IMAGE_DIMENSION:
                img.thumbnail((EMBEDDING_MAX_IMAGE_DIMENSION, EMBEDDING_MAX_IMAGE_DIMENSION))

            # If format is already supported and no resize occurred, keep original bytes
            if img_format in BEDROCK_EMBEDDING_IMAGE_FORMATS and img.size == (width, height):
                return img_bytes, img_format

            # Otherwise, convert to PNG
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return buffer.getvalue(), "png"

    def load_attachment(self, file_path: str, mime_type: Optional[str] = None) -> dict:
        """Load and preprocess an attachment (image/document)."""
        if file_path == "-":  # Read from stdin
            file_content = BytesIO(sys.stdin.buffer.read())
            mime_type = mime_type or "application/octet-stream"
        elif file_path.startswith("http://") or file_path.startswith("https://"):
            # Download file from URL
            response = request.get(file_path)
            response.raise_for_status()
            file_content = BytesIO(response.content)
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(file_path)
        else:
            # Load from local file
            with open(file_path, "rb") as fp:
                file_content = BytesIO(fp.read())
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(file_path)

        if not mime_type or mime_type not in self.attachment_types:
            raise ValueError(f"Unsupported attachment type: {mime_type or 'unknown'}")

        return {
            "inlineData": {
                "data": b64encode(file_content.getvalue()).decode("utf-8"),
                "mimeType": mime_type,
            }
        }

    def prepare_payload(self, prompt: llm.Prompt, embedding_length: int) -> dict:
        """Convert an llm.Prompt into a Bedrock Embeddings API payload."""
        payload = {"embeddingConfig": {"outputEmbeddingLength": embedding_length}}

        # Handle attachments if any
        if prompt.options.bedrock_attach:
            for file_path in prompt.options.bedrock_attach:
                file_path = os.path.expanduser(file_path.strip())
                attachment = self.load_attachment(file_path, prompt.options.attachment_type)
                payload.update(attachment)

        # Add input text if present
        if prompt.prompt:
            payload["inputText"] = prompt.prompt[:MAX_TOKENS_TEXT_EMBED]

        return payload

    def invoke_bedrock(self, payload: dict) -> dict:
        """Invoke the Bedrock Embeddings model with the given payload."""
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
            return json.loads(response["body"].read().decode("utf-8"))
        except Exception as e:
            logging.error(f"Bedrock embedding request failed for {self.model_id}: {e}")
            raise

    def embed_batch(self, texts: Iterable[str], embedding_length: int) -> Iterable[List[float]]:
        """
        Placeholder method to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BedrockTitanTextEmbeddingV2(BedrockEmbeddingBase, llm.EmbeddingModel):
    """Amazon Titan Text Embeddings V2 Model."""

    def embed_batch(
        self, texts: Iterable[str], embedding_length: Optional[int] = DEFAULT_EMBEDDING_LENGTH
    ) -> Iterable[List[float]]:
        """
        Generate embeddings for a batch of text inputs.
        :param texts: Iterable of input strings to embed.
        :param embedding_length: Desired embedding length (256, 384, or 1024).
        :return: Iterable of embedding vectors (list of floats).
        """
        results = []
        for text in texts:
            truncated_text = text[:MAX_TOKENS_TEXT_EMBED]
            payload = {
                "inputText": truncated_text,
                "embeddingConfig": {"outputEmbeddingLength": embedding_length},
            }

            try:
                response = self.invoke_bedrock(payload)
                embedding = response.get("embedding", [])
                results.append(embedding)
            except Exception:
                # Append an empty list if embedding fails
                results.append([])

        return results


class BedrockTitanImageEmbeddingG1(BedrockEmbeddingBase, llm.EmbeddingModel):
    """Amazon Titan Multimodal Embeddings G1 Model."""

    def embed_batch(
        self,
        inputs: Iterable[Union[str, dict]],
        embedding_length: Optional[int] = DEFAULT_EMBEDDING_LENGTH,
    ) -> Iterable[List[float]]:
        """
        Generate embeddings for a batch of text and/or image inputs.
        :param inputs: Iterable of text strings or dictionaries with "inputImage".
                       Example: ["text1", {"inputImage": "base64_image_string"}]
        :param embedding_length: Desired embedding length (256, 384, or 1024).
        :return: Iterable of embedding vectors (list of floats).
        """
        results = []
        for item in inputs:
            payload = {"embeddingConfig": {"outputEmbeddingLength": embedding_length}}

            try:
                if isinstance(item, str):
                    # Treat as text input
                    truncated_text = item[:MAX_TOKENS_IMAGE_EMBED]
                    payload["inputText"] = truncated_text
                elif isinstance(item, dict) and "inputImage" in item:
                    # Treat as image input
                    image_data = item["inputImage"]
                    # Assuming image_data is already a base64-encoded string
                    payload["inputImage"] = image_data
                else:
                    logging.warning("Unsupported input type for image embedding.")
                    results.append([])
                    continue

                response = self.invoke_bedrock(payload)
                embedding = response.get("embedding", [])
                results.append(embedding)
            except Exception:
                # Append an empty list if embedding fails
                results.append([])

        return results
