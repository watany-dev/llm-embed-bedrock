import os
import logging
import llm
import boto3
import json
from typing import Iterable, List

# Default environment variables for Bedrock model and AWS region
BEDROCK_EMBEDDING_MODEL_ID = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "my-embedding-model-id")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Content type for Bedrock model invocation
CONTENT_TYPE = "application/json"
MAX_LENGTH = 8192  # Maximum input text length for the model


class BedrockEmbeddingModel(llm.EmbeddingModel):
    """
    LLM embedding model implementation using AWS Bedrock.
    """

    def __init__(self, model_id: str = BEDROCK_EMBEDDING_MODEL_ID, region: str = AWS_REGION):
        self.model_id = model_id
        self.region = region
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def embed_batch(self, texts: Iterable[str]) -> Iterable[List[float]]:
        """
        Embed a batch of texts using the Bedrock embedding model.
        :param texts: Iterable of input strings to embed.
        :return: Iterable of embedding vectors (list of floats).
        """
        embeddings = []
        for text in texts:
            try:
                # Truncate text to fit within model's input constraints
                truncated_text = text[:MAX_LENGTH]

                # Prepare request payload
                request_body = {"input": truncated_text}

                # Invoke the Bedrock model
                response = self._client.invoke_model(
                    modelId=self.model_id,
                    accept=CONTENT_TYPE,
                    contentType=CONTENT_TYPE,
                    body=json.dumps(request_body).encode("utf-8"),
                )

                # Parse the response body for embedding vector
                response_body = response["body"].read().decode("utf-8")
                vector = json.loads(response_body).get("embedding", [])
                embeddings.append(vector)

            except Exception as e:
                # Log error and return an empty vector for failed cases
                logging.error(f"Bedrock embedding request failed: {e}")
                embeddings.append([])

        return embeddings


@llm.hookimpl
def register_embedding_models(register):
    """
    Hook for registering the Bedrock embedding model with the LLM framework.
    :param register: Registration function for embedding models.
    """
    register(BedrockEmbeddingModel(BEDROCK_EMBEDDING_MODEL_ID))
