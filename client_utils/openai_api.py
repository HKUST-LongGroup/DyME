import time

import httpx
from openai import OpenAI
from typing import Optional


# It's good practice to define a simple configuration object or use a dictionary
# for passing credentials, rather than a generic object.
# For this example, we'll assume a config object like this.
class ClientConfig:
    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id


class OpenAIClient:
    """
    A client wrapper for interacting with the OpenAI ChatCompletion API.
    It handles client initialization and API calls with retry logic.
    """

    def __init__(self, config: ClientConfig, max_retries: int = 3):
        # The OpenAI client is initialized directly within the class constructor.
        # This improves encapsulation by making the class self-contained.
        # It takes the configuration object as a direct argument.
        custom_http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(
            api_key=config['api_key'],  # Required: your API key
            base_url=config['api_base'],  # Optional: only needed for third-party services
            http_client=custom_http_client,
        )
        self.model_id = config['model_id']
        self.max_retries = max_retries

    def get_completion(
            self,
            user_prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 1024
    ) -> Optional[str]:
        """
        Calls the OpenAI ChatCompletion API and returns the result.
        Includes retry logic for handling transient errors.

        Args:
            user_prompt (str): The main input/prompt from the user.
            system_prompt (Optional[str]): The system-level instruction for the model. Defaults to None.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 1024.

        Returns:
            Optional[str]: The content of the model's response, or None if the API call fails after all retries.
        """
        # Build the message list based on provided prompts.
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        # Implement a clear retry loop instead of 'while True'.
        for attempt in range(self.max_retries):
            try:
                # Make the API call to the chat completions endpoint.
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    # temperature=0.2 # You can add other parameters here as needed.
                )
                # If the call is successful, return the message content and exit the loop.
                return response.choices[0].message.content

            except Exception as e:
                # If an error occurs, print a helpful message.
                print(f"API call failed on attempt {attempt + 1}/{self.max_retries}. Error: {e}")

                # If this was the last attempt, break the loop to return None.
                if attempt + 1 == self.max_retries:
                    print("All retry attempts failed.")
                    break

                # Wait for a short period before trying again.
                print("Retrying in 2 seconds...")
                time.sleep(2)

        # Return None if all retries fail.
        return None


# --- How to Use the Refactored Class ---
if __name__ == '__main__':
    # 1. Define your configuration
    # Replace with your actual credentials and mode

    CLIENT_CONFIG = {
        "client_type": "openai",  
        "api_key": "none",  
        "api_base": "http://127.0.0.1:23333/v1",  
        "timeout": 60,  
        "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",  
        "init_port": 23333, 
        "num_server": 8
    }
    # 2. Instantiate the client
    my_client = OpenAIClient(config=CLIENT_CONFIG)

    # 3. Define your prompts
    user_message = "What is the capital of France?"
    system_message = "You are a helpful assistant that provides concise answers."

    # 4. Get the model's response
    response_content = my_client.get_completion(
        user_prompt=user_message,
        system_prompt=system_message
    )

    # 5. Print the result
    if response_content:
        print("\nModel Response:")
        print(response_content)
    else:
        print("\nFailed to get a response from the model.")