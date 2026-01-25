import time
from openai import OpenAI  # We use the standard client, not Azure

_verbose = True  # Set to True to see what's happening locally

class LMStudioClient:
    """
    A client to interact with a local LLM hosted via LM Studio.
    """
    
    # In LM Studio, the 'model_id' often doesn't matter as much because 
    # the server usually loads one specific model, but it's good practice to keep it.
    _params = {
        "temperature": 0.0,   # Keep deterministic for testing
        "max_tokens": -1,
        "seed": 100,
    }

    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio", model="granite-3.3-2b-instruct"):
        """
        Initializes the client pointing to the local LM Studio server.
        
        :param base_url: The endpoint where LM Studio is listening (default: http://localhost:1234/v1)
        :param api_key: LM Studio doesn't strictly need a key, but the library requires a non-empty string.
        :param model: The model identifier
        """
        
        # We instantiate the standard OpenAI client but point it to "localhost"
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.params = dict(self._params)
        self.model = model

    def generate_response(self, user_input: str, system_input="You are a helpful assistant.") -> str:
        """
        Sends the prompt to the local model and retrieves the response.
        """
        
        # Prepare the message payload
        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input}
        ]
        
        # Prepare arguments for the API call
        # Note: We filter params to ensure we don't send unsupported Azure-specific args if any existed
        call_params = {
            "model": self.model, # This is often a placeholder in LM Studio
            "messages": messages,
            **self.params
        }

        if _verbose:
            print(f"--- Sending Request to {self.client.base_url} ---")

        try:
            time0 = time.time()
            
            # The structure is identical to the Azure call, just using self.client.chat.completions
            response = self.client.chat.completions.create(**call_params)
            
            time1 = time.time()
            elapsed = time1 - time0
            
            if _verbose:
                print(f"--- Inference Complete ({elapsed:.2f}s) ---")

        except Exception as e:
            print(f"Failed to call Local LLM: {str(e)}")
            # Common error: Server not started in LM Studio
            if "Connection refused" in str(e):
                print("Hint: Did you click 'Start Server' in LM Studio?")
            raise

        # Extract the content
        try:
            output = response.choices[0].message.content.strip()
        except Exception as e:
            print("Exception parsing response:", response)
            output = None
            
        return output