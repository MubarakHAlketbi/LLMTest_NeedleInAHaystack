import asyncio
import random
from typing import Callable, Any

async def retry_with_exponential_backoff(
    operation: Callable[..., Any],
    max_retries: int = 5,
    base_delay: float = 1,
    max_delay: float = 60,
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Retry an asynchronous operation with exponential backoff.

    Args:
        operation (Callable): The async function to retry.
        max_retries (int): Maximum number of retries before giving up.
        base_delay (float): The base delay in seconds for exponential backoff.
        max_delay (float): The maximum delay in seconds.
        *args: Positional arguments to pass to the operation.
        **kwargs: Keyword arguments to pass to the operation.

    Returns:
        The result of the operation if successful.

    Raises:
        Exception: If the operation fails after max_retries attempts.
    """
    retries = 0
    while True:
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise Exception(f"Operation failed after {max_retries} attempts. Last error: {str(e)}")
            
            delay = min(max_delay, (2 ** retries * base_delay) + (random.random() * base_delay))
            print(f"Operation failed: {str(e)}. Retrying in {delay:.2f} seconds (attempt {retries}/{max_retries})")
            await asyncio.sleep(delay)
