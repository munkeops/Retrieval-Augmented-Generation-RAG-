from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate


def chatbot_llm_chain(prompt):
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    n_gpu_layers = 15000  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="./llama-2-7b.Q8_0.gguf",
        temperature=0.75,
        max_tokens=2500,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        # verbose=True,  # Verbose is required to pass to the callback manager
    )

    return llm

    # llm_chain = LLMChain(prompt=prompt, llm=llm)

    # return llm_chain
