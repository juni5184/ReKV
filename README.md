# ReKV

Official PyTorch code of "Streaming Video Question-Answering with In-context Video KV-Cache Retrieval", *ICLR* 2025.

Thanks for your attention! Our code will be released this weekend.

## Abstract

We propose **ReKV**, a novel training-free approach that seamlessly integrates with existing Video-LLMs to enable efficient streaming video question-answering (**StreamingVQA**).

Traditional VideoQA systems struggle with long videos, as they must process the entire video before responding to queries, and repeat this process for each new question. In contrast, our approach analyzes long videos streamingly, allowing for prompt responses as soon as user queries are received. 
- Building on a common Video-LLM, we first incorporate a sliding-window attention mechanism, ensuring that input frames attend to a limited number of preceding frames, thereby reducing computational overhead.
- To prevent information loss, we store processed video key-value caches (KV-Caches) in RAM and disk, reloading them into GPU memory as needed. 
- Additionally, we introduce a retrieval method that leverages an external retriever or the parameters within Video-LLMs to retrieve only query-relevant KV-Caches, ensuring both efficiency and accuracy in question answering.
