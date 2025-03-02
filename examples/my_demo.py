import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np

import nest_asyncio
nest_asyncio.apply()

WORKING_DIR = r"C:\Users\ZHAO YOU\ipynb\LightRAG\dickens-1"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 建立索引时用v3,问答时用r1
    return await openai_complete_if_cache(
        # "deepseek-v3-241226",
        "deepseek-r1-250120",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key='8c60aacc-d9e2-48de-bcb3-e2f9a291315e',
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        **kwargs,
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="doubao-embedding-large-text-240915",
        api_key='8c60aacc-d9e2-48de-bcb3-e2f9a291315e',
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

# async def llm_model_func(
#     prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
# ) -> str:
#     # 无并发限制, 建造索引时使用
#     return await openai_complete_if_cache(
#         "DMXAPI-DeepSeek-V3",
#         prompt,
#         system_prompt=system_prompt,
#         history_messages=history_messages,
#         api_key='sk-9MTkarjT4o0QsZBdEc46899b6d5846CdA595C1Fd6f8dF812',
#         base_url="https://www.DMXapi.com/v1",
#         **kwargs,
#     )

# async def embedding_func(texts: list[str]) -> np.ndarray:
#     return await openai_embed(
#         texts,
#         model="text-embedding-3-large",
#         api_key='sk-9MTkarjT4o0QsZBdEc46899b6d5846CdA595C1Fd6f8dF812',
#         base_url="https://www.DMXapi.com/v1",
#     )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("你好")
    print("llm_model_func: ", result)

    result = await embedding_func(["你好"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def main():
    # 执行 test_funcs() 以确保 llm_model_func 和 embedding_func 正常工作
    # await test_funcs()

    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            # llm_model_max_async=5,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                # max_token_size=8192,
                max_token_size=4096,  # 豆包模型最大支持4096
                func=embedding_func,
            ),
            addon_params={"language": "Simplfied Chinese"},
            chunk_token_size = 600,
        )

        QUESTION = "AEO企业多久重认一次？"
        path_file = r"C:\Users\ZHAO YOU\ipynb\通义千文\海关文档.txt"

        with open(path_file, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())


        query_types_list = [
            # "naive",
            # "local",
            # "global",
            "hybrid",
            # "mix"
        ]
        print(query_types_list)
        for query_type in query_types_list:
            print(f"-----------------{query_type} search-----------------")
            print(
                await rag.aquery(
                    QUESTION,
                    param=QueryParam(
                        mode=query_type,
                        only_need_context=False,
                        # top_k=200,
                        # history_turns=1,
                        # max_token_for_text_unit=6000,
                    ),
                )
            )
            print("=============================================")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 确保 finalize_storages 被正确调用
        if 'rag' in locals():
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
