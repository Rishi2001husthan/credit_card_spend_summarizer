from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

llm_vlm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def get_image_description(img_b64: str, caption: str = "") -> str:
    """
    Uses a VLM to generate a searchable description of the image.
    """

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You are an expert document analyst. "
                        "Describe this image so it can be used for semantic search. "
                        "Focus on objects, relationships, charts, and meaning. "
                        "Be concise but information-dense."
                        + (f"\nCaption: {caption}" if caption else "")
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    },
                },
            ]
        )
    ]

    response = llm_vlm.invoke(messages)
    return response.content.strip()