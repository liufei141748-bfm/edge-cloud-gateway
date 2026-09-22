import httpx
import pytest

openai = pytest.importorskip("openai")

from edge_cloud_gateway.adapters import MockAdapter
from edge_cloud_gateway.app import create_app
from edge_cloud_gateway.config import Settings
from edge_cloud_gateway.storage import Store


@pytest.mark.asyncio
async def test_standard_openai_python_sdk_uses_chat_completions_without_gateway_fields():
    app = create_app(Settings(), cloud=MockAdapter("cloud"), local=MockAdapter("local"),
                     store=Store(":memory:"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        client = openai.AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="local-placeholder",
            http_client=http_client,
        )
        completion = await client.chat.completions.create(
            model="adaptive",
            messages=[{"role": "user", "content": "hello from sdk"}],
        )
        assert completion.object == "chat.completion"
        assert completion.choices[0].message.content
        await client.close()
