"""Small interactive OpenAI SDK client for the local Gateway."""

from __future__ import annotations

try:
    from openai import OpenAI, OpenAIError
except ModuleNotFoundError as exc:
    raise SystemExit(
        "OpenAI SDK is not installed. Install it in the project virtual environment first."
    ) from exc


BASE_URL = "http://127.0.0.1:8787/v1"
API_KEY = "local-gateway-placeholder"
MODEL = "adaptive"


def build_request(messages: list[dict[str, str]], context: str | None) -> dict:
    request: dict = {"model": MODEL, "messages": messages}
    if context:
        request["extra_body"] = {"gateway_context": {"blocks": [{
            "id": "daily-reference",
            "source": "daily-client",
            "kind": "document",
            "content": context,
            "optional": True,
        }]}}
    return request


def read_context() -> str | None:
    print("粘贴参考资料；输入空行结束：")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    value = "\n".join(lines).strip()
    return value or None


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    history: list[dict[str, str]] = []
    context: str | None = None
    print("Gateway 日常客户端：/context 设置资料，/clear 清空资料，/quit 退出。")

    try:
        while True:
            try:
                question = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question == "/quit":
                break
            if question == "/context":
                context = read_context()
                print("参考资料已更新。" if context else "参考资料为空，已清除。")
                continue
            if question == "/clear":
                context = None
                print("参考资料已清除。")
                continue

            messages = history + [{"role": "user", "content": question}]
            try:
                raw = client.chat.completions.with_raw_response.create(
                    **build_request(messages, context)
                )
                completion = raw.parse()
                answer = completion.choices[0].message.content or ""
            except (OpenAIError, IndexError) as exc:
                print(f"请求失败：{exc}")
                continue

            print(f"助手> {answer}")
            route = raw.headers.get("x-gateway-route", "unknown")
            reason = raw.headers.get("x-gateway-reason", "unknown")
            print(f"[route] {route}；reason={reason}")
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
    finally:
        client.close()


if __name__ == "__main__":
    main()
