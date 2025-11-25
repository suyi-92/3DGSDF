"""
Minimal, self-contained example showing how to use AdapterBase and ExchangeBus.
最小可运行示例，演示如何使用 AdapterBase 与 ExchangeBus。

运行方式：
    python -m fusion.adapter_bus_minimal_example

要点：
- 定义两个适配器，一个发布消息，一个订阅消息。
- 通过 register_api 注册可调用 API，并用 APIRegistry.call 调用。
- ExchangeBus 用于在适配器间传递中间结果，无需直接依赖。
"""

from __future__ import annotations

from typing import Any

from fusion.common import AdapterBase, APIRegistry, ExchangeBus


class DummyDataService:
    """Placeholder data service used just for the example."""

    def __init__(self):
        self.saved_results = []

    def save(self, item: Any):
        self.saved_results.append(item)


class RenderAdapter(AdapterBase):
    """Publishes render statistics through the bus and exposes a render API."""

    def __init__(self, registry: APIRegistry, bus: ExchangeBus, data_service: DummyDataService):
        super().__init__("render", registry, bus, data_service)
        self.register_api("render", self.render, "Render a frame and publish stats")

    def render(self, frame_id: int):
        # 假装渲染了一帧并产生统计信息
        stats = {"frame": frame_id, "loss": 0.123}
        # 发布统计信息，供其他适配器订阅处理
        self.bus.publish("render.stats", stats)
        return stats


class StatsAdapter(AdapterBase):
    """Subscribes to render statistics and records them in the data service."""

    def __init__(self, registry: APIRegistry, bus: ExchangeBus, data_service: DummyDataService):
        super().__init__("stats", registry, bus, data_service)
        # 注册一个 API，可用于查询保存的统计信息
        self.register_api("get", self.get_all, "Return collected stats")
        # 订阅渲染统计，当收到消息时写入 data_service
        self.bus.subscribe("render.stats", self._on_render_stats)

    def _on_render_stats(self, payload):
        self.data_service.save({"source": self.name, **payload})

    def get_all(self):
        return list(self.data_service.saved_results)


def main():
    registry = APIRegistry()
    bus = ExchangeBus()
    data_service = DummyDataService()

    render = RenderAdapter(registry, bus, data_service)
    stats = StatsAdapter(registry, bus, data_service)

    # 通过注册表调用 render API，触发一次渲染与发布
    result = registry.call("render.render", frame_id=1)
    print("Render result:", result)

    # 再通过注册表调用 stats API，查看订阅收到的内容
    collected = registry.call("stats.get")
    print("Collected stats:", collected)


if __name__ == "__main__":
    main()
