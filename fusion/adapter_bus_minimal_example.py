"""
Minimal, self-contained example showing how to use AdapterBase and ExchangeBus.
最小可运行示例，演示 AdapterBase 与 ExchangeBus 的内在逻辑与调用流程。

运行方式：
    python -m fusion.adapter_bus_minimal_example

阅读提示：
- 每一步都用注释和打印语句标出，便于跟踪执行顺序。
- render 适配器发布“渲染统计”主题，stats 适配器订阅并写入 data_service。
- 所有对外暴露的接口都通过 APIRegistry.call 访问，避免耦合到具体类。
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

    def __init__(
        self, registry: APIRegistry, bus: ExchangeBus, data_service: DummyDataService
    ):
        super().__init__("render", registry, bus, data_service)
        self.register_api("render", self.render, "Render a frame and publish stats")

    def render(self, frame_id: int):
        # 假装渲染了一帧并产生统计信息
        stats = {"frame": frame_id, "loss": 0.123}
        # 发布统计信息，供其他适配器订阅处理
        # 主题名采用 "命名空间.具体含义" 形式：render 表示来源（渲染相关），stats 表示内容是统计数据
        print("[RenderAdapter] 发布 render.stats: ", stats)
        self.bus.publish("render.stats", stats)
        return stats


class StatsAdapter(AdapterBase):
    """Subscribes to render statistics and records them in the data service."""

    def __init__(
        self, registry: APIRegistry, bus: ExchangeBus, data_service: DummyDataService
    ):
        super().__init__("stats", registry, bus, data_service)
        # 注册一个 API，可用于查询保存的统计信息
        self.register_api("get", self.get_all, "Return collected stats")
        # 订阅渲染统计，当收到消息时写入 data_service
        self.bus.subscribe("render.stats", self._on_render_stats)

    def _on_render_stats(self, payload):
        print(f"[StatsAdapter] 收到 render.stats: {payload}")
        self.data_service.save({"source": self.name, **payload})

    def get_all(self):
        return list(self.data_service.saved_results)


def main():
    registry = APIRegistry()
    bus = ExchangeBus()
    data_service = DummyDataService()

    render = RenderAdapter(registry, bus, data_service)
    stats = StatsAdapter(registry, bus, data_service)

    # 打印可用 API
    apis = registry.describe()
    print(f"已注册 {len(apis)} 个 API:")
    for name, desc in list(apis.items())[:5]:
        print(f"  - {name}: {desc}")
    if len(apis) > 5:
        print(f"  ... 以及其他 {len(apis) - 5} 个 API")
    print()

    # 通过注册表调用 render API，触发一次渲染与发布。
    # 调用时无需直接持有 RenderAdapter 实例，只需知道完整端点名。
    print("[Main] 调用 API: render.render(frame_id=1)")
    result = registry.call("render.render", frame_id=1)
    print("Render result:", result)

    # 再通过注册表调用 stats API，查看订阅收到的内容。
    # 由于 StatsAdapter 在初始化时已订阅 render.stats，数据已经写入。
    print("[Main] 调用 API: stats.get()")
    collected = registry.call("stats.get")
    print("Collected stats:", collected)


if __name__ == "__main__":
    main()
