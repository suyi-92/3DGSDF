# AdapterBase + ExchangeBus 详解与改造指南

本文基于 `fusion/adapter_bus_minimal_example.py` 的最小示例，逐步拆解内部逻辑、调用路径，并给出如何扩展或修改的建议。通过对照代码运行输出，可以直观看到 API 注册、消息发布/订阅的工作方式。

## 核心对象
- **APIRegistry**：保存可调用端点。适配器通过 `register_api(endpoint, func, description)` 把方法注册成 `"{adapter}.{endpoint}"` 这样的全局名字，外部调用统一使用 `registry.call(full_name, **kwargs)`。
- **ExchangeBus**：轻量级发布/订阅总线，`publish(topic, payload)` 将消息广播到订阅了该主题的回调；`subscribe(topic, callback)` 为主题添加订阅者。
- **AdapterBase**：适配器基类。保存 `registry`、`bus` 和 `data_service` 引用，并封装 `register_api`，简化子类开发。

## 示例运行流程（按时间顺序）
1. **初始化**
   ```python
   registry = APIRegistry()
   bus = ExchangeBus()
   data_service = DummyDataService()
   render = RenderAdapter(registry, bus, data_service)
   stats = StatsAdapter(registry, bus, data_service)
   ```
   - `RenderAdapter` 在 `__init__` 中调用 `register_api("render", self.render, ...)`，因此可通过 `render.render` 远程调用。
   - `StatsAdapter` 在 `__init__` 中完成两件事：
     1) 注册查询接口 `stats.get`；
     2) `bus.subscribe("render.stats", self._on_render_stats)` 监听渲染统计。

2. **调用渲染 API**
   ```python
   registry.call("render.render", frame_id=1)
   ```
   - `APIRegistry` 找到对应的函数并执行 `RenderAdapter.render(frame_id=1)`。
   - `render()` 构造统计字典并调用 `bus.publish("render.stats", stats)`。
   - `ExchangeBus` 遍历该主题的回调列表，依次执行（此处就是 `StatsAdapter._on_render_stats`）。

3. **订阅回调处理**
   ```python
   def _on_render_stats(self, payload):
       print(f"[StatsAdapter] 收到 render.stats: {payload}")
       self.data_service.save({"source": self.name, **payload})
   ```
   - 回调收到 `payload`，打印确认并写入共享的 `data_service`。

4. **查询结果**
   ```python
   registry.call("stats.get")
   ```
   - 直接返回 `DummyDataService` 中累计的记录，证明发布/订阅链路已生效。

运行时的打印顺序与数据流完全对应上述步骤，方便对照理解。

## 如何扩展或修改
- **增加新的业务接口**：在对应适配器中新增方法，并调用 `self.register_api("endpoint", self.new_method, "描述")`。之后即可用 `registry.call("{adapter}.endpoint", ...)` 访问。
- **新增消息主题或数据流**：
  1) 选择发布方，在逻辑合适的位置 `self.bus.publish("topic.name", payload)`；
  2) 在需要消费的适配器初始化时调用 `self.bus.subscribe("topic.name", self.callback)` 并实现回调。
- **替换存储或服务**：`data_service` 通过依赖注入传入，若需要写入数据库/文件，只需实现相同接口（如 `save` 方法）并在创建适配器时传入新的实例。
- **排查或理解执行顺序**：利用示例中的打印风格，给自己的回调和 API 加上清晰的标记输出，运行脚本即可看到完整链路。

## 快速验证
```bash
python -m fusion.adapter_bus_minimal_example
```
观察输出：
- `[Main] 调用 API: render.render(frame_id=1)`
- `[RenderAdapter] 发布 render.stats: ...`
- `[StatsAdapter] 收到 render.stats: ...`
- `Collected stats: [...]`
上述顺序即是完整的数据流与调用链。

通过这份示例和讲解，可以在不改动核心框架的情况下自信地增加新适配器、接口或消息流。
