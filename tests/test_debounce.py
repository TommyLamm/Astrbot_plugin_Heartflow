def test_harness_loads_plugin(plugin_factory):
    plugin = plugin_factory()
    assert plugin.debounce_seconds == 0.01
