# What is in this folder?

- `examples`: Sample usages of Aria data provider for reading Aria sequences.
- `players`: Player implementations for every Aria stream, which allows storing/retrieving VRS configuration/data records.
- `AriaVrsDataProvider.h/.cpp`: A VRS data provider class `AriaVrsDataProvider` with an interface that enables attaching players to a `vrs::RecordFileReader` instance and exposing its basic operations.
- `AriaDataProvider.h`: The base class `AriaDataProvider` that folder data provider class under `examples` and VRS data provider class inherit. Primarily defined for enabling the usage of functionality that exists in both data provider types.
- `utils.h/cpp`: Contains helper functions for data provider.
