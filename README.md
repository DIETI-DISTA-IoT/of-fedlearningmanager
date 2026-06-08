# of-fedlearningmanager

Federated Learning aggregation manager node for the **SereBench** platform.

This container hosts the `FederatedLearningManager`, extracted from the
`Wandber` node so that federated learning can be started and stopped
independently of observability/W&B logging during a run.

## Responsibilities

- Subscribes to all `{vehicle_name}_weights` Kafka topics published by vehicle
  consumers.
- Periodically aggregates the collected local weights into a global model
  using a pluggable aggregation strategy (`FedAvg`, `FedYogi`, `FedMedian`,
  `FedProx`).
- Publishes the aggregated global weights to the `global_weights` Kafka topic.
- Exposes a Flask REST API (via `ContainerAPI` from `of-core`, port 5000)
  through which the dashboard can `start_federated_learning` /
  `stop_federated_learning` at runtime.

## Branch policy

This repository is part of the SereBench platform. All SereBench-related
development happens on the `sereBench` branch.

## Running

The container is built and orchestrated as part of the root `SereBench`
docker-compose cluster — see the root repository's `Makefile` and
`docker-compose.yml`. The dashboard sends an HTTP `/command` request with
`start_federated_learning` / `stop_federated_learning` to control this node
at runtime, exactly as it does for the other containers (producers,
consumers, wandber).
