from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
import pickle
import json

from OpenFAIR.packet_loss import PacketLossSimulator
from OpenFAIR.network_delay import NetworkDelaySimulator

class WeightsReporter:
    def __init__(self, logger, **kwargs):
        conf_prod_weights={
        'bootstrap.servers': kwargs.get('kafka_broker_url'),  # Kafka broker URL
        'key.serializer': StringSerializer('utf_8'),
        'value.serializer': lambda v, ctx: pickle.dumps(v)
         }
        self.producer = SerializingProducer(conf_prod_weights)
        self.packet_loss = PacketLossSimulator(kwargs.get('packet_loss_rate', 0.1))
        # Simulated latency+jitter on the outbound global_weights update (same
        # policy as packet_loss — global_metrics is W&B-bound and never delayed,
        # see GlobalMetricsReporter).
        self.network_delay = NetworkDelaySimulator(
            kwargs.get('delay_mean_ms', 0.0), kwargs.get('jitter_std_ms', 0.0))
        self.logger = logger

    def push_weights(self, weights):
        weights_topic=f"global_weights"
        if self.packet_loss.should_drop():
            self.logger.debug(f"[packet-loss] dropped global weights update "
                              f"(rate={self.packet_loss.packet_loss_rate})")
            return

        def _deliver():
            try:
                self.producer.produce(topic=weights_topic, value=weights)
                self.producer.flush()
                self.logger.info(f"Sent global weights to topic: {weights_topic}")
            except Exception as e:
                self.logger.error(f"Failed to send global weights: {e}")

        self.network_delay.send(_deliver)


class GlobalMetricsReporter:
    # Publishes to global_metrics, which Wandber subscribes to directly for
    # W&B logging (aggregation-round diagnostics). Never subject to simulated
    # packet loss — only the global_weights actually used for training are.
    def __init__(self, logger, **kwargs):
        conf_prod_weights={
        'bootstrap.servers': kwargs.get('kafka_broker_url'),  # Kafka broker URL
        'key.serializer': StringSerializer('utf_8'),
        'value.serializer': lambda v, ctx: json.dumps(v)
         }
        self.producer = SerializingProducer(conf_prod_weights)
        self.logger = logger

    def report_metrics(self, metrics):
        global_metrics_topic=f"global_metrics"

        try:
            self.producer.produce(topic=global_metrics_topic, value=metrics)
            self.producer.flush()
            self.logger.info(f"Sent global weights to topic: {global_metrics_topic}")
        except Exception as e:
            self.logger.error(f"Failed to send global weights: {e}")
