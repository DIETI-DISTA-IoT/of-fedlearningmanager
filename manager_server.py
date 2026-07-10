import logging
import atexit
import traceback
import faulthandler
import signal
import sys
import string
import random
import pickle
import json
import threading
import hashlib

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.admin import AdminClient

from OpenFAIR.container_api import ContainerAPI
from modules import build_model
from preprocessing import GenericBuffer
from reporting import WeightsReporter, GlobalMetricsReporter
from aggregation import federated_averaging, FedYogi, fed_median, fed_prox

FL_MANAGER = "FED_LEARNING_MANAGER"
FEDERATED_LEARNING = "FEDERATED_LEARNING"

aggregation_functions = {
    "fedavg": federated_averaging,
    "fedyogi": FedYogi,
    "fedmedian": fed_median,
    "fedprox": fed_prox,
}


def _model_signature(state_dict):
    """Compact, deterministic fingerprint of a model's parameters (for logging).

    Returns (n_params, n_tensors, sha10, keys). sha10 hashes the 'key:shape'
    list, so two state-dicts share a signature iff they have the same parameter
    names AND shapes — i.e. the same architecture/topology. Never raises.
    """
    try:
        items = [(k, tuple(v.shape)) for k, v in state_dict.items()]
        n_params = int(sum(v.numel() for v in state_dict.values()))
        canon = ';'.join('%s:%s' % (k, 'x'.join(map(str, shp))) for k, shp in items)
        sha = hashlib.sha1(canon.encode()).hexdigest()[:10]
        return n_params, len(items), sha, [k for k, _ in items]
    except Exception as exc:  # diagnostics must never crash aggregation
        return -1, -1, 'ERR(%r)' % exc, []



def _delete_topics(kafka_broker_url, topics, logger):
    """Delete a list of Kafka topics, logging each result. Tolerates Kafka being down."""
    try:
        admin = AdminClient({'bootstrap.servers': kafka_broker_url})
        futures = admin.delete_topics(topics, operation_timeout=10)
        for topic, future in futures.items():
            try:
                future.result()
                logger.info(f"Deleted Kafka topic: {topic}")
            except Exception as e:
                logger.warning(f"Could not delete topic {topic} (may not exist or Kafka down): {e}")
    except Exception as e:
        logger.warning(f"Topic deletion failed (Kafka may be down): {e}")


class FederatedLearningManager:

    def __init__(self, args):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=str(args['logging_level']).upper())
        self.logger = logging.getLogger(FEDERATED_LEARNING)
        self.logger.info("Initializing federated learning manager")
        self.aggregation_interval_secs = args['aggregation_interval_secs']
        self.kafka_broker_url = args['kafka_broker_url']

        self.global_model = build_model(**args)
        self.global_model.initialize_weights(args['initialization_strategy'])
        self.logger.info(
            f"Global model ({str(args.get('model_type', 'mlp')).lower()}) initialized "
            f"using {args['initialization_strategy']} initialization."
        )

        # Resolve the aggregation strategy once, per FL session.  Stateful
        # strategies (e.g. FedYogi) are registered as classes and must be
        # instantiated so the call routes through ``__call__`` instead of
        # ``__init__``; instantiating here (rather than at module import)
        # also gives each FL session — and each architecture switch — a fresh
        # optimizer state instead of leaking buffers across runs.
        strategy = args.get('aggregation_strategy')
        try:
            aggregator = aggregation_functions[strategy]
        except KeyError:
            raise ValueError(
                f"Unknown aggregation strategy '{strategy}'. "
                f"Available: {sorted(aggregation_functions)}"
            )
        self.aggregation_function = aggregator() if isinstance(aggregator, type) else aggregator

        self.admin_client = AdminClient({'bootstrap.servers': args['kafka_broker_url']})
        _delete_topics(self.kafka_broker_url, ["global_weights"], self.logger)
        self.vehicle_weights_topics = self.check_vehicle_weights_topics(args)

        self.weights_buffer = self.create_weights_buffer(**args)
        self.weights_reporter = WeightsReporter(self.logger, **args)
        self.global_metrics_reporter = GlobalMetricsReporter(self.logger, **args)

        self.logger.info(f"Starting FL with {len(self.vehicle_weights_topics)}" + \
                         f" for vehicles: {self.vehicle_weights_topics}")
        self.stop_threads = False
        self._stop_event = threading.Event()
        self.aggregation_thread = None

        self.consuming_thread = threading.Thread(
            target=self.consume_weights_data,
            kwargs=args,
            daemon=True
            )

        self.consuming_thread.start()

        if self.aggregation_interval_secs > 0:
            self.aggregation_thread = threading.Thread(
                target=self.aggregate_weights_periodically,
                kwargs=args)
            self.aggregation_thread.start()


    def graceful_shutdown(self):
        self.stop_threads = True
        self._stop_event.set()

        join_timeout = max(self.aggregation_interval_secs + 5, 15)
        if self.consuming_thread:
            self.consuming_thread.join(timeout=join_timeout)
            if self.consuming_thread.is_alive():
                self.logger.warning("consuming_thread did not stop within timeout — proceeding.")
            else:
                self.logger.info("consuming_thread stopped.")
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=join_timeout)
            if self.aggregation_thread.is_alive():
                self.logger.warning("aggregation_thread did not stop within timeout — proceeding.")
            else:
                self.logger.info("aggregation_thread stopped.")
        _delete_topics(
            self.kafka_broker_url,
            ["global_weights", "global_metrics"],
            self.logger
        )
        self.logger.info(f"Federated learning manager stopped.")


    def aggregate_weights_periodically(self, **kwargs):
        while not self.stop_threads:
            self._stop_event.wait(timeout=kwargs.get('aggregation_interval_secs'))
            if self.stop_threads:
                break
            # Guard the round so a single failure logs a traceback and retries
            # instead of silently terminating the aggregation thread for the run.
            try:
                self.aggregate_weights(**kwargs)
            except Exception:
                self.logger.exception("Aggregation round failed; will retry on next interval.")


    def consume_weights_data(self, **kwargs):

        consumer = self.create_consumer(**kwargs)
        consumer.subscribe(self.vehicle_weights_topics)
        self.logger.info(f"will start consuming {self.vehicle_weights_topics}")

        try:
            while not self.stop_threads:
                msg = consumer.poll(5.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        self.logger.info(f"End of partition reached for {msg.topic()}")
                    else:
                        self.logger.error(f"consumer error: {msg.error()}")
                    continue

                deserialized_data = self.deserialize_message(msg)
                if deserialized_data:
                    self.process_message(msg.topic(), deserialized_data, **kwargs)

        except KeyboardInterrupt:
            self.logger.info(f" Consumer interrupted by user.")
        except Exception as e:
            self.logger.error(f" Error in consumer: {e}")
        finally:
            consumer.close()
            self.logger.info(f" Consumer closed.")


    def create_consumer(self, **kwargs):
        def generate_random_string(length=10):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for i in range(length))
        conf_cons = {
            'bootstrap.servers': kwargs.get('kafka_broker_url'),
            'group.id': kwargs.get('kafka_consumer_group_id', 'FEDERATED_LEARNING') + generate_random_string(7),
            'auto.offset.reset': kwargs.get('kafka_auto_offset_reset')
        }
        return Consumer(conf_cons)


    def deserialize_message(self, msg):

        try:
            message_value = pickle.loads(msg.value())
            self.logger.debug(f"received message from topic [{msg.topic()}]")
            return message_value
        except json.JSONDecodeError as e:
            self.logger.error(f"Error deserializing message: {e}")
            return None


    def aggregate_weights(self, **kwargs):

        if all([len(buffer) > 0 for buffer in self.weights_buffer.values()]):
            self.logger.info(f"Aggregating the weights from {len(self.weights_buffer)} vehicles.")
            aggregation_function = self.aggregation_function

            # ───────────── aggregation diagnostics (logging only) ─────────────
            strategy = kwargs.get('aggregation_strategy')
            self._agg_round = getattr(self, '_agg_round', 0) + 1
            self.logger.info("========== FL AGGREGATION ROUND %d ==========" % self._agg_round)
            self.logger.info(
                "strategy=%r | resolved=%s | kind=%s | model_type_cfg=%s"
                % (strategy,
                   getattr(aggregation_function, '__name__', type(aggregation_function).__name__),
                   'CLASS (instantiated on each call)' if isinstance(aggregation_function, type)
                   else type(aggregation_function).__name__,
                   kwargs.get('model_type', '<not in FL config>')))
            _g_params, _g_n, _g_sha, _g_keys = _model_signature(self.global_model.state_dict())
            self.logger.info("GLOBAL model: class=%s n_tensors=%d n_params=%d sig=%s"
                             % (type(self.global_model).__name__, _g_n, _g_params, _g_sha))
            _client_shas = []
            for _topic, _buf in self.weights_buffer.items():
                _c_params, _c_n, _c_sha, _c_keys = _model_signature(_buf.get())
                _client_shas.append(_c_sha)
                self.logger.info("CLIENT %-24s n_tensors=%d n_params=%d sig=%s -> %s"
                                 % (_buf.label, _c_n, _c_params, _c_sha,
                                    'MATCHES global' if _c_sha == _g_sha else 'DIFFERS from global !!'))
                if _c_sha != _g_sha:
                    self.logger.warning("  topology mismatch %s: client-only keys=%s | global-only keys=%s"
                                        % (_buf.label,
                                           [k for k in _c_keys if k not in _g_keys][:6],
                                           [k for k in _g_keys if k not in _c_keys][:6]))
            self.logger.info("client signatures: %d distinct across %d clients -> %s"
                             % (len(set(_client_shas)), len(_client_shas),
                                'HOMOGENEOUS' if len(set(_client_shas)) == 1 else 'HETEROGENEOUS'))
            # ──────────────────────────────────────────────────────────────────


            for buffer in self.weights_buffer.values():
                candidate_state_dict = buffer.get()
                if any(torch.isnan(param).any() for param in candidate_state_dict.values()):
                    self.logger.error(f"Candidate weights from {buffer.label} contain NaNs. Skipping update.")
                    return

            if aggregation_function is federated_averaging:
                aggregated_state_dict = aggregation_function(
                    self.global_model.state_dict(),
                    [buffer.get() for buffer in self.weights_buffer.values()])
            else:
                aggregated_state_dict = aggregation_function(
                    self.global_model.state_dict(),
                    [buffer.get() for buffer in self.weights_buffer.values()], **kwargs)

            if any(torch.isnan(param).any() for param in aggregated_state_dict.values()):
                self.logger.error("Aggregated state dict contains NaNs. Skipping update.")
                return
            for buffer in self.weights_buffer.values():
                buffer.pop()
            self.global_model.load_state_dict(aggregated_state_dict)
            self.weights_reporter.push_weights(self.global_model.state_dict())

            # ───────────── aggregation diagnostics (logging only) ─────────────
            if isinstance(aggregation_function, FedYogi):
                self.logger.info("FedYogi INSTANCE ran: step=%s | moments_initialised=%s"
                                 % (aggregation_function.step, aggregation_function.moment_1 is not None))
            _a_params, _, _a_sha, _ = _model_signature(aggregated_state_dict)
            self.logger.info("AGGREGATION OK via '%s' -> pushed global weights "
                             "(n_params=%d sig=%s) round=%d"
                             % (kwargs.get('aggregation_strategy'), _a_params, _a_sha, self._agg_round))
            self.logger.info("=============================================")
            # ──────────────────────────────────────────────────────────────────
        else:
            self.logger.info(f"Waiting for more data to aggregate the weights.")


    def process_message(self, topic, msg, **kwargs):

        self.weights_buffer[topic].add(msg)

        if kwargs.get('aggregation_interval_secs') == 0:
            self.aggregate_weights(**kwargs)


    def create_weights_buffer(self, **kwargs):
        weights_buffer = {}
        for topic in self.vehicle_weights_topics:
            weights_buffer[topic] = GenericBuffer(size=kwargs.get('weights_buffer_size', 3), label=topic)
        return weights_buffer


    def check_vehicle_weights_topics(self, args):
        existing_topics = self.admin_client.list_topics(timeout=10).topics.keys()
        vehicle_topics = [topic for topic in existing_topics if topic.endswith("_weights") and topic != "global_weights"]
        self.logger.debug("Found the following vehicle topics: %s", vehicle_topics)
        return vehicle_topics


class FedLearningManagerAPI(ContainerAPI):

    def __init__(self, port: int = 5000):
        super().__init__(
            container_type='fedlearningmanager',
            container_name='fedlearningmanager',
            port=port
            )
        self.fl_instance = None


    def get_detailed_status(self):
        status = {"federated_learning_running": self.fl_instance is not None}
        if self.fl_instance is not None:
            # global_metrics is W&B-bound and never lossy (see reporting.py);
            # only global_weights (the actual training signal) carries loss.
            status["packet_loss"] = {
                "global_weights": self.fl_instance.weights_reporter.packet_loss.stats(),
            }
        return status


    def handle_command(self, command, params):
        if command == "start_federated_learning":
            if self.fl_instance is None:
                self.fl_instance = FederatedLearningManager(params)
                return "Succesfully started federated learning"
            else:
                return "Federated learning is already running"
        elif command == "stop_federated_learning":
            if self.fl_instance is not None:
                instance = self.fl_instance
                self.fl_instance = None
                instance.graceful_shutdown()
                return "Succesfully stopped federated learning"
            else:
                return "Federated learning is not running"
        else:
            return "Unrecognized command"


def signal_handler(sig, frame):
    global api
    print(f"{FL_MANAGER}: Received signal {sig}. Gracefully stopping federated learning and its threads.", flush=True)
    if api.fl_instance is not None:
        api.fl_instance.graceful_shutdown()
    sys.exit(0)


def _atexit_handler():
    print("[FED_LEARNING_MANAGER][atexit] Process is exiting. Stack trace at exit time:", flush=True)
    for line in traceback.format_stack():
        print(line, end='', flush=True)


def main():
    global api
    api = FedLearningManagerAPI()

    faulthandler.enable()

    atexit.register(_atexit_handler)
    signal.signal(signal.SIGTERM, lambda sig, frame: (
        print(f"[FED_LEARNING_MANAGER] SIGTERM received — calling signal_handler.", flush=True),
        signal_handler(sig, frame)
    ))
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

    api.run()


if __name__ == "__main__":
    main()
