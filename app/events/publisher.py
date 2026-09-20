"""Broker-independent event publishing with local test sinks."""

import logging
import time
from pathlib import Path
from typing import Protocol

from app.events.schemas import RecommendationEvent


logger = logging.getLogger("uvicorn.error")


class EventPublishError(RuntimeError):
    pass


class EventSink(Protocol):
    def send(self, event: RecommendationEvent) -> None:
        """Send one event or raise an exception."""


class LoggingEventSink:
    """Temporary sink for local development before broker infrastructure exists."""

    def send(self, event: RecommendationEvent) -> None:
        logger.info("Published event: %s", event.model_dump_json())


class JsonLinesEventSink:
    """Append each event as one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, event: RecommendationEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as event_file:
            event_file.write(event.model_dump_json() + "\n")


class CompositeEventSink:
    """Publish each event to all configured sinks."""

    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = sinks

    def send(self, event: RecommendationEvent) -> None:
        for sink in self.sinks:
            sink.send(event)


class InMemoryEventSink:
    """Test sink that retains successfully published events."""

    def __init__(self) -> None:
        self.events: list[RecommendationEvent] = []

    def send(self, event: RecommendationEvent) -> None:
        self.events.append(event)


class EventPublisher:
    def __init__(
        self,
        sink: EventSink,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.25,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.sink = sink
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds

    def publish(self, event: RecommendationEvent) -> None:
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.sink.send(event)
                return
            except Exception as exc:
                logger.warning(
                    "Event publish failed for %s on attempt %s/%s: %s",
                    event.eventId,
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if attempt < self.max_attempts and self.retry_delay_seconds > 0:
                    time.sleep(self.retry_delay_seconds)

        raise EventPublishError(
            f"Failed to publish event {event.eventId} after {self.max_attempts} attempts"
        )


EVENTS_FILE = Path(__file__).resolve().parent / "events.jsonl"
default_publisher = EventPublisher(
    CompositeEventSink(
        LoggingEventSink(),
        JsonLinesEventSink(EVENTS_FILE),
    )
)


def publish_safely(event: RecommendationEvent) -> bool:
    """Publish without failing an already generated recommendation."""
    try:
        default_publisher.publish(event)
        return True
    except EventPublishError:
        logger.exception("Event %s could not be published", event.eventId)
        return False