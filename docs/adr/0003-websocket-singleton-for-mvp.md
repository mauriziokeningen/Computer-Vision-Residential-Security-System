# ADR 0003: Use of In-Memory Singleton for WebSockets (MVP Phase)

**Date:** April 9, 2026
**Status:** Accepted (Technical Debt Logged)
**Authors:** Armando, Mauricio

## Context

The residential security system requires real-time push notifications with ultra-low latency (sub-second) to alert operators when AI modules (computer vision) detect anomalies such as weapons, unknown faces, or aggressive poses.

To achieve this, we implemented a WebSocket server using FastAPI's native technology. However, an architectural question arises regarding how to store and manage active client connections (guards' browsers) so that the system can iterate quickly toward a Minimum Viable Product (MVP).

## Decision

We have decided to implement WebSocket connection management using a **local in-memory Singleton** (`ConnectionManager` instantiated at the application level) that stores active connections in a standard list structure (`List[WebSocket]`).

The immediate introduction of an external Message Broker (such as Redis Pub/Sub or Apache Kafka) was ruled out to avoid infrastructure and configuration overhead during the initial system validation phase.

## Consequences

### Positive
* **Development Speed:** Allows for immediate end-to-end system deployment without external infrastructure dependencies.
* **Simplicity:** Reduces the complexity of `docker-compose.yml` and facilitates local testing for any developer cloning the repository.

### Negative (Technical Risk)
* **Horizontal Scaling Block (Stateful Trap):** Connection state lives exclusively in the RAM of the current process (worker).
* **Load Balancing Failure:** If the application is deployed in multiple containers behind a load balancer (e.g., Nginx/AWS ALB), an alert generated in Container A will not be transmitted to users connected to the WebSockets in Container B. Notifications would be silently lost.

## Mitigation Plan (Production Roadmap)

This decision is accepted **strictly for single-node environments**.

It is recorded as **Critical Technical Debt**. Before migrating the architecture to a distributed cluster (Kubernetes / Multiple cloud instances), the `ConnectionManager` **must be refactored** to adopt a **Fan-out** pattern.

**Target Technology:** Redis Pub/Sub.
* FastAPI workers will act as **Publishers** of events to Redis.
* All workers will simultaneously act as **Subscribers**, listening to the Redis channel to retransmit the final payload to their local in-memory connections.

*(See GitHub Issue #XXX for tracking this migration).*