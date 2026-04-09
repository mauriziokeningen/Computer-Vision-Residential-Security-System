# Integration Testing Roadmap: Business Rules (RN)

This document outlines the current testing coverage for the Security System's Incident Rule Engine. Several test cases are classified as "Pending" or "Blocked" due to dependencies on upstream modules currently under development.

## 1. Current Coverage Status

| Rule ID | Description | Status | Verification Method |
| :--- | :--- | :--- | :--- |
| **RN-01** | Authorized Access (Resident) | PENDING | Validate access_logs entry; Verify no Incident creation. |
| **RN-02** | Unknown Person Detected | PASSED | test_incident_creation_rules |
| **RN-03** | Repeated Failed Access | BLOCKED | Requires Stateful Counter and Cache Integration. |
| **RN-04** | Aggression Detected | PASSED | test_incident_creation_rules |
| **RN-05** | Fall Detected | PASSED | test_incident_creation_rules |
| **RN-06** | Unknown Face + Weapon | PENDING | Requires Event Correlation (Time-window logic). |
| **RN-07** | Known Resident + Fall | PENDING | Requires Event Correlation (Time-window logic). |
| **WEAPON**| Weapon Detected (General) | PASSED | test_incident_creation_rules |

---

## 2. Technical Dependencies and Blocked Implementations

### 2.1 Authorized Access (RN-01)
- **Dependency:** `feat/vector-search-engine`
- **Requirement:** Implementation of the pgvector similarity search.
- **Validation Logic:** Once a face embedding is matched against the `persons` table with a cosine distance < 0.6, the system must record a successful entry in the audit logs without triggering a security incident.

### 2.2 Event Correlation (RN-06 & RN-07)
- **Dependency:** Orchestrator Temporal Logic.
- **Requirement:** Implementation of a sliding time window for multi-module event processing.
- **Validation Logic:** Test sequences of asynchronous events. For example, an `Unknown Face` event followed within 10 seconds by a `Weapon Detected` event must result in a single `CRITICAL` priority incident rather than two independent alerts.

### 2.3 Stateful Tracking (RN-03)
- **Dependency:** Re-identification (RE-ID) and Metadata Persistence.
- **Requirement:** Distributed cache or state manager to track unique subject appearances.
- **Validation Logic:** Verify that the third occurrence of a specific unknown subject within a 300-second window triggers a priority escalation from `MEDIUM` to `HIGH`.