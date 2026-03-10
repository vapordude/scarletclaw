# Performance Optimization Rationale: Async Secret Audit

## Optimization Summary
Refactored the secrets audit process to use asynchronous I/O and parallelized file scanning.

## Problem
The previous implementation of the secrets audit relied heavily on synchronous file system operations (`fs.readFileSync`, `fs.existsSync`, `fs.readdirSync`). In Node.js, synchronous I/O blocks the event loop, preventing the application from handling other tasks (like network requests or UI updates) until the I/O operation completes. While often negligible for small files, this can become a bottleneck as the number of agents, auth profiles, and configuration files grows.

## Implementation Details
1.  **Async I/O:** Switched from `fs.*Sync` methods to their asynchronous counterparts (`fs.promises.*`). This ensures that the event loop remains unblocked during file reads and directory scans.
2.  **Parallelization:** Leveraged `Promise.all` to parallelize independent I/O tasks. Specifically:
    *   Scanning multiple `auth-profiles.json` files is now performed in parallel.
    *   Scanning multiple `models.json` files for agents is now performed in parallel.
    *   Scanning `.env` and legacy `auth.json` files is performed in parallel with other audit tasks.

## Expected Benefits
*   **Responsiveness:** Improved overall application responsiveness by avoiding event loop starvation.
*   **Efficiency:** Reduced wall-clock time for the audit process, especially in environments with many agents or large configuration sets, by overlapping I/O operations.

## Measurement Notes
A quantitative baseline measurement was impractical in the current development environment due to restricted network access and missing benchmarking dependencies. However, switching to asynchronous I/O is a well-established architectural best practice in Node.js for improving scalability and responsiveness.
