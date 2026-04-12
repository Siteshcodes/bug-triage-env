# server/task.py
import sys
import random
import hashlib
sys.path.insert(0, "/app")

from typing import Tuple, List, Dict, Any
from model import BugReport, TriageAction


# ---------------------------------------------------------------------------
#  LABEL SYNONYM MAP — allows semantic matching
# ---------------------------------------------------------------------------

LABEL_SYNONYMS: Dict[str, set] = {
    "bug":              {"defect", "issue", "error", "fault", "broken"},
    "security":         {"vulnerability", "cve", "exploit", "auth", "injection"},
    "performance":      {"perf", "slow", "latency", "optimization", "speed", "memory"},
    "ux":               {"ui", "frontend", "user-experience", "design", "usability"},
    "data-integrity":   {"data-loss", "corruption", "data", "consistency"},
    "payments":         {"billing", "payment", "stripe", "checkout", "revenue"},
    "documentation":    {"docs", "typo", "readme", "wiki"},
    "infrastructure":   {"infra", "devops", "deploy", "ci", "cd", "docker"},
    "api":              {"endpoint", "rest", "graphql", "http", "request"},
    "database":         {"db", "sql", "query", "migration", "schema"},
}

# ---------------------------------------------------------------------------
#  BUG TEMPLATE SYSTEM — generates hundreds of unique bugs
# ---------------------------------------------------------------------------

_BUG_TEMPLATES = {
    "crash": {
        "titles": [
            "{service} crashes on {trigger}",
            "{service} throws {error_type} when {trigger}",
            "Fatal error in {service} during {trigger}",
            "Unhandled exception in {service}: {error_type}",
            "{service} segfaults under {condition}",
        ],
        "bodies": [
            "When a user {trigger}, the {service} crashes immediately. "
            "Error: {error_type}. Stack trace points to {component}. "
            "Affects {impact}. {workaround}",
            "The {service} is failing with {error_type} every time a user {trigger}. "
            "No error message is shown to the user — the process just dies. "
            "Impact: {impact}. {workaround}",
        ],
        "vars": {
            "service": ["auth service", "payment gateway", "search API", "notification worker",
                        "session manager", "user profile service", "file upload handler",
                        "webhook processor", "background job runner", "cache layer"],
            "trigger": ["submits a form with special characters", "uploads a file larger than 10MB",
                        "logs in with SSO", "resets their password", "exports data to CSV",
                        "switches between tabs rapidly", "uses the bulk import feature",
                        "accesses the admin panel", "triggers a webhook", "runs a scheduled job"],
            "error_type": ["NullPointerException", "SegmentationFault", "OutOfMemoryError",
                           "ConnectionTimeoutException", "StackOverflowError",
                           "IndexOutOfBoundsException", "TypeError", "KeyError"],
            "component": ["UserController.java:142", "PaymentService.py:89",
                          "AuthMiddleware.ts:56", "SearchIndex.go:203",
                          "NotificationQueue.rb:77", "FileHandler.py:234"],
            "impact": ["100% of users on this flow", "all mobile users", "EU region users only",
                       "users with accounts older than 1 year", "approximately 30% of sessions",
                       "every request during peak hours"],
            "workaround": ["No workaround exists — the feature is completely broken.",
                           "Workaround: users can retry after clearing browser cache.",
                           "Temporary fix: restart the service every 2 hours.",
                           "No known workaround. Users are blocked."],
            "condition": ["high concurrent load", "memory pressure above 80%",
                          "when connection pool is exhausted", "after running for 6+ hours"],
        },
        "answer_template": {
            "severe": {"priority": "P0", "labels": ["bug"], "assigned_team": "backend", "milestone": "hotfix"},
            "moderate": {"priority": "P1", "labels": ["bug"], "assigned_team": "backend", "milestone": "v2.1"},
        },
        "severity_keywords": {
            "severe": ["100%", "all mobile", "No workaround", "completely broken", "blocked",
                       "SegmentationFault", "OutOfMemoryError"],
            "moderate": ["retry", "30%", "Temporary fix", "restart"],
        },
    },

    "security": {
        "titles": [
            "SQL injection vulnerability in {endpoint}",
            "XSS attack possible via {input_field}",
            "Authentication bypass in {service}",
            "Sensitive data exposed in {location}",
            "{credential_type} not invalidated after {event}",
            "SSRF vulnerability in {endpoint}",
        ],
        "bodies": [
            "The {endpoint} endpoint does not sanitize {input_field} inputs. "
            "Crafted queries can {exploit_result}. PoC attached and verified on {env}. "
            "Treat as confidential — do not discuss publicly until patched. {additional_context}",
            "When a user {event}, existing {credential_type} remain valid for {duration}. "
            "An attacker who {attack_vector} can continue to access the account. "
            "This is a {vuln_category} vulnerability. {additional_context}",
        ],
        "vars": {
            "endpoint": ["/api/search", "/api/users", "/api/export", "/admin/query",
                         "/api/upload", "/graphql", "/api/webhook"],
            "input_field": ["search query", "username field", "file upload name",
                            "comment body", "profile bio", "webhook URL"],
            "service": ["login flow", "OAuth callback", "API gateway", "admin panel",
                        "password reset", "2FA verification"],
            "location": ["API error responses", "debug logs shipped to client",
                         "public S3 bucket", "unencrypted cookies", "localStorage"],
            "credential_type": ["JWT tokens", "session cookies", "API keys", "OAuth tokens"],
            "event": ["changes their password", "revokes API access",
                      "is suspended by admin", "enables 2FA"],
            "exploit_result": ["dump the entire user table including password hashes",
                               "execute arbitrary JavaScript in other users' browsers",
                               "access any user's account without credentials",
                               "read internal service endpoints via SSRF"],
            "env": ["production", "staging", "production replica"],
            "duration": ["up to 24 hours", "indefinitely", "until manual cache clear",
                         "for the full token TTL (7 days)"],
            "attack_vector": ["previously stole a token", "intercepted a session cookie",
                              "obtained a leaked API key"],
            "vuln_category": ["session management", "access control",
                              "injection", "broken authentication"],
            "additional_context": [
                "OWASP A03 — Injection.",
                "OWASP A07 — Identification and Authentication Failures.",
                "CVSS score estimated at 9.1 (Critical).",
                "Compliance impact: potential GDPR violation if user PII is exfiltrated.",
                "Bounty hunter reported this 48 hours ago — disclosure deadline approaching.",
            ],
        },
        "answer_template": {
            "default": {"priority": "P0", "labels": ["bug", "security"],
                        "assigned_team": "security", "milestone": "hotfix"},
        },
        "severity_keywords": {"default": []},
    },

    "performance": {
        "titles": [
            "{page} loads slowly for {dataset_size}",
            "Memory leak in {service} causes OOM after {duration}",
            "API response time degrades under {load_condition}",
            "{operation} takes {duration} for {dataset_size}",
            "CPU spikes to 100% when {trigger}",
        ],
        "bodies": [
            "When {condition}, the {page} takes {response_time} to load. "
            "{diagnostic_info}. {impact}. {workaround}",
            "The {service} allocates memory during {operation} and never frees it. "
            "Server runs out of memory every {duration}. {diagnostic_info}. "
            "{workaround}",
        ],
        "vars": {
            "page": ["dashboard", "analytics page", "user list", "search results",
                     "audit log", "reports page", "admin overview"],
            "service": ["background job processor", "cache warming service",
                        "log aggregator", "image resizer", "ETL pipeline"],
            "dataset_size": ["large datasets (10k+ rows)", "enterprise accounts",
                             "tables with 100k+ entries", "files over 50MB"],
            "duration": ["6 hours", "4 hours", "12 hours", "30+ seconds",
                         "2+ minutes", "an entire day"],
            "load_condition": ["concurrent load", "peak traffic", "batch processing",
                               "more than 50 simultaneous users"],
            "operation": ["bulk export", "report generation", "data migration",
                          "full-text search", "image processing"],
            "trigger": ["running bulk exports", "processing large uploads",
                        "generating PDF reports", "reindexing search"],
            "condition": ["a dataset has more than 10k rows",
                          "multiple users trigger exports simultaneously",
                          "the nightly ETL job runs alongside user traffic"],
            "response_time": ["30+ seconds", "over a minute", "2-3 minutes",
                              "timeout after 60 seconds"],
            "diagnostic_info": ["CPU spikes to 100%", "Heap profiler confirms the leak",
                                "Database EXPLAIN shows full table scan",
                                "N+1 query pattern detected in APM",
                                "Garbage collector running every 500ms"],
            "impact": ["Affects power users with large accounts",
                       "All users experience slowness during peak hours",
                       "Requires manual restart to recover",
                       "Operational overhead: scheduled restarts every 4 hours"],
            "workaround": ["Workaround: export data and use offline tools.",
                           "Workaround: scheduled restarts every 4 hours.",
                           "No workaround — users just wait.",
                           "Workaround: paginate results (but UX is degraded)."],
        },
        "answer_template": {
            "severe": {"priority": "P1", "labels": ["bug", "performance"],
                       "assigned_team": "backend", "milestone": "v2.1"},
            "moderate": {"priority": "P2", "labels": ["bug", "performance"],
                         "assigned_team": "backend", "milestone": "v2.1"},
        },
        "severity_keywords": {
            "severe": ["OOM", "100%", "manual restart", "timeout", "No workaround",
                       "all users", "never frees"],
            "moderate": ["Workaround", "power users", "paginate"],
        },
    },

    "ui_bug": {
        "titles": [
            "{ui_element} breaks layout on {browser}",
            "{ui_element} not rendering correctly in {mode}",
            "Responsive layout broken on {device}",
            "{feature} toggle not persisting across {context}",
            "Accessibility: {ui_element} missing {a11y_attr}",
        ],
        "bodies": [
            "Switching to {mode} on {browser} causes {ui_element} to {visual_issue}. "
            "{other_browsers}. {workaround}",
            "On {device}, the {ui_element} is {visual_issue}. "
            "Tested on {browser}. {impact}. {workaround}",
        ],
        "vars": {
            "ui_element": ["navigation bar", "sidebar menu", "modal dialog",
                           "dropdown selector", "data table", "footer",
                           "toast notifications", "breadcrumb trail"],
            "browser": ["Safari 16", "Firefox ESR", "Chrome on Android",
                        "Edge on Windows", "iOS Safari", "Samsung Internet"],
            "mode": ["dark mode", "high contrast mode", "RTL layout",
                     "compact view", "print view"],
            "device": ["iPhone SE", "tablets in portrait", "screens below 768px",
                       "ultra-wide monitors", "4K displays"],
            "feature": ["dark mode", "compact view", "language preference",
                        "notification settings"],
            "context": ["page reloads", "different tabs", "sessions",
                        "browser restarts"],
            "visual_issue": ["overlap the main content", "disappear entirely",
                             "render with incorrect colors", "become unclickable",
                             "overflow beyond the viewport"],
            "other_browsers": ["Chrome and Firefox are unaffected.",
                               "Only reproducible on this specific browser.",
                               "Affects all WebKit-based browsers."],
            "a11y_attr": ["ARIA labels", "keyboard focus indicators",
                          "screen reader text", "proper heading hierarchy"],
            "impact": ["Cosmetic issue, no functional impact.",
                       "Users cannot access the affected feature.",
                       "Usability is degraded but the feature works."],
            "workaround": ["Workaround: use a different browser.",
                           "Workaround: manually resize the window.",
                           "No workaround for this browser.",
                           "Workaround: disable the feature in settings."],
        },
        "answer_template": {
            "severe": {"priority": "P2", "labels": ["bug", "ux"],
                       "assigned_team": "frontend", "milestone": "v2.1"},
            "moderate": {"priority": "P3", "labels": ["bug", "ux"],
                         "assigned_team": "frontend", "milestone": "backlog"},
        },
        "severity_keywords": {
            "severe": ["cannot access", "unclickable", "disappear", "No workaround"],
            "moderate": ["Cosmetic", "different browser", "resize"],
        },
    },

    "data_corruption": {
        "titles": [
            "Race condition in {feature}: {consequence}",
            "Data inconsistency in {feature} under concurrent writes",
            "{export_format} export produces corrupted output for {edge_case}",
            "Stale data served from cache after {trigger}",
            "Duplicate records created when {trigger}",
        ],
        "bodies": [
            "Under concurrent load, {feature} can {consequence} due to a race condition "
            "in {root_cause}. Frequency: {frequency}. {impact}. {workaround}",
            "When {feature} data contains {edge_case}, the exported {export_format} file "
            "is corrupted and cannot be {consumer}. {impact}. {workaround}",
        ],
        "vars": {
            "feature": ["file upload", "order processing", "user registration",
                        "inventory update", "comment system", "permission assignment"],
            "consequence": ["files occasionally overwrite each other",
                            "orders are duplicated or lost",
                            "users get assigned wrong permissions",
                            "inventory counts become negative"],
            "root_cause": ["temp file naming logic", "lack of database locking",
                           "non-atomic read-modify-write cycle",
                           "missing unique constraint"],
            "frequency": ["approximately 1 in 10,000 operations",
                          "consistently under 50+ concurrent users",
                          "intermittently — hard to reproduce",
                          "every time the batch job runs"],
            "edge_case": ["non-ASCII characters (e.g., café, naïve)",
                          "values containing commas or quotes",
                          "null or empty fields",
                          "timestamps crossing DST boundaries"],
            "export_format": ["CSV", "Excel", "JSON", "PDF"],
            "consumer": ["opened in Excel", "parsed by downstream services",
                         "imported back into the system"],
            "trigger": ["double-clicking the submit button",
                        "cache TTL expires during a write operation",
                        "two users edit the same record simultaneously",
                        "the nightly sync job overlaps with user activity"],
            "impact": ["Potential data loss confirmed.",
                       "No data loss confirmed yet, but risk exists.",
                       "Affects users with international data.",
                       "Breaks downstream pipeline processing."],
            "workaround": ["Workaround: enable sequential mode in settings.",
                           "Workaround: manually re-export after cleanup.",
                           "No reliable workaround — data must be manually verified.",
                           "Workaround: add a mutex lock externally (operational overhead)."],
        },
        "answer_template": {
            "severe": {"priority": "P1", "labels": ["bug", "data-integrity"],
                       "assigned_team": "backend", "milestone": "v2.1"},
            "moderate": {"priority": "P2", "labels": ["bug", "data-integrity"],
                         "assigned_team": "backend", "milestone": "v2.1"},
        },
        "severity_keywords": {
            "severe": ["data loss", "No reliable workaround", "consistently",
                       "permissions", "overwrite", "negative"],
            "moderate": ["No data loss", "intermittently", "sequential mode",
                         "re-export", "non-ASCII"],
        },
    },

    "documentation": {
        "titles": [
            "Typo in {location}",
            "Outdated {doc_type} on {page}",
            "Missing documentation for {feature}",
            "Incorrect {doc_element} in {location}",
        ],
        "bodies": [
            "There is a {issue_type} on the {page}: {detail}. No functional impact, "
            "purely cosmetic. {extra}",
            "The {doc_type} for {feature} is {issue_type}. {detail}. {extra}",
        ],
        "vars": {
            "location": ["homepage docs", "API reference", "README", "changelog",
                         "contributing guide", "onboarding wiki"],
            "doc_type": ["installation guide", "API documentation", "changelog",
                         "migration guide", "code comments"],
            "page": ["landing page", "docs homepage", "getting started page",
                     "FAQ section", "footer"],
            "feature": ["new webhook API", "batch processing endpoint",
                        "SSO integration", "rate limiting"],
            "doc_element": ["code example", "endpoint URL", "parameter description",
                            "copyright year", "version number"],
            "issue_type": ["a typo", "outdated", "missing", "incorrect", "misleading"],
            "detail": ["'Welccome' should be 'Welcome'",
                       "references removed v1.x API that no longer exists",
                       "completely undocumented despite being a core feature",
                       "shows '© 2022' but should be '© 2024'",
                       "the curl example uses the wrong HTTP method"],
            "extra": ["", "Low priority — does not block any workflow.",
                      "New users have reported confusion.",
                      "Only noticed by contributors reading source code."],
        },
        "answer_template": {
            "default": {"priority": "P3", "labels": ["documentation"],
                        "assigned_team": "devx", "milestone": "backlog"},
        },
        "severity_keywords": {"default": []},
    },

    "api_bug": {
        "titles": [
            "API rate limiter {issue} after {trigger}",
            "{endpoint} returns {status_code} instead of {expected_code}",
            "Pagination broken on {endpoint}: {symptom}",
            "Webhook delivery {issue} for {event_type} events",
            "API versioning: {endpoint} behaves differently on v1 vs v2",
        ],
        "bodies": [
            "After receiving a {status_code} response, {consequence}. "
            "The {root_cause}. {impact}. {workaround}",
            "The {endpoint} endpoint {symptom} when {trigger}. "
            "Expected behavior: {expected}. Actual: {actual}. {impact}.",
        ],
        "vars": {
            "endpoint": ["/api/users", "/api/search", "/api/export",
                         "/api/webhooks", "/api/billing", "/api/analytics"],
            "issue": ["blocks legitimate users", "fails silently",
                      "returns incorrect retry headers", "drops events"],
            "trigger": ["a 429 error", "rate limit window resets",
                        "a burst of requests from CI/CD", "server restart"],
            "status_code": ["429", "500", "502", "504", "403"],
            "expected_code": ["200", "201", "204", "404"],
            "symptom": ["returns duplicate entries",
                        "skips items between pages",
                        "returns empty page despite more data existing"],
            "event_type": ["payment.completed", "user.created",
                           "subscription.cancelled", "deployment.finished"],
            "consequence": ["legitimate users remain blocked for 1 hour",
                            "data is silently lost with no error",
                            "downstream services receive stale data"],
            "root_cause": ["unblock logic has a bug — it never clears the blocked flag",
                           "cursor-based pagination uses wrong sort order",
                           "retry-after header reports seconds instead of milliseconds"],
            "expected": ["200 OK with paginated results",
                         "successful delivery with retry on failure",
                         "proper rate limit reset after window expires"],
            "actual": ["empty response with 200 status",
                       "permanent block until manual intervention",
                       "events dropped without any error log"],
            "impact": ["Affects CI/CD pipelines hitting the API.",
                       "External integrations break silently.",
                       "Customer-facing dashboards show wrong data.",
                       "Retry-After header causes clients to wait too long."],
            "workaround": ["Workaround: manually clear Redis key.",
                           "Workaround: add client-side deduplication.",
                           "No workaround — requires server-side fix.",
                           "Workaround: pin API version to v1 in headers."],
        },
        "answer_template": {
            "severe": {"priority": "P1", "labels": ["bug", "api"],
                       "assigned_team": "backend", "milestone": "v2.1"},
            "moderate": {"priority": "P2", "labels": ["bug", "api"],
                         "assigned_team": "backend", "milestone": "v2.1"},
        },
        "severity_keywords": {
            "severe": ["silently lost", "permanent block", "No workaround",
                       "dropped", "external integrations"],
            "moderate": ["Workaround", "pin API", "deduplication"],
        },
    },
}


# The original handcrafted bugs — kept as a gold-standard subset
_HANDCRAFTED_BUGS = {
    "easy": {
        "bugs": [
            BugReport(
                id="easy-001",
                title="App crashes on login with correct credentials",
                body="When I enter my correct username and password, the app crashes immediately. "
                     "This started after the v2.0 release. Affects 100% of users. "
                     "No workaround exists — users cannot log in at all.",
                author="user123",
                labels_hint=[],
                comments=["Confirmed on iOS and Android.", "Happens every time."],
                severity_signals=["100% of users", "crashes", "no workaround"],
                stack_trace="NullPointerException at AuthController.java:87",
                affected_component="auth-service",
            ),
            BugReport(
                id="easy-002",
                title="Typo in documentation homepage",
                body="There is a typo on the homepage docs: 'Welccome' should be 'Welcome'. "
                     "No functional impact, purely cosmetic.",
                author="docs_fan",
                labels_hint=["documentation"],
                comments=[],
                severity_signals=["cosmetic", "no functional impact"],
                stack_trace="",
                affected_component="docs",
            ),
            BugReport(
                id="easy-003",
                title="Dashboard loads slowly for large datasets",
                body="When a dataset has more than 10k rows, the dashboard takes 30+ seconds to load. "
                     "Workaround: export data and use offline tools. Affects power users only.",
                author="power_user",
                labels_hint=["performance"],
                comments=["Noticed after the last deploy.", "CPU spikes to 100%."],
                severity_signals=["workaround exists", "power users only"],
                stack_trace="",
                affected_component="dashboard",
            ),
            BugReport(
                id="easy-004",
                title="Email notifications not sent after password reset",
                body="Users who reset their password do not receive the confirmation email. "
                     "SMTP logs show the job is queued but never dispatched. "
                     "Affects all users attempting password reset.",
                author="support_team",
                labels_hint=["bug"],
                comments=["Reported by 12 users this week.",
                           "Started after email service migration."],
                severity_signals=["all users", "never dispatched"],
                stack_trace="",
                affected_component="email-service",
            ),
            BugReport(
                id="easy-005",
                title="Incorrect copyright year in footer",
                body="The footer shows '© 2022' but it should be '© 2024'. "
                     "No functional impact.",
                author="intern_dev",
                labels_hint=["documentation"],
                comments=[],
                severity_signals=["no functional impact"],
                stack_trace="",
                affected_component="frontend",
            ),
        ],
        "answers": {
            "easy-001": {"priority": "P0"},
            "easy-002": {"priority": "P3"},
            "easy-003": {"priority": "P2"},
            "easy-004": {"priority": "P1"},
            "easy-005": {"priority": "P3"},
        },
    },

    "medium": {
        "bugs": [
            BugReport(
                id="med-001",
                title="Payment fails silently on checkout",
                body="Checkout completes without error but payment is never charged. "
                     "No error shown to user. Stripe logs show declined transaction. "
                     "Direct revenue loss — every failed checkout is a lost sale.",
                author="store_owner",
                labels_hint=["bug"],
                comments=["Revenue impact confirmed.", "Happening since Tuesday."],
                severity_signals=["revenue loss", "silently", "every failed checkout"],
                stack_trace="Stripe API: card_declined at PaymentService.py:145",
                affected_component="payment-service",
            ),
            BugReport(
                id="med-002",
                title="Search results include deleted posts",
                body="Deleted blog posts still appear in search results for up to 24 hours. "
                     "Users can read content that was explicitly removed by moderators. "
                     "Potential GDPR violation if deleted content belongs to EU users.",
                author="moderator_jane",
                labels_hint=[],
                comments=["GDPR concern — deleted content still visible."],
                severity_signals=["GDPR violation", "deleted content visible"],
                stack_trace="",
                affected_component="search-index",
            ),
            BugReport(
                id="med-003",
                title="Dark mode toggle breaks layout on Safari",
                body="Switching to dark mode on Safari 16 causes nav bar to overlap content. "
                     "Chrome and Firefox unaffected. Workaround: use a different browser.",
                author="safari_user",
                labels_hint=["bug", "ux"],
                comments=["Only on Safari, not Chrome/Firefox."],
                severity_signals=["workaround exists", "single browser"],
                stack_trace="",
                affected_component="frontend-css",
            ),
            BugReport(
                id="med-004",
                title="CSV export produces corrupted file for non-ASCII characters",
                body="When table data contains accented characters (e.g. café, naïve), "
                     "the exported CSV file is corrupted and cannot be opened in Excel. "
                     "Affects users with international data.",
                author="data_analyst",
                labels_hint=["bug"],
                comments=["Encoding issue — UTF-8 not respected.",
                           "Workaround: manual copy-paste."],
                severity_signals=["corrupted", "workaround exists"],
                stack_trace="",
                affected_component="export-service",
            ),
            BugReport(
                id="med-005",
                title="API rate limiter blocks legitimate users after 429 error",
                body="After receiving a 429 Too Many Requests response, legitimate users "
                     "remain blocked for 1 hour even after the rate limit window resets. "
                     "The unblock logic has a bug — it never clears the blocked flag.",
                author="api_user",
                labels_hint=["bug"],
                comments=["Affects CI/CD pipelines hitting the API.",
                           "Retry-After header is wrong."],
                severity_signals=["permanent block", "never clears", "bug in logic"],
                stack_trace="",
                affected_component="api-gateway",
            ),
        ],
        "answers": {
            "med-001": {"priority": "P0", "labels": ["bug", "payments"],        "assigned_team": "backend"},
            "med-002": {"priority": "P1", "labels": ["bug", "security"],        "assigned_team": "security"},
            "med-003": {"priority": "P2", "labels": ["bug", "ux"],              "assigned_team": "frontend"},
            "med-004": {"priority": "P2", "labels": ["bug", "data-integrity"],  "assigned_team": "backend"},
            "med-005": {"priority": "P1", "labels": ["bug", "performance"],     "assigned_team": "backend"},
        },
    },

    "hard": {
        "bugs": [
            BugReport(
                id="hard-001",
                title="SQL injection vulnerability in search endpoint",
                body="The /api/search endpoint does not sanitize inputs. "
                     "Crafted queries can dump the entire user table including password hashes. "
                     "PoC attached. Verified on production. Treat as confidential — "
                     "do not discuss publicly until patched.",
                author="security_researcher",
                labels_hint=[],
                comments=["Critical. Affects production.", "Do not discuss publicly."],
                severity_signals=["SQL injection", "password hashes", "production",
                                  "confidential"],
                stack_trace="",
                affected_component="search-api",
            ),
            BugReport(
                id="hard-002",
                title="Memory leak in background job processor causes OOM after 6 hours",
                body="The job processor allocates ~50MB per job and never frees it. "
                     "Server runs out of memory every 6 hours, requiring a manual restart. "
                     "Heap profiler confirms leak introduced in v1.9. "
                     "Workaround: scheduled restarts every 4 hours (operational overhead).",
                author="devops_alice",
                labels_hint=["performance"],
                comments=["Verified with heap profiler.", "Started in v1.9."],
                severity_signals=["memory leak", "OOM", "manual restart", "never frees"],
                stack_trace="HeapDump: JobProcessor.process() -> 50MB/call, never GC'd",
                affected_component="job-processor",
            ),
            BugReport(
                id="hard-003",
                title="Race condition in file upload: files occasionally overwrite each other",
                body="Under concurrent load, two users uploading simultaneously can get "
                     "each other's files due to a race condition in the temp file naming logic. "
                     "Frequency: approximately 1 in 10,000 uploads under normal load. "
                     "No data loss confirmed yet and a workaround exists: "
                     "enable sequential upload mode in settings (disabled by default). "
                     "Risk is low-probability but affects data integrity.",
                author="qa_bot",
                labels_hint=["bug"],
                comments=["Reproduced with locust at 50 concurrent users.",
                           "Sequential mode avoids it."],
                severity_signals=["race condition", "data integrity",
                                  "workaround exists", "low-probability"],
                stack_trace="",
                affected_component="file-upload",
            ),
            BugReport(
                id="hard-004",
                title="Auth token not invalidated after password change",
                body="When a user changes their password, existing JWT tokens remain valid "
                     "for up to 24 hours. An attacker who previously stole a token can "
                     "continue to access the account even after the password is reset. "
                     "This is a session management security vulnerability.",
                author="pentest_team",
                labels_hint=["security"],
                comments=["Verified on staging.",
                           "OWASP A07 — Identification and Authentication Failures."],
                severity_signals=["JWT not invalidated", "attacker", "security vulnerability",
                                  "stolen token"],
                stack_trace="",
                affected_component="auth-service",
            ),
            BugReport(
                id="hard-005",
                title="Infinite loop in webhook retry logic causes CPU spike",
                body="When a webhook endpoint returns a 500 error, the retry logic enters "
                     "an infinite loop with no backoff or retry cap. "
                     "This causes CPU to spike to 100% within minutes and starves other services. "
                     "Triggered in production twice this week. Requires process kill to recover.",
                author="oncall_eng",
                labels_hint=["bug", "performance"],
                comments=["PagerDuty alert fired twice.",
                           "Needs exponential backoff + max retry cap."],
                severity_signals=["infinite loop", "100%", "production",
                                  "process kill", "starves other services"],
                stack_trace="Thread dump: WebhookRetrier.retry() → recursive call, no exit",
                affected_component="webhook-service",
            ),
        ],
        "answers": {
            "hard-001": {
                "priority": "P0", "labels": ["bug", "security"],
                "assigned_team": "security", "milestone": "hotfix",
            },
            "hard-002": {
                "priority": "P1", "labels": ["bug", "performance"],
                "assigned_team": "backend", "milestone": "v2.1",
            },
            "hard-003": {
                "priority": "P1", "labels": ["bug", "data-integrity"],
                "assigned_team": "backend", "milestone": "v2.1",
            },
            "hard-004": {
                "priority": "P0", "labels": ["bug", "security"],
                "assigned_team": "security", "milestone": "hotfix",
            },
            "hard-005": {
                "priority": "P0", "labels": ["bug", "performance"],
                "assigned_team": "backend", "milestone": "hotfix",
            },
        },
    },
}


# Combine into single TASKS dict (backward compatible)
TASKS = _HANDCRAFTED_BUGS


# ---------------------------------------------------------------------------
#  PROCEDURAL BUG GENERATOR
# ---------------------------------------------------------------------------

def _determine_severity(text: str, keywords: Dict[str, list]) -> str:
    """Check which severity level the generated text matches."""
    text_lower = text.lower()
    for level, kws in keywords.items():
        if level == "default":
            return "default"
        hits = sum(1 for kw in kws if kw.lower() in text_lower)
        if hits >= 1:
            return level
    # fallback to first non-default key
    return list(keywords.keys())[0] if keywords else "moderate"


def generate_bug(task_key: str, seed: int = None) -> Tuple[BugReport, dict]:
    """Generate a procedural bug report with its correct answer."""
    rng = random.Random(seed)

    # Weight categories by difficulty
    weights = {
        "easy": {"documentation": 3, "ui_bug": 3, "performance": 2,
                 "crash": 1, "api_bug": 1},
        "medium": {"crash": 3, "performance": 3, "api_bug": 2,
                   "data_corruption": 2, "ui_bug": 1},
        "hard": {"security": 4, "crash": 3, "data_corruption": 3,
                 "performance": 2, "api_bug": 2},
    }

    task_weights = weights.get(task_key, weights["medium"])
    categories = []
    for cat, w in task_weights.items():
        categories.extend([cat] * w)
    category = rng.choice(categories)

    template = _BUG_TEMPLATES[category]

    # Pick random variable values
    chosen_vars = {}
    for var_name, options in template["vars"].items():
        chosen_vars[var_name] = rng.choice(options)

    # Build title and body
    title_tmpl = rng.choice(template["titles"])
    body_tmpl = rng.choice(template["bodies"])

    # Safe format — ignore missing keys
    def safe_format(tmpl, vars_dict):
        result = tmpl
        for k, v in vars_dict.items():
            result = result.replace("{" + k + "}", v)
        return result

    title = safe_format(title_tmpl, chosen_vars)
    body = safe_format(body_tmpl, chosen_vars)

    # Generate unique ID from seed
    bug_id = f"gen-{seed or rng.randint(0, 999999):06d}"

    # Pick author
    authors = ["user_report", "qa_engineer", "support_team", "dev_oncall",
               "security_bot", "customer_jane", "automated_monitor",
               "intern_dev", "senior_eng", "pm_feedback"]
    author = rng.choice(authors)

    # Build comments
    comment_templates = [
        "Confirmed on our side.", "Reproduced in staging.",
        "Multiple reports from users.", "Started after last deployment.",
        "Urgent — customer escalation.", "Low priority — no user complaints.",
        "Needs investigation.", "Related to ticket from last sprint.",
    ]
    num_comments = rng.randint(0, 3)
    comments = rng.sample(comment_templates, min(num_comments, len(comment_templates)))

    # Determine severity and answer
    full_text = f"{title} {body} {' '.join(comments)}"
    severity_kws = template.get("severity_keywords", {})
    severity = _determine_severity(full_text, severity_kws)

    answer_templates = template["answer_template"]
    answer = dict(answer_templates.get(severity, list(answer_templates.values())[0]))

    # For easy tasks, only priority matters
    if task_key == "easy":
        answer = {"priority": answer["priority"]}
    elif task_key == "medium":
        answer.pop("milestone", None)

    bug = BugReport(
        id=bug_id,
        title=title,
        body=body,
        author=author,
        labels_hint=rng.sample(["bug", "needs-triage", "reported"], rng.randint(0, 2)),
        comments=comments,
        severity_signals=[],
        stack_trace="",
        affected_component=chosen_vars.get("service", chosen_vars.get("endpoint", "")),
    )

    return bug, answer


# ---------------------------------------------------------------------------
#  BUG SAMPLER — uses handcrafted bugs first, then procedural for variety
# ---------------------------------------------------------------------------

def sample_bug(task_key: str, seed: int = None) -> Tuple[BugReport, dict]:
    """Return a bug and its answer. Mixes handcrafted + procedural."""
    rng = random.Random(seed)

    # 40% chance of handcrafted, 60% procedural
    if rng.random() < 0.4 and task_key in _HANDCRAFTED_BUGS:
        bugs = _HANDCRAFTED_BUGS[task_key]["bugs"]
        bug = rng.choice(bugs)
        answer = _HANDCRAFTED_BUGS[task_key]["answers"][bug.id]
        return bug, answer
    else:
        gen_seed = seed if seed is not None else rng.randint(0, 999999)
        return generate_bug(task_key, seed=gen_seed)


# ---------------------------------------------------------------------------
#  GRADING — with semantic label matching
# ---------------------------------------------------------------------------

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _priority_score(predicted: str, correct: str) -> float:
    """Score priority assignment with partial credit for near-misses."""
    if predicted == correct:
        return 0.95
    pred_rank = PRIORITY_ORDER.get(predicted, 99)
    corr_rank = PRIORITY_ORDER.get(correct, 99)
    diff = abs(pred_rank - corr_rank)
    if diff == 1:
        return 0.5
    elif diff == 2:
        return 0.2
    return 0.05


def _normalize_label(label: str) -> str:
    """Normalize a label to its canonical form."""
    label_lower = label.lower().strip()
    for canonical, synonyms in LABEL_SYNONYMS.items():
        if label_lower == canonical or label_lower in synonyms:
            return canonical
    return label_lower


def _label_score(predicted: List[str], correct: List[str]) -> float:
    """Score labels using semantic matching via synonym groups."""
    pred_normalized = set(_normalize_label(l) for l in predicted)
    corr_normalized = set(_normalize_label(l) for l in correct)

    if not corr_normalized:
        return 0.95

    intersection = pred_normalized & corr_normalized
    union = pred_normalized | corr_normalized

    raw = len(intersection) / len(union) if union else 0.0
    return max(0.05, min(0.95, raw))


def _reasoning_score(reasoning: str, answer: dict) -> float:
    """Bonus for reasoning that mentions relevant signals."""
    if not reasoning or len(reasoning.strip()) < 10:
        return 0.0

    key_signals = {
        "P0": ["production", "all users", "data loss", "security", "crash",
               "revenue", "injection", "vulnerability", "100%"],
        "P1": ["major", "significant", "no workaround", "broken",
               "gdpr", "blocked", "leak", "never"],
        "P2": ["degraded", "workaround", "partial", "slow",
               "affected", "power users"],
        "P3": ["minor", "cosmetic", "docs", "typo", "low",
               "no functional impact"],
    }

    expected_priority = answer.get("priority", "P2")
    signals = key_signals.get(expected_priority, [])
    reasoning_lower = reasoning.lower()

    hits = sum(1 for s in signals if s in reasoning_lower)
    return min(0.15, hits * 0.05)


def grade_action(task_key: str, bug: BugReport, action: TriageAction,
                 answer: dict = None) -> Tuple[float, str]:
    """Grade the agent's triage action against the correct answer."""

    # Backward compatibility: look up answer from handcrafted if not provided
    if answer is None:
        if task_key in _HANDCRAFTED_BUGS and bug.id in _HANDCRAFTED_BUGS[task_key]["answers"]:
            answer = _HANDCRAFTED_BUGS[task_key]["answers"][bug.id]
        else:
            return 0.5, "No answer key found for this bug."

    feedback_parts = []
    reasoning_bonus = _reasoning_score(action.reasoning, answer)

    if task_key == "easy":
        score = _priority_score(action.priority, answer["priority"])
        symbol = "✓" if score >= 0.9 else "~" if score >= 0.4 else "✗"
        feedback_parts.append(
            f"Priority: {symbol} (got {action.priority}, expected {answer['priority']})")
        score = score + reasoning_bonus
        score = max(0.01, min(0.99, score))
        return round(score, 3), " | ".join(feedback_parts)

    elif task_key == "medium":
        p_score = _priority_score(action.priority, answer["priority"])
        l_score = _label_score(action.labels, answer.get("labels", []))
        expected_team = answer.get("assigned_team", "")
        t_score = 0.95 if expected_team and action.assigned_team.lower() == expected_team.lower() else 0.05

        score = 0.45 * p_score + 0.40 * l_score + 0.15 * t_score + reasoning_bonus

        feedback_parts.append(
            f"Priority: {p_score:.2f} (got {action.priority}, expected {answer['priority']})")
        feedback_parts.append(f"Labels: {l_score:.2f} (semantic match)")
        feedback_parts.append(
            f"Team: {t_score:.2f} (got {action.assigned_team}, expected {expected_team})")
        if reasoning_bonus > 0:
            feedback_parts.append(f"Reasoning bonus: +{reasoning_bonus:.2f}")

        score = max(0.01, min(0.99, score))
        return round(score, 3), " | ".join(feedback_parts)

    else:  # hard
        p_score = _priority_score(action.priority, answer["priority"])
        l_score = _label_score(action.labels, answer.get("labels", []))
        t_score = 0.95 if action.assigned_team.lower() == answer["assigned_team"].lower() else 0.05
        m_score = 0.95 if action.milestone.lower() == answer["milestone"].lower() else 0.05

        score = 0.35 * p_score + 0.30 * l_score + 0.20 * t_score + 0.15 * m_score + reasoning_bonus

        feedback_parts.append(
            f"Priority: {p_score:.2f} (got {action.priority}, expected {answer['priority']})")
        feedback_parts.append(f"Labels: {l_score:.2f} (semantic match)")
        feedback_parts.append(
            f"Team: {t_score:.2f} (got {action.assigned_team}, expected {answer['assigned_team']})")
        feedback_parts.append(
            f"Milestone: {m_score:.2f} (got {action.milestone}, expected {answer['milestone']})")

        if reasoning_bonus > 0:
            feedback_parts.append(f"Reasoning bonus: +{reasoning_bonus:.2f}")

        # Security escalation penalty
        if answer.get("assigned_team") == "security" and action.assigned_team.lower() != "security":
            score = max(0.01, score - 0.15)
            feedback_parts.append("⚠ Security escalation missed (-0.15)")

        score = max(0.01, min(0.99, score))
        return round(score, 3), " | ".join(feedback_parts)


# ---------------------------------------------------------------------------
#  NAMED GRADER FUNCTIONS — referenced by openenv.yaml
# ---------------------------------------------------------------------------

def priority_match(*args, **kwargs):
    if len(args) < 2:
        return 0.5
    bug, action = args[0], args[1]
    score, _ = grade_action("easy", bug, action)
    return float(score)


def priority_label_team(*args, **kwargs):
    if len(args) < 2:
        return 0.5
    bug, action = args[0], args[1]
    score, _ = grade_action("medium", bug, action)
    return float(score)


def full_triage(*args, **kwargs):
    if len(args) < 2:
        return 0.5
    bug, action = args[0], args[1]
    score, _ = grade_action("hard", bug, action)
    return float(score)


__all__ = [
    "priority_match",
    "priority_label_team",
    "full_triage",
    "sample_bug",
    "generate_bug",
    "grade_action",
    "TASKS",
    "LABEL_SYNONYMS",
]