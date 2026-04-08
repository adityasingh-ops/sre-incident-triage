"""
Quick local test to verify no score is exactly 0.0 or 1.0.
Run from project root: python test_scores_valid.py
"""
import sys
sys.path.insert(0, ".")

from server.graders.grader import Grader, clamp_score

grader = Grader()

# Simulate a perfect diagnosis (would have returned 1.0 before the fix)
perfect_diagnosis = {
    "root_cause": "cpu_exhaustion",
    "affected_service": "fraud-service",
}
perfect_incident = {
    "root_cause": "cpu_exhaustion",
    "affected_service": "fraud-service",
    "correct_playbook": "scale_horizontal",
    "correct_rollback": False,
    "cascade_service": None,
}

print("=== Testing score validity (all must be strictly between 0 and 1) ===\n")

errors = []

for task_id in ["task1", "task2", "task3", "task4"]:
    for steps in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        max_steps = {"task1": 6, "task2": 8, "task3": 10, "task4": 12}[task_id]
        if steps > max_steps:
            continue

        actions = ["list_alerts", "query_logs", "get_metrics", "submit_diagnosis"][:steps]

        reward = grader.grade(
            task_id=task_id,
            incident=perfect_incident,
            actions_taken=actions,
            diagnosis=perfect_diagnosis,
            steps_taken=steps,
            max_steps=max_steps,
        )

        for field in ["total", "correctness", "efficiency", "speed", "partial_credit"]:
            val = getattr(reward, field)
            if val <= 0.0 or val >= 1.0:
                errors.append(f"FAIL task={task_id} steps={steps} {field}={val}")
            else:
                print(f"  OK  task={task_id} steps={steps} {field}={val:.6f}")

print()
if errors:
    print("FAILURES:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
else:
    print("All scores valid! Safe to submit.")
