# Teacher Diagnostic Tool: Plan for Actionable Recommendations and Early Detection

## The Shift

This project is **not an AI tutor.** It is a diagnostic instrument that gives teachers
real-time visibility into the specific misconceptions their students hold, so they can
intervene earlier and more precisely than any dashboard showing percentages.

The teacher educates the student. The system tells the teacher what to look for.

---

## What We Already Have (Backend)

| Component | Status | What It Does |
|-----------|--------|-------------|
| Misconception classifier | Working (91% accuracy) | Takes free-form student text, returns specific misconception ID + confidence |
| Knowledge graph | Working (5 concepts, 19 misconceptions) | Prerequisite chain, BKT params, misconception-to-concept mapping |
| BKT mastery model | Working | Per-student, per-concept mastery probability updated after each response |
| Problem bank | Working (28 problems) | Problems tagged by concept and difficulty |
| Adaptive engine | Working | next_action() decides remediate/progress/review based on student state |

All of this currently powers a CLI demo. None of it is accessible to a teacher.

---

## Behavior B: Actionable Teacher Recommendations

### What This Means

The system does not just show data. It tells the teacher **what to do next** with
specific students or groups, based on the diagnostic signal.

### B1: Misconception-Based Student Grouping

**Input:** A class of N students, each with BKT state and misconception history.

**Output:** "These students share the same gap. Address them together."

How it works:
1. After a problem set, classify each student's responses
2. Cluster students by their active misconception(s)
3. Surface groups to the teacher: "5 students all have `dist_first_term_only` -
   they distribute to only the first term"
4. Rank groups by size (biggest group = highest-leverage intervention)

**API shape:**
```
POST /api/classroom/{class_id}/groups
Response: [
  {
    "misconception": "dist_first_term_only",
    "label": "Only distributes to first term",
    "students": ["Maria", "James", "Aiden", "Sofia", "Liam"],
    "count": 5,
    "evidence": [
      {"student": "Maria", "problem": "Expand: 2(x + 3)", "wrote": "2x + 3", "correct": "2x + 6"},
      ...
    ],
    "suggested_action": "Small-group mini-lesson on distribution. Use area model visual: draw a rectangle with width 2 and length (x + 3), show both parts get multiplied."
  },
  ...
]
```

### B2: Individual Student Intervention Cards

For any student, generate a one-screen summary a teacher can glance at in 30 seconds:

- **Mastery map:** Which concepts are solid, which are shaky, which are blocked
- **Active misconceptions:** The specific errors this student keeps making, with examples from their own work
- **Prerequisite bottleneck:** "This student can't progress to solving equations because their order-of-operations foundation is weak"
- **Recommended next step:** One concrete action - "Give Maria 3 problems focused on distributing negatives. If she gets 2/3 right, she's ready to move on."

**API shape:**
```
GET /api/student/{student_id}/card
Response: {
  "student": "Maria",
  "mastery": {
    "integer_sign_ops": {"level": 0.92, "status": "mastered"},
    "order_of_operations": {"level": 0.88, "status": "mastered"},
    "distributive_property": {"level": 0.31, "status": "struggling", "blocking": ["combining_like_terms", "solving_linear_equations"]},
    ...
  },
  "active_misconceptions": [
    {
      "id": "dist_first_term_only",
      "label": "Only distributes to first term",
      "times_observed": 4,
      "last_seen": "2026-03-20",
      "evidence": [{"problem": "...", "wrote": "...", "correct": "..."}]
    }
  ],
  "bottleneck": "distributive_property is blocking 2 downstream concepts",
  "recommended_action": "Assign 3 distribution problems with visual scaffolding. Focus on negative factor distribution.",
  "recommended_problems": ["dist_3", "dist_4", "dist_5"]
}
```

### B3: Class Readiness Check

Before moving to a new topic, teacher asks: "Is my class ready?"

**API shape:**
```
GET /api/classroom/{class_id}/readiness?target_concept=combining_like_terms
Response: {
  "target": "combining_like_terms",
  "prerequisite": "distributive_property",
  "class_size": 28,
  "ready": 19,
  "not_ready": 9,
  "at_risk_students": [
    {"name": "Maria", "blocker": "dist_first_term_only", "mastery": 0.31},
    {"name": "James", "blocker": "dist_sign_error_negative", "mastery": 0.42},
    ...
  ],
  "recommendation": "9 students still have distributive property gaps. Consider a 15-minute review session before introducing combining like terms. Focus on: distributing to all terms (5 students), sign errors with negatives (3 students), dropping parentheses (1 student)."
}
```

---

## Behavior C: Earlier Detection

### What This Means

Instead of waiting for a student to fail a unit test, the system detects prerequisite
gaps that **predict** upcoming failure and alerts the teacher before it happens.

### C1: Prerequisite Gap Alert

The knowledge graph encodes: you can't do combining_like_terms if you don't have
distributive_property. You can't do distributive_property if you don't have
order_of_operations.

When a student's mastery on a prerequisite concept drops below threshold while the
class is about to move forward:

```
ALERT: Maria's order_of_operations mastery is 0.42.
The class is starting distributive_property next week.
She will likely struggle because distribution requires fluent operation ordering.
Intervene now - 3 targeted problems on operation precedence could close this gap.
```

**How it works:**
1. Track where the class curriculum is heading (teacher sets the schedule or we infer from problem assignments)
2. For each student, check if prerequisite mastery for the upcoming topic meets threshold
3. Surface students who are below threshold on prerequisites for whatever's next
4. The earlier this fires, the more time the teacher has to intervene

### C2: Misconception Persistence Tracking

Some misconceptions resolve with one correction. Others persist for weeks. Track
how each misconception evolves over time per student:

```
WARNING: Maria has exhibited dist_first_term_only on 4 separate occasions
across 2 weeks. Standard correction is not working.
Consider: hands-on manipulative activity, peer tutoring, or parent conference.
```

**Signal:** If a misconception appears 3+ times after the teacher has been alerted,
escalate the severity.

### C3: Cascade Risk Detection

Use the prerequisite graph to identify students at risk of cascading failure:

```
AT RISK: James has mastery 0.38 on integer_sign_ops (Level 1).
This is the foundation for ALL downstream concepts.
He cannot meaningfully progress until this is addressed.
Every topic he attempts will be built on a weak foundation.
Priority: highest.
```

**How it works:**
1. For each student with a below-threshold concept, count how many downstream
   concepts depend on it (directly or transitively)
2. The more downstream concepts blocked, the higher the cascade risk
3. Level 1 gaps are always highest priority since everything depends on them

---

## API Architecture

### Tech Stack

- **Backend:** Python (FastAPI) - reuses existing src/ modules directly
- **Database:** SQLite for prototype (student records, response history, class rosters)
- **Auth:** Simple teacher login (email/password) for prototype
- **Inference:** Classifier runs on-device (no API calls needed - DistilBERT is 250MB)
- **Frontend:** Separate concern - see UI research prompt

### Core API Endpoints

```
# Teacher & Classroom Management
POST   /api/auth/login
GET    /api/classrooms
POST   /api/classrooms
GET    /api/classrooms/{id}/students

# Student Work Submission (teacher assigns, student submits)
POST   /api/assignments                          # Teacher creates assignment
POST   /api/assignments/{id}/responses           # Student submits answer
GET    /api/assignments/{id}/results             # Teacher views results

# Diagnostic Views (Behavior B)
GET    /api/classrooms/{id}/overview             # Class misconception heat map
GET    /api/classrooms/{id}/groups               # B1: Misconception-based groups
GET    /api/classrooms/{id}/readiness            # B3: Readiness check
GET    /api/students/{id}/card                   # B2: Individual student card

# Early Detection (Behavior C)
GET    /api/classrooms/{id}/alerts               # C1+C2+C3: All active alerts
GET    /api/students/{id}/trajectory             # Mastery over time
GET    /api/classrooms/{id}/cascade-risks        # C3: Students at risk of cascading failure

# Problem Bank
GET    /api/problems?concept={id}&difficulty={level}
POST   /api/problems/recommend/{student_id}      # System picks best next problems
```

### Data Model

```
Teacher
  id, name, email, password_hash

Classroom
  id, teacher_id, name, grade_level

Student
  id, classroom_id, name

Assignment
  id, classroom_id, problems[], assigned_at, due_at

Response
  id, assignment_id, student_id, problem_id
  student_text, timestamp
  classified_misconception, confidence
  correct (bool)

StudentMastery  (BKT state, updated after each response)
  student_id, concept_id, mastery_level, attempts, last_updated

Alert
  id, student_id, classroom_id, type, severity
  message, created_at, acknowledged_at
```

---

## Build Sequence

### Phase 1: API Core (Foundation)

- FastAPI app with SQLite
- Teacher auth (simple JWT)
- Classroom/student CRUD
- Wire in existing classifier and KnowledgeGraph modules
- POST endpoint: submit student response, classify, update BKT state

### Phase 2: Behavior B Endpoints

- B1: Grouping endpoint (cluster students by misconception)
- B2: Student card endpoint
- B3: Readiness check endpoint
- Recommendation generation (rule-based first, LLM-enhanced later)

### Phase 3: Behavior C Endpoints

- C1: Prerequisite gap alerts
- C2: Misconception persistence tracking
- C3: Cascade risk detection
- Alert creation and delivery

### Phase 4: Frontend (Separate Effort)

- Teacher dashboard (see UI research prompt)
- Student submission interface (minimal - just answer input)
- Mobile-friendly (teachers walk around classrooms)

---

## What Makes This Different From Khan's Dashboard

| Khan Academy | This System |
|-------------|-------------|
| "Maria scored 60% on distributive property" | "Maria specifically distributes to only the first term - she's done it 4 times" |
| Shows progress per skill | Shows the WHY behind the gap |
| Teacher must interpret data | System recommends specific action |
| No grouping by shared misconception | "These 5 students share the same error - address them together" |
| No prerequisite cascade warnings | "James can't progress because Level 1 is weak - fix this first" |
| Multiple choice inputs | Free-form text - sees HOW students think, not just WHAT they picked |
| Student-facing platform that teachers monitor | Teacher-facing tool that students contribute to |

## What This Does NOT Do

- Tutor students directly
- Replace the teacher's judgment
- Assign grades
- Work without the teacher actively using it
- Claim to improve test scores (until we run a real pilot)

The teacher is the intervention. The system is the diagnostic.
