# UI Design Specification: Teacher Misconception Diagnostic Dashboard

## Overview

Design specification for a teacher-facing web dashboard that surfaces algebra misconception
diagnostics. The primary user is a middle/high school algebra teacher (grades 6-10) with
20-35 students per class. The system detects 19 specific misconception types across 5
algebra concepts from free-form student text, tracks mastery via Bayesian Knowledge Tracing,
and generates actionable intervention recommendations.

This document defines every screen, component, interaction pattern, and visual direction a
frontend developer needs to implement the dashboard.

---

## Design Principles

1. Teacher time is sacred. Every screen answers a question in under 30 seconds.
2. Show the WHY, not just the WHAT. A score of 60% is useless. "Distributes to first term only" is actionable.
3. Surface problems, don't make teachers hunt. Alerts push to the teacher; the teacher never needs to dig.
4. Progressive disclosure. Class pulse first, then groups, then individual evidence. Two clicks maximum from overview to any student's work.
5. Professional, not playful. This is a clinical diagnostic tool, not a gamified student app.

---

## Site Map

```text
/login
/classrooms                          ← Landing page (class selector)
/classrooms/:id                      ← Redirects to /classrooms/:id/overview
/classrooms/:id/overview             ← Class overview (heat map + summary cards)
/classrooms/:id/groups               ← Misconception-based student groupings
/classrooms/:id/readiness            ← Readiness check before next topic
/classrooms/:id/alerts               ← Active alerts (prerequisite gaps, persistence, cascade)
/classrooms/:id/live                 ← Live classroom feed (during-class mobile view)
/classrooms/:id/assignments          ← Assignment management
/classrooms/:id/assignments/:aid     ← Single assignment results
/students/:id                        ← Individual student detail
/students/:id/history                ← Full response history
/settings                            ← Account and preferences
/settings/classrooms                 ← Manage class rosters
/settings/schedule                   ← Curriculum schedule (which concept is next)

Student-facing (separate entry point, no navigation overlap):
/join                                ← Student enters class code
/work/:assignment_id                 ← Student answers problems (one at a time)
/work/:assignment_id/done            ← Student completion screen
```

### Navigation Flow

```text
Login → Classrooms (list)
         └→ Class Overview (default tab)
              ├→ Groups tab
              ├→ Readiness tab
              ├→ Alerts tab
              ├→ Live tab (during class)
              ├→ Assignments tab
              │    └→ Assignment results
              └→ Click any student → Student Detail
                   └→ Full History
```

---

## Screen-by-Screen Wireframe Descriptions

### 1. Login (`/login`)

Single centered card on a light gray background.

* Email input field
* Password input field
* "Sign In" button (primary action, indigo)
* "Forgot password?" link below
* No student login here. Students join via class code at `/join`.

No sidebar. No navigation. Just the login card.

### 2. Class Selector (`/classrooms`)

The landing page after login. Teachers often teach 3-5 sections.

**Layout:** Centered grid of class cards (2-3 columns on desktop, single column on mobile).

Each class card contains:

* Class name (e.g., "Period 2 - Algebra I")
* Student count (e.g., "28 students")
* Alert badge: red circle with count if active alerts exist (e.g., "3")
* Class pulse indicator: a small horizontal bar colored proportionally (green segment = mastered students, amber = progressing, red = struggling). This gives a glanceable sense of the class state before clicking in.

**Interactions:**

* Click a card to enter that class's overview
* "Add Class" button in the top right opens a creation form

### 3. Class Overview (`/classrooms/:id/overview`)

The most important screen. A teacher spends 80% of their time here.

**Layout:** Left sidebar navigation + main content area.

**Sidebar (persistent across all class sub-pages):**

* Class name at top with a dropdown to switch classes without returning to `/classrooms`
* Navigation tabs (vertical list, icon + label):
  * Overview (grid icon)
  * Groups (users icon)
  * Readiness (check-circle icon)
  * Alerts (bell icon, with red badge count)
  * Live (radio icon, pulses when activity is happening)
  * Assignments (clipboard icon)
* Divider
* Settings link at bottom

On mobile (< 768px), the sidebar collapses into a bottom tab bar with 5 icons: Overview, Groups, Alerts, Live, More.

**Main content area - three zones stacked vertically:**

#### Zone 1: Summary Cards (top strip, 4 cards in a row)

Four metric cards, each ~150px wide, spanning the top:

| Card | Label | Value | Color Logic |
|------|-------|-------|-------------|
| Class Pulse | "On Track" | "72%" | Green if > 70%, amber 40-70%, red < 40% |
| Active Alerts | "Needs Attention" | "3 students" | Red if > 0, green if 0 |
| Common Misconception | "Top Issue" | "Distributes to first term only (5 students)" | Always amber (informational) |
| Next Topic Readiness | "Ready for Combining Like Terms" | "19/28 ready" | Green if > 80%, amber 50-80%, red < 50% |

Each card is clickable: Pulse goes to overview scroll, Alerts to `/alerts`, Common Misconception to `/groups`, Readiness to `/readiness`.

#### Zone 2: Heat Map (main area)

A matrix visualization: **rows = students (sorted by urgency), columns = 5 concepts**.

**Dimensions:** 28-35 rows x 5 columns. Each cell is ~40px x 40px on desktop.

**Column headers:** The 5 concepts in prerequisite order, left to right:
1. Integer & Sign Ops
2. Order of Operations
3. Distributive Property
4. Combining Like Terms
5. Solving Linear Equations

**Row structure (each row = one student):**

* Student name (left-aligned, 120px column, truncated with ellipsis if needed)
* 5 mastery cells
* Alert icon in a 6th column if student has any active alert (small red/amber dot)

**Cell color encoding (4 states):**

| Mastery Range | Color | Hex | Label |
|---------------|-------|-----|-------|
| >= 0.85 | Green | `#22C55E` | Mastered |
| 0.60 - 0.84 | Blue | `#3B82F6` | Progressing |
| 0.35 - 0.59 | Amber | `#F59E0B` | Struggling |
| < 0.35 | Red | `#EF4444` | Critical |

**Misconception overlay:** When a student has an active misconception on a concept, the cell gets a small triangle badge in the top-right corner (dark version of the cell's color). This lets teachers see at a glance which cells have mastery issues caused by a specific identified misconception vs. general low performance.

**Row sorting:** Critical students (with cascade risks or multiple red cells) sort to the top. Within the same urgency tier, alphabetical. A dropdown toggle allows switching to alphabetical-only sorting.

**Interactions:**

* Hover over any cell: popover shows mastery percentage, active misconception name (if any), and last response timestamp. Example: "Maria - Distributive Property: 31% mastery. Active misconception: Only distributes to first term. Last seen: 2 hours ago."
* Click any cell: navigates to `/students/:id` with the relevant concept section expanded.
* Click student name: navigates to `/students/:id`.

#### Zone 3: Quick Groups Preview (below heat map)

A horizontally scrollable row of group summary cards. Each card shows:

* Misconception name (bold)
* Student count badge
* First 3-4 student names (truncated)
* "View Group" link

These cards preview the `/groups` page. Maximum 5 cards shown; if more groups exist, show "View all N groups" link.

### 4. Misconception Groups (`/classrooms/:id/groups`)

Teachers use this to plan small-group instruction.

**Layout:** Vertical list of group cards, sorted by group size (largest first).

Each group card:

```text
┌──────────────────────────────────────────────────────────────┐
│ ● dist_first_term_only                                       │
│   "Only distributes to first term"                           │
│   5 students                                      [Concept: Distributive Property] │
│                                                              │
│   Students: Maria, James, Aiden, Sofia, Liam                │
│                                                              │
│   Evidence (most recent):                                    │
│   ┌─────────────────────────────────────────────┐            │
│   │ Maria  |  Expand: 2(x + 3)                  │            │
│   │        |  Wrote: 2x + 3     Correct: 2x + 6 │            │
│   │ James  |  Expand: 4(y + 5)                   │            │
│   │        |  Wrote: 4y + 5     Correct: 4y + 20 │            │
│   └─────────────────────────────────────────────┘            │
│                                                              │
│   Suggested intervention:                                    │
│   "Small-group mini-lesson on distribution. Use area model   │
│    visual: draw a rectangle with width 2 and length (x + 3),│
│    show both parts get multiplied."                          │
│                                                              │
│   [Assign Practice Problems]  [Mark as Addressed]            │
└──────────────────────────────────────────────────────────────┘
```

**Interactions:**

* Click any student name to navigate to their detail page
* "Assign Practice Problems" creates a targeted assignment for just those students, pre-populated with problems that exercise the relevant misconception
* "Mark as Addressed" dismisses the group (it will reappear if students continue making the same error)
* Filter dropdown at the top: "All Concepts" / specific concept name
* Groups with only 1 student appear in a collapsed "Individual Issues" section at the bottom

### 5. Class Readiness (`/classrooms/:id/readiness`)

Answers: "Is my class ready for the next topic?"

**Layout:**

Top: Concept selector dropdown. Default is the next concept in the prerequisite chain (inferred from curriculum schedule or the concept after the most-recently-assigned one).

Below the selector, a two-column layout:

**Left column (60%):** "Ready" student list

* Green check icon per student
* Shows their prerequisite mastery level

**Right column (40%):** "Not Ready" student list (visually distinct, light red background)

* Each entry shows:
  * Student name
  * Blocking concept and mastery level
  * Specific misconception causing the block
  * Suggested remediation action

**Bottom:** Summary bar

* "19/28 students are ready for Combining Like Terms"
* "9 students need prerequisite remediation"
* "Assign remediation problems to not-ready students" button

### 6. Alerts (`/classrooms/:id/alerts`)

Surfaces everything that needs attention. modeled after PagerDuty's triage interface.

**Layout:** Vertical list of alert cards, grouped by severity.

**Severity tiers:**

| Tier | Label | Icon | Color | Alert Types |
|------|-------|------|-------|-------------|
| Critical | "Act Now" | Red circle with ! | `#EF4444` bg, `#FEE2E2` card | Cascade risk (Level 1 gap blocking all downstream) |
| Warning | "Monitor" | Amber triangle with ! | `#F59E0B` bg, `#FEF3C7` card | Persistent misconception (3+ occurrences after alert) |
| Info | "Heads Up" | Blue info circle | `#3B82F6` bg, `#DBEAFE` card | Prerequisite gap for next topic |

**Alert card structure:**

```text
┌─ CRITICAL ──────────────────────────────────────────────────┐
│ 🔴 Cascade Risk: James                                      │
│                                                              │
│ James's Integer & Sign Operations mastery is 0.38.           │
│ This is the foundation for ALL downstream concepts.          │
│ He cannot meaningfully progress until this is addressed.     │
│ Blocks: Order of Operations, Distributive Property,          │
│         Combining Like Terms, Solving Linear Equations       │
│                                                              │
│ Recommended: 5 targeted sign operation problems with         │
│ concrete number line scaffolding.                            │
│                                                              │
│ [View Student]  [Assign Remediation]  [Acknowledge]          │
│                                              Created: Mar 20 │
└──────────────────────────────────────────────────────────────┘
```

**Alert throttling rules (to prevent fatigue):**

* Maximum 5 active alerts per class at any time
* One alert per student per type (no duplicates)
* When 5+ students share the same misconception alert, bundle into a single group alert: "5 students have persistent dist_first_term_only" with a link to the group
* Prerequisite gap alerts auto-resolve when the student's mastery rises above threshold
* Persistent misconception alerts auto-resolve when the student gives 2 correct responses in a row on that concept
* Cascade risk alerts require manual acknowledgment (they're too important to auto-dismiss)

**Also displayed:** A persistent alert badge on the sidebar "Alerts" tab showing the count. If there are critical alerts, the badge is red. If only warnings/info, it's amber.

### 7. Live Classroom Feed (`/classrooms/:id/live`)

During-class view. Optimized for mobile. Real-time via SSE.

**Layout:** Single-column feed, newest at top. No heat map on this view (too dense for mobile).

**Top bar:** Class pulse - a single horizontal progress bar showing the percentage of students on track for the current assignment. Updates in real time. Green/amber/red segments.

**Feed entries** (appear as students submit responses):

```text
┌──────────────────────────────────────────────────────────────┐
│ Maria  ·  2 min ago                                   [❌]   │
│ Expand: 2(x + 3)                                            │
│ Wrote: 2x + 3                                               │
│ Misconception: Only distributes to first term                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ James  ·  3 min ago                                   [✅]   │
│ Evaluate: 2 + 3 × 4                                         │
│ Wrote: 14                                                    │
└──────────────────────────────────────────────────────────────┘
```

Correct answers show with a green check and minimal detail (no expansion needed). Wrong answers show with a red X, the student's response, and the detected misconception. This lets the teacher walk over to Maria immediately.

**Interactions:**

* Tap student name to open a bottom sheet (mobile) or slide-over panel (desktop) with the student's current session summary: problems attempted, correct count, active misconceptions
* Filter toggle: "Show all" / "Show errors only" (errors-only is default during class)
* Scroll down to see earlier submissions

**Real-time behavior:** New entries slide in at the top with a subtle animation. A sound/vibration option (off by default) can be enabled in settings for wrong answers on critical concepts.

### 8. Assignment Management (`/classrooms/:id/assignments`)

**Layout:** List of assignments, most recent first.

Each assignment row:

* Assignment name / problem set title
* Date assigned
* Completion: "24/28 submitted"
* Accuracy: "71% correct" (overall class)
* Status badge: Active (green), Completed (gray), Draft (amber)

**Interactions:**

* Click to view results for that assignment
* "New Assignment" button opens a creation form
  * Select problems from the problem bank (filterable by concept, difficulty)
  * Or "Auto-generate" based on class needs (system picks problems targeting current gaps)
  * Set due date
  * Generate class code or share link for students

### 9. Assignment Results (`/classrooms/:id/assignments/:aid`)

**Layout:** Two sections.

**Top section:** Summary stats for this assignment.

* Problems count, submission count, average accuracy
* Misconception breakdown: pie chart or horizontal bar chart showing which misconceptions appeared and how frequently

**Bottom section:** Per-student results table.

| Student | P1 | P2 | P3 | P4 | P5 | Score | Misconceptions |
|---------|----|----|----|----|-------|-------|----------------|
| Maria | ✅ | ❌ | ✅ | ❌ | ✅ | 3/5 | dist_first_term_only (2x) |
| James | ✅ | ✅ | ✅ | ✅ | ❌ | 4/5 | leq_reverse_operation |

Click any ❌ cell to see: the problem, what the student wrote, and the detected misconception.

### 10. Student Detail (`/students/:id`)

The deep-dive view for planning periods or parent conferences.

**Layout:** Vertical page with four collapsible sections.

#### Section 1: Action Card (always visible, top of page)

A highlighted card with a light indigo background:

```text
┌──────────────────────────────────────────────────────────────┐
│  Recommended Next Step for Maria                             │
│                                                              │
│  Focus: Distributive Property (31% mastery)                  │
│  Issue: Only distributes to first term (4 occurrences)       │
│                                                              │
│  Assign 3 distribution problems with visual scaffolding.     │
│  Focus on negative factor distribution.                      │
│                                                              │
│  [Assign Recommended Problems]  [Dismiss]                    │
└──────────────────────────────────────────────────────────────┘
```

#### Section 2: Mastery Snapshot (5 horizontal bars)

Five horizontal progress bars, one per concept, in prerequisite order top to bottom:

```text
Integer & Sign Ops     ████████████████████░░  92%  ✅ Mastered
Order of Operations    █████████████████░░░░░  88%  ✅ Mastered
Distributive Property  ██████░░░░░░░░░░░░░░░  31%  🔴 Critical
Combining Like Terms   ░░░░░░░░░░░░░░░░░░░░░  --   ⬜ Not Started
Solving Linear Eqs     ░░░░░░░░░░░░░░░░░░░░░  --   ⬜ Not Started
```

Each bar is colored with the 4-state palette. Concepts that have prerequisites not yet mastered show as "Blocked" in gray with a lock icon.

#### Section 3: Active Misconceptions (collapsible, open by default)

A list of misconception evidence cards. Each card:

```text
┌──────────────────────────────────────────────────────────────┐
│ dist_first_term_only - "Only distributes to first term"      │
│ Seen 4 times  ·  Last: March 20  ·  Concept: Distributive   │
│                                                              │
│ Evidence:                                                    │
│   Mar 20  Expand: 2(x + 3)    Wrote: 2x + 3    ✗           │
│   Mar 18  Expand: 4(y + 5)    Wrote: 4y + 5    ✗           │
│   Mar 15  Expand: 3(m + 7)    Wrote: 3m + 7    ✗           │
│   Mar 12  Expand: 5(x + 1)    Wrote: 5x + 1    ✗           │
│                                                              │
│ Persistence: HIGH (4 occurrences across 8 days)              │
└──────────────────────────────────────────────────────────────┘
```

The evidence column uses a two-column layout: left shows the problem, right shows student work vs. correct answer. This side-by-side format lets the teacher see the pattern instantly.

#### Section 4: Mastery Trajectory (collapsible, collapsed by default)

A line chart (Recharts or Nivo) showing mastery probability over time for each concept as a separate colored line. X-axis: dates. Y-axis: 0-100% mastery. A horizontal dashed line at 85% marks the mastery threshold. Hovering a data point shows the date, mastery value, and which problems were submitted that day.

Below the chart: a chronological response log (paginated, 20 per page) showing every response this student has ever given, with timestamps, problems, answers, and classification results.

**Footer:** "Export PDF" button generates a printable summary: mastery snapshot + active misconceptions + trajectory chart + last 10 responses. Formatted for letter-size paper, suitable for parent conferences.

### 11. Student-Facing Interface (`/join`, `/work/:aid`, `/work/:aid/done`)

Completely separate from the teacher dashboard. No shared navigation.

#### Join Screen (`/join`)

* Large centered input field: "Enter your class code"
* Below: "Your name" input field (first name + last initial is sufficient)
* "Join" button
* No password, no account creation, no login. Students type a code and a name.
* The class code is a 6-character alphanumeric string the teacher gets from the assignment creation flow.

#### Work Screen (`/work/:aid`)

Absolutely minimal. One problem at a time.

```text
┌──────────────────────────────────────────────────────────────┐
│   Problem 3 of 10                                            │
│                                                              │
│   Expand: 2(x + 3)                                          │
│                                                              │
│   ┌──────────────────────────────────────────────┐           │
│   │  Your answer: [                            ] │           │
│   └──────────────────────────────────────────────┘           │
│                                                              │
│   [ x ]  [ + ]  [ - ]  [ = ]  [ ^ ]  [ ( ]  [ ) ]          │
│                                                              │
│              [  Submit  ]                                     │
│                                                              │
│   ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░  3/10                    │
└──────────────────────────────────────────────────────────────┘
```

**Math input approach:** Plain text input field with a symbol toolbar below. The toolbar has buttons for common math symbols that students might not know how to type: `x`, `+`, `-`, `=`, `^`, `(`, `)`. Tapping a symbol button inserts it at the cursor position. Below the text field, a live KaTeX preview renders what the student is typing so they can verify it looks right: e.g., typing `x^2 + 3x` shows x² + 3x in rendered math.

**Why plain text, not a structured editor:** The NLP classifier is trained on plain text input. The messy, imperfect way students type carries its own diagnostic signal. A structured editor would normalize away information we want. MathQuill and MathLive add complexity, increase bundle size, and create accessibility issues on older Chromebooks. Plain text with a symbol helper is sufficient for algebraic expressions at this level.

**No feedback shown.** After submitting, the student moves to the next problem. No "correct" or "incorrect" indication. This is a diagnostic tool, not a quiz. Showing results would change student behavior (guessing, anxiety, peer comparison).

**Progress bar** at the bottom shows how many problems are done.

#### Done Screen (`/work/:aid/done`)

* "All done! You can close this tab."
* A simple check mark illustration
* No score, no results, no recap

### 12. Settings (`/settings`)

Standard settings page:

* Account: name, email, password change
* Classrooms: manage class rosters (add/remove students), view/regenerate class codes
* Schedule: set which concept the class is working on now, and the next planned concept (used by the readiness check and prerequisite gap alerts)
* Preferences: notification preferences, alert sound toggle, timezone

---

## Reusable Component Inventory

| Component | Description | Used On |
|-----------|-------------|---------|
| `MasteryCell` | 40x40px colored square with optional misconception badge triangle. 4-state color. Hover shows popover. | Heat map |
| `MasteryBar` | Horizontal progress bar with percentage label and status text. 4-state color. | Student detail |
| `SummaryCard` | Metric card with label, value, and color-coded indicator. Clickable. | Class overview top strip |
| `AlertCard` | Severity-colored card with icon, message, student name, recommended action, and action buttons. | Alerts page, sidebar badge |
| `GroupCard` | Misconception name, student list, evidence table, suggested intervention, action buttons. | Groups page |
| `StudentRow` | Table row: name + 5 mastery cells + alert indicator. Clickable. | Heat map |
| `EvidenceRow` | Problem, student response, correct answer in side-by-side layout. Marked ✓ or ✗. | Student detail, groups, assignment results |
| `MisconceptionBadge` | Compact tag showing misconception label. Color matches concept. | Groups, student detail, alerts |
| `LiveFeedEntry` | Student name, timestamp, problem, response, misconception (if wrong). Correct/incorrect icon. | Live feed |
| `ClassPulseBar` | Horizontal bar with green/amber/red segments proportional to class state. | Class overview, live feed, class selector cards |
| `ReadinessRow` | Student name, mastery level, blocker description. Green (ready) or red (not ready) styling. | Readiness page |
| `ActionCard` | Highlighted recommendation card. Light indigo background. Action buttons. | Student detail top |
| `ConceptSelector` | Dropdown selecting a concept. Shows prerequisite chain order. | Readiness page |
| `ClassSwitcher` | Dropdown at top of sidebar for switching between classes. | Sidebar |
| `BottomSheet` | Mobile slide-up panel for student quick view. | Mobile live feed |
| `SymbolToolbar` | Row of math symbol buttons that insert characters into a text input. | Student work interface |
| `KaTeXPreview` | Live-rendered math preview of a text input. | Student work interface |

---

## Interaction Patterns

### Navigation

* **Sidebar** (desktop): persistent left sidebar, 240px wide, with navigation tabs. Active tab highlighted with indigo left border and indigo text. Inactive tabs in slate gray.
* **Bottom tab bar** (mobile < 768px): 5 icons at screen bottom. Active icon filled, inactive outlined.
* **Class switching**: dropdown at top of sidebar. Selecting a new class navigates to that class's overview.
* **Breadcrumb**: shown below the top bar on student detail and assignment results pages. Example: "Period 2 > Maria" or "Period 2 > Assignment 3".

### Progressive Disclosure

* **Heat map cells**: hover for popover summary, click for full student detail. Two levels of disclosure only.
* **Student detail sections**: Action card always visible. Misconceptions open by default. Trajectory collapsed by default. Click section header to expand/collapse.
* **Groups page**: evidence table shows 2 rows by default, "Show all evidence" expands.
* **Alert cards**: show message and recommendation by default. "View Student" navigates away only when the teacher wants the full picture.

### Drill-Down Path

Every piece of data is at most 2 clicks from the overview:

| Starting Point | 1 Click | 2 Clicks |
|---------------|---------|----------|
| Heat map cell | Student detail with concept expanded | - |
| Group card student name | Student detail | Response history |
| Alert "View Student" | Student detail | - |
| Summary card (Alerts) | Alerts page | Student detail |
| Summary card (Top Issue) | Groups page | Student detail via student name |

### Filtering

* **Groups page**: filter by concept (dropdown)
* **Alerts page**: filter by severity (all / critical / warning / info), filter by concept
* **Live feed**: toggle "All responses" / "Errors only"
* **Heat map**: no filtering (it shows everything at once; sorting handles prioritization)

### Real-Time Updates

* Live feed receives events via **Server-Sent Events (SSE)**, not WebSockets. SSE works through school firewalls and proxy servers that often block WebSocket connections. The connection is a simple GET request that streams events.
* Events: `student_response` (new submission), `mastery_update` (BKT recalculation), `alert_created`, `alert_resolved`
* Heat map cells update in place when a `mastery_update` arrives (cell color transition animation, 300ms)
* Live feed entries slide in with a 200ms ease-in animation

### Export

* Student detail: "Export PDF" button renders a printable summary. Uses `@react-pdf/renderer` or browser print CSS.
* Groups page: "Print Groupings" formats the current groups and interventions for printing (e.g., to post on a planning board).

---

## Visual Design Direction

### Color Palette

**Primary brand color:** Indigo
```text
Indigo-600:  #4F46E5   (primary buttons, active nav, links)
Indigo-500:  #6366F1   (hover states)
Indigo-100:  #E0E7FF   (action card backgrounds, selected states)
Indigo-50:   #EEF2FF   (light highlights)
```

**Neutral scale (Slate):**
```text
Slate-900:   #0F172A   (primary text)
Slate-700:   #334155   (secondary text)
Slate-500:   #64748B   (placeholder text, metadata)
Slate-300:   #CBD5E1   (borders)
Slate-100:   #F1F5F9   (backgrounds, alternating rows)
Slate-50:    #F8FAFC   (page background)
White:       #FFFFFF   (cards, inputs)
```

**Semantic status colors (mastery states):**
```text
Green-500:   #22C55E   (mastered, on track, correct)
Blue-500:    #3B82F6   (progressing, info alerts)
Amber-500:   #F59E0B   (struggling, warning alerts)
Red-500:     #EF4444   (critical, cascade risk, incorrect)
```

**Semantic backgrounds (for alert cards, status sections):**
```text
Green-50:    #F0FDF4
Blue-50:     #EFF6FF
Amber-50:    #FFFBEB
Red-50:      #FEF2F2
```

### Color-Blind Safe Alternative Palette (Heat Map)

For users with color vision deficiency, provide a toggle in settings to switch the heat map to a diverging blue-to-orange scale:

```text
Mastered:    #2166AC   (dark blue)
Progressing: #67A9CF   (light blue)
Struggling:  #F4A582   (light orange)
Critical:    #B2182B   (dark red-brown)
```

This palette is distinguishable by users with protanopia, deuteranopia, and tritanopia. Additionally, every color-coded element includes a text label or icon (check, arrow, warning, X) so that color is never the sole information channel.

### Typography

**Font stack:** System fonts for performance and familiarity.

```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
             'Helvetica Neue', Arial, sans-serif;
```

**Scale:**

| Use | Size | Weight | Line Height |
|-----|------|--------|-------------|
| Page title | 24px / 1.5rem | 700 (bold) | 32px |
| Section heading | 20px / 1.25rem | 600 (semibold) | 28px |
| Card title | 16px / 1rem | 600 | 24px |
| Body text | 14px / 0.875rem | 400 (regular) | 20px |
| Small text / metadata | 12px / 0.75rem | 400 | 16px |
| Heat map cell label | 11px / 0.6875rem | 500 (medium) | 14px |

**Math rendering:** KaTeX for rendering mathematical expressions in problem displays and student answers. Load only the fonts needed for basic algebra (no CJK, no script variants).

### Spacing System

8px base unit. All spacing values are multiples of 8:

```text
4px   (xs)   - tight padding inside badges
8px   (sm)   - inner padding in compact elements
16px  (md)   - card padding, input padding
24px  (lg)   - section margins
32px  (xl)   - page margins on desktop
48px  (2xl)  - spacing between major page sections
```

### Elevation / Shadows

```text
Cards:     0 1px 3px rgba(0,0,0,0.1)
Popovers:  0 4px 12px rgba(0,0,0,0.15)
Modals:    0 8px 24px rgba(0,0,0,0.2)
```

### Anti-Patterns to Avoid

* No gamification elements (badges, points, streaks, leaderboards)
* No cartoon mascots or illustrations
* No confetti or celebration animations
* No dark mode (unnecessary complexity for V1; teachers use this in well-lit classrooms)
* No gradient backgrounds or bright accent colors
* No rounded avatar circles (students don't upload photos; use initials in a neutral circle)

---

## Accessibility Plan

### WCAG 2.1 AA Compliance

| Requirement | Implementation |
|-------------|----------------|
| Color contrast | All text meets 4.5:1 ratio against its background. Large text (18px+) meets 3:1. Tested with axe-core. |
| Non-color indicators | Every status uses color + icon + text label. Heat map cells include tooltip text. Alerts use icon shapes (circle, triangle, info-i) in addition to color. |
| Keyboard navigation | All interactive elements focusable via Tab. Heat map cells navigable with arrow keys. Enter/Space activates buttons. Escape closes popovers and modals. |
| Screen readers | Heat map rendered as an HTML `<table>` with `<th>` headers and `aria-label` on each cell ("Maria, Distributive Property, 31% mastery, critical"). Live feed uses `aria-live="polite"` region. Alert counts announced via `aria-live`. |
| Focus indicators | Visible focus ring (2px indigo outline, 2px offset) on all interactive elements. Never removed. |
| Motion | Animations respect `prefers-reduced-motion` media query. When enabled, all transitions are instant. |
| Touch targets | All interactive elements are minimum 44x44px (48px preferred). Mobile bottom tab bar buttons are 48x48px. |

### Multilingual Support

* V1: English + Spanish
* Approximately 250-300 translatable strings
* Use `next-intl` for React internationalization
* Math content (problem text, student responses) is language-neutral and not translated
* Misconception labels and descriptions exist in both languages in the knowledge graph data
* Date formatting and number formatting use `Intl` browser APIs

### Low-Bandwidth Optimization

* Service Worker caches the app shell and static assets (offline-first for the PWA)
* API responses include ETags for cache validation
* Heat map data is a single JSON payload (~4KB for 35 students x 5 concepts)
* Images: none in the core UI (all data is text and colored cells)
* Target: first meaningful paint under 3 seconds on a 3G connection
* Lazy-load the trajectory chart component (Recharts is ~40KB gzipped)

---

## Tech Stack Recommendation

### Framework: Next.js 14+ (App Router)

| Factor | Recommendation | Rationale |
|--------|---------------|-----------|
| Framework | Next.js (React) | Largest ecosystem, best hiring pool, hybrid rendering (SSR for initial load, client for interactivity). App Router provides layouts and loading states. |
| Component library | shadcn/ui + Radix primitives | Free, accessible by default (Radix handles ARIA), copy-paste model means no dependency lock-in. Tailwind-based styling. |
| Styling | Tailwind CSS | Utility-first, excellent Pui performance (purges unused CSS), consistent spacing/color system, great for responsive design. |
| Charts | Recharts (line charts/bars) + custom SVG (heat map) | Recharts is React-native, declarative, lightweight. The heat map is simple enough (colored `<div>` grid or SVG `<rect>`) to build custom rather than pulling in D3. |
| Math rendering | KaTeX | Faster than MathJax, smaller bundle, server-side rendering support. |
| Real-time | SSE via `EventSource` API | Works through school firewalls (it's a regular HTTP GET). No WebSocket complexity. FastAPI supports SSE natively with `StreamingResponse`. |
| Internationalization | next-intl | Built for Next.js App Router. Type-safe message keys. |
| Offline/PWA | Serwist (Next.js-native service worker) | Handles app shell caching, offline fallback, and background sync. |
| PDF export | Browser print CSS + `@media print` | Simpler than a rendering library. The student detail page gets a print stylesheet. |
| State management | React Context + SWR | SWR for data fetching with cache/revalidation. Context for minimal global state (current class, user). No Redux needed at this scale. |

### Bundle Budget

| Chunk | Target Size (gzipped) |
|-------|-----------------------|
| App shell (Next.js + React + routing) | < 80KB |
| shadcn/ui components (used subset) | < 20KB |
| Tailwind CSS (purged) | < 15KB |
| Recharts (lazy-loaded) | < 40KB |
| KaTeX (lazy-loaded, algebra subset) | < 30KB |
| Total initial load | < 120KB |

### Deployment

* Frontend: Vercel (free tier handles the traffic for a pilot). Automatic preview deployments per PR.
* Backend: FastAPI on a $5-10/mo VPS (Railway, Render, or Fly.io). SQLite for pilot; migrate to PostgreSQL if needed.
* Domain: custom domain (e.g., `app.eddiagnose.com`)
* TLS: automatic via Vercel and LetsEncrypt

### Development Setup

```text
/web
  /app              ← Next.js App Router pages
    /classrooms
    /students
    /join
    /work
    /settings
  /components       ← Reusable UI components (from inventory above)
  /lib              ← API client, SSE hook, utility functions
  /styles           ← Tailwind config, global styles, print stylesheet
  /public           ← Static assets
  /messages         ← i18n translation files (en.json, es.json)
  next.config.js
  tailwind.config.js
  package.json
```

---

## References

### Ed-Tech Products Studied

* Google Classroom - class-first navigation, minimal design, teacher familiarity baseline
* Canvas (Instructure) - analytics dashboard, mastery gradebook patterns
* ASSISTments (Worcester Polytechnic) - teacher reports, item-level analysis, misconception-aware reporting
* Eedi - misconception detection quiz platform, teacher insights on specific errors
* Khan Academy - teacher dashboard with mastery tracking, student progress views
* ALEKS - adaptive learning with knowledge space mastery visualization
* Desmos - clean math rendering, minimal professional design language

### Design Systems and Patterns

* shadcn/ui - accessible React components built on Radix primitives
* NNGroup - progressive disclosure principles (Jakob Nielsen, 2006)
* PagerDuty incident management UI - alert triage patterns, severity tiering, acknowledgment workflows
* Grafana dashboards - monitoring entity matrices, color-coded status grids

### Alert Fatigue Research

* Clinical alert fatigue studies show that clinicians respond to < 30% of alerts when volume exceeds 10-15 per day. Teacher dashboards should target 3-5 actionable alerts per class maximum to maintain attention and trust.
* Layered notification architecture (banner for critical, inline for warning, badge for info) reduces notification blindness compared to a single channel.

### Accessibility

* WCAG 2.1 AA guidelines (W3C)
* Color-blind safe palettes: ColorBrewer diverging scales (Cynthia Brewer, Penn State)
* Radix UI accessibility primitives (built-in ARIA patterns)

---

## Open Questions

Decisions that should involve teacher feedback before finalizing:

1. **Sorting preference:** Do teachers prefer the heat map sorted by urgency (struggling students first) or alphabetically (familiar order)?
2. **Alert channel:** Should critical alerts also send an email digest, or is in-app sufficient?
3. **Student authentication:** Is class code + first name sufficient, or do schools need something stronger?
4. **Intervention library depth:** Should suggested interventions include links to external resources (YouTube videos, worksheets), or stay as text descriptions?
5. **Print formatting:** What do teachers actually bring to parent conferences? A single page per student, or a class-wide summary?
6. **Curriculum schedule:** Do teachers want to manually set "next topic" or should the system infer it from the prerequisite chain and current class mastery?
7. **Co-teaching:** Do V1 classrooms need multiple teacher accounts, or is single-teacher sufficient for pilot?
8. **Historical retention:** How far back should student data persist? Current semester only, or across school years?
