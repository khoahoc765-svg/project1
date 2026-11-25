# Week 1 Presentation: Problem Definition & Dataset Survey
**Project:** Automated Food & Produce Freshness Detection System using YOLOv11  
**Student:** Trần Lê Xuân Mai (`24020011`)

---

## Slide 1: What, When, Why, Who, How Much (Week 1)
- **WHAT:** Project Charter, Problem Definition (FIFO Warehouse Sorting), and Roboflow Dataset Survey (`>21,000` images).
- **WHEN:** Week 1 Milestone (Completed on schedule: 15h planned / 15h spent).
- **WHY:** Manual produce inspection is a major bottleneck in quick-commerce warehousing; automated computer vision prevents spoilage propagation.
- **WHO:** Trần Lê Xuân Mai (`24020011`) — End-to-End AI Engineer (Data Survey, SWOT Analysis, Scope Definition).
- **HOW MUCH:** Planned: `15 hours` | Spent: `15 hours` | Gap: `0 hours`.

---

## Slide 2: Problem Formulation & Strategic PoC Scope (6 Classes)
- **Dual Computer Vision Task:**
  1. **Object Detection (`Bounding Box`):** Locate each fruit/vegetable on the conveyor belt.
  2. **Freshness Classification:** Classify state into `Fresh` vs. `Rotten`.
- **6 Target Classes (3 Visual Archetypes × 2 States):**
  - `Fresh Apple` & `Rotten Apple` *(Glossy skin reflection archetype)*
  - `Fresh Banana` & `Rotten Banana` *(Dark-spot bruising & pigmentation archetype)*
  - `Fresh Bell Pepper` & `Rotten Bell Pepper` *(Multi-lobed geometry & stem archetype)*

---

## Slide 3: SWOT Analysis & Initial Risks
- **Strengths:** Large-scale pre-annotated Roboflow dataset (`freshness-detection-rhrze`).
- **Weaknesses:** Raw data contains noisy synthetic augmentations and empty annotation files.
- **Opportunities:** Seamless Transfer Learning with modern lightweight YOLO architectures.
- **Threats:** Complex warehouse lighting variations and overlapping fruits on conveyor belts.
